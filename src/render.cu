#include <Qx/render.hpp>

#include "array.cuh"
#include "bvh.cuh"

#include <Qx/bvh.hpp>
#include <Qx/random.hpp>

#include <cmath>
#include <cstdint>
#include <cstdlib>

#include <chrono>
#include <cstdio>

namespace Qx {

namespace {

void
check_cuda(cudaError_t result, const char* func, const char* file, int line)
{
  if (result != cudaSuccess) {
    std::fprintf(stderr, "%s:%s:%d: %s\n", file, func, line, cudaGetErrorString(result));
    cudaDeviceReset();
    std::exit(EXIT_FAILURE);
  }
}

} // namespace

#define checkCudaError(val) check_cuda((val), #val, __FILE__, __LINE__)

namespace {

__global__ void
renderPixelKernel(const Vec3 eye,
            const Vec3 dir,
            const Vec3 right,
            const Vec3 up,
            const Triangle* triangles,
            const Node* nodes,
            int width,
            int height,
            int padded_width,
            Vec3* rgb)
{
  const int x = threadIdx.x + (blockIdx.x * blockDim.x);
  const int y = threadIdx.y + (blockIdx.y * blockDim.y);

  const int pixelIndex = (y * padded_width) + x;

  rgb[pixelIndex] = Renderer::renderPixel<true>(triangles, nodes, eye, dir, right, up, x, y, width, height);
}

class RendererImpl final : public Renderer
{
public:
  RendererImpl(const Triangle* triangles, const HostBvh& bvh)
    : m_triangles(bvh.prim_indices.size())
    , m_bvh(make_device_bvh(bvh))
  {
    Array<Triangle, false> tmp_triangles = permute(bvh, triangles);

    hostToDevice(tmp_triangles, m_triangles);
  }

  void render(const Vec3& eye, const Vec3& dir, const Vec3& right, const Vec3& up, int width, int height, float* rgb)
    override
  {
    const int tx = 8;
    const int ty = 8;

    const int padded_w = ((width + (tx - 1)) / tx) * tx;
    const int padded_h = ((height + (ty - 1)) / ty) * ty;

    DeviceArray<Vec3> device_rgb(padded_w * padded_h);

    checkCudaError(cudaGetLastError());

    dim3 blocks(padded_w / tx, padded_h / ty);

    dim3 threads(tx, ty);

    auto t0 = std::chrono::high_resolution_clock::now();

    renderPixelKernel<<<blocks, threads>>>(eye, dir, right, up, &m_triangles[0], &m_bvh.nodes[0], width, height, padded_w, &device_rgb[0]);

    cudaDeviceSynchronize();

    auto t1 = std::chrono::high_resolution_clock::now();

    auto t = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    printf("frame completed in %d ms\n", int(t));

    checkCudaError(cudaGetLastError());

    HostArray<Vec3> host_rgb(padded_w * padded_h);

    deviceToHost(device_rgb, host_rgb);

    checkCudaError(cudaGetLastError());

    for (int y = 0; y < height; y++) {

      for (int x = 0; x < width; x++) {

        const int src = (y * padded_w) + x;

        const int dst = (y * width) + x;

        rgb[(dst * 3) + 0] = host_rgb[src][0];
        rgb[(dst * 3) + 1] = host_rgb[src][1];
        rgb[(dst * 3) + 2] = host_rgb[src][2];
      }
    }
  }

private:
  DeviceArray<Triangle> m_triangles;

  DeviceBvh m_bvh;
};

} // namespace

std::unique_ptr<Renderer>
Renderer::create(const Triangle* triangles, const HostBvh& bvh)
{
  return std::unique_ptr<Renderer>(new RendererImpl(triangles, bvh));
}

} // namespace Qx
