#include <Qx/render.hpp>

#include <Qx/bvh.hpp>
#include <Qx/random.hpp>

#include <cmath>
#include <cstdint>

namespace Qx {

namespace {

class RendererImpl final : public Renderer
{
public:
  RendererImpl(const Triangle* triangles, const HostBvh& bvh)
    : m_triangles(permute(bvh, triangles))
    , m_bvh(bvh)
  {}

  void render(const Vec3& eye, const Vec3& dir, const Vec3& right, const Vec3& up, int width, int height, float* rgb)
    override
  {
#pragma omp parallel for
    for (int y = 0; y < height; y++) {

      for (int x = 0; x < width; x++) {

        const Vec3 result = renderPixel(m_triangles, m_bvh, eye, dir, right, up, x, y, width, height);

        const int offset = 3 * ((y * width) + x);

        rgb[offset + 0] = result[0];
        rgb[offset + 1] = result[1];
        rgb[offset + 2] = result[2];
      }
    }
  }

private:
  Array<Triangle, false> m_triangles;

  const HostBvh& m_bvh;
};

} // namespace

std::unique_ptr<Renderer>
Renderer::create(const Triangle* triangles, const HostBvh& bvh)
{
  return std::unique_ptr<Renderer>(new RendererImpl(triangles, bvh));
}

} // namespace Qx
