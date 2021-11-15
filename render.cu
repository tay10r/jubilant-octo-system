#include "render.hpp"

#include "common.hpp"
#include "random.hpp"

#include <cmath>
#include <cstdint>

namespace {

Vec3
on_miss(const Ray& ray)
{
  const Vec3 lo(1.0f, 1.0f, 1.0f);
  const Vec3 hi(0.5f, 0.7f, 1.0f);
  const Vec3 up(0.0f, 1.0f, 0.0f);
  const float level = (dot(up, ray.dir) + 1.0f) * 0.5f;
  return lo + ((hi - lo) * level);
}

template<typename Rng>
Vec3
sample_hemisphere(Rng& rng)
{
  const float azimuth = random_float(rng) * M_PI * 2;

  const float y = random_float(rng);

  const float sin_elevation = std::sqrt(1 - (y * y));

  const float x = sin_elevation * std::cos(azimuth);
  const float z = sin_elevation * std::sin(azimuth);

  return Vec3(x, y, z);
}

template<typename Rng>
Vec3
sample_hemisphere(const Vec3& normal, const Vec3& tangent, Rng& rng)
{
  const Vec3 dir = sample_hemisphere(rng);

  const Vec3 bitangent = cross(normal, tangent);

  return Vec3((dir[0] * bitangent[0]) + (dir[1] * normal[0]) + (dir[2] * tangent[0]),
              (dir[0] * bitangent[1]) + (dir[1] * normal[1]) + (dir[2] * tangent[1]),
              (dir[0] * bitangent[2]) + (dir[1] * normal[2]) + (dir[2] * tangent[2]));
}

template<typename Rng>
Vec3
trace(const Triangle* triangles, const Bvh& bvh, Ray& ray, Rng& rng)
{
  int depth = 0;

  const int max_depth = 5;

  Vec3 color(1, 1, 1);

  while (depth < max_depth) {

    auto hit = bvh.traverse(ray, triangles);
    if (!hit)
      return color * on_miss(ray);

    const Vec3 albedo(0.8, 0.8, 0.8);

    color = color * albedo;

    const Vec3 hitPos = ray.org + (ray.dir * hit.intersection.distance());

    const Triangle& tri = triangles[hit.prim_index];

    const Vec3 e0 = tri.p0 - tri.p1;
    const Vec3 e1 = tri.p2 - tri.p0;

    const Vec3 hitNorm = normalize(cross(e1, e0));

    const Vec3 secondRayDir = sample_hemisphere(hitNorm, normalize(e0), rng);

    const float shadowBias = 0.0001;

    ray = Ray(hitPos, secondRayDir, shadowBias);

    depth++;
  }

  return Vec3(0, 0, 0);
}

void
renderPixel(const Vec3& eye,
            const Vec3& dir,
            const Vec3& right,
            const Vec3& up,
            const Bvh& bvh,
            const Triangle* triangles,
            float rcp_width,
            float rcp_height,
            float* rgb)
{
    const int x = threadIdx.x + (blockIdx.x * blockDim.x);
    const int y = threadIdx.y + (blockIdx.y * blockDim.y);

    const float u = (x + 0.0) * rcp_width;
    const float v = (y + 0.0) * rcp_height;

    const int pixelIndex = ((y * blockDim.x) + x) * 3;

    rgb[pixelIndex + 0] = u;
    rgb[pixelIndex + 1] = v;
    rgb[pixelIndex + 2] = 1;
}

} // namespace

void
render(const Vec3& eye,
       const Vec3& dir,
       const Vec3& right,
       const Vec3& up,
       const Bvh& bvh,
       const Triangle* triangles,
       int width,
       int height,
       float* rgb)
{
  const int tx = 8;
  const int ty = 8;

  dim3 blocks(width / tx, height / ty);

  dim3 threads(tx, ty);

  renderPixel << blocks, threads >> (eye, dir, right, up, bvh, triangles, width, height, rgb);

  cudaDeviceSynchronize();
}
