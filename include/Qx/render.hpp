#pragma once

#include <Qx/array.hpp>
#include <Qx/bvh.hpp>
#include <Qx/cuda_macros.hpp>
#include <Qx/random.hpp>
#include <Qx/ray.hpp>
#include <Qx/triangle.hpp>
#include <Qx/vec3.hpp>

#include <memory>

namespace Qx {

struct Triangle;

class Renderer
{
public:
  static std::unique_ptr<Renderer> create(const Triangle* triangles, const Bvh<false>& bvh);

  virtual ~Renderer() = default;

  virtual void
  render(const Vec3& eye, const Vec3& dir, const Vec3& right, const Vec3& up, int width, int height, float* rgb) = 0;

  template<bool DeviceFlag>
  static Vec3 __device__ renderPixel(const Triangle* triangles,
                                     const Node* nodes,
                                     const Vec3& eye,
                                     const Vec3& dir,
                                     const Vec3& right,
                                     const Vec3& up,
                                     int x,
                                     int y,
                                     int width,
                                     int height)
  {
    Pcg rng((y * width) + x);

    Vec3 hdr_color(0, 0, 0);

    const int spp = 1;

    for (int i = 0; i < spp; i++) {

      const float u = (float(x) + random_float(rng)) / float(width);
      const float v = (float(y) + random_float(rng)) / float(height);

      const float d_x = (u * 2.0f) - 1.0f;
      const float d_y = 1.0f - (v * 2.0f);

      Ray ray(eye, normalize(dir + (d_x * right) + (d_y * up)));

      hdr_color = hdr_color + trace<DeviceFlag>(triangles, nodes, ray, rng);
    }

    const Vec3 ldr_color = hdr_color / (hdr_color + Vec3(1, 1, 1));

    return ldr_color;
  }

protected:
  static Vec3 __device__ on_miss(const Ray& ray)
  {
    const Vec3 lo(1.0f, 1.0f, 1.0f);
    const Vec3 hi(0.5f, 0.7f, 1.0f);
    const Vec3 up(0.0f, 1.0f, 0.0f);
    const float level = (dot(up, ray.dir) + 1.0f) * 0.5f;
    return lo + ((hi - lo) * level);
  }

  template<typename Rng>
  static Vec3 __device__ sample_hemisphere(Rng& rng)
  {
    const float azimuth = random_float(rng) * M_PI * 2;

    const float y = random_float(rng);

    const float sin_elevation = std::sqrt(1 - (y * y));

    const float x = sin_elevation * std::cos(azimuth);
    const float z = sin_elevation * std::sin(azimuth);

    return Vec3(x, y, z);
  }

  template<typename Rng>
  static Vec3 __device__ sample_hemisphere(const Vec3& normal, const Vec3& tangent, Rng& rng)
  {
    const Vec3 dir = sample_hemisphere(rng);

    const Vec3 bitangent = cross(normal, tangent);

    return Vec3((dir[0] * bitangent[0]) + (dir[1] * normal[0]) + (dir[2] * tangent[0]),
                (dir[0] * bitangent[1]) + (dir[1] * normal[1]) + (dir[2] * tangent[1]),
                (dir[0] * bitangent[2]) + (dir[1] * normal[2]) + (dir[2] * tangent[2]));
  }

  template<bool DeviceFlag, typename Rng>
  static Vec3 __device__ trace(const Triangle* triangles, const Node* nodes, Ray& ray, Rng& rng)
  {
    int depth = 0;

    const int max_depth = 6;

    Vec3 color(1, 1, 1);

    while (depth < max_depth) {

      auto hit = Bvh<DeviceFlag>::traverse(ray, triangles, nodes);
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
};

} // namespace Qx
