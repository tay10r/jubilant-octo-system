#pragma once

#include <Qx/cuda_macros.hpp>
#include <Qx/scalar.hpp>

namespace Qx {

struct TriangleIntersection final
{
  float t = infinity();
  float u = 0;
  float v = 0;

  constexpr float __device__ distance() const noexcept { return t; }

  __device__ operator bool() const noexcept { return t != infinity(); }
};

struct Triangle final
{
  using Intersection = TriangleIntersection;

  Vec3 p0;
  Vec3 p1;
  Vec3 p2;

  Triangle() = default;

  constexpr Triangle(const Vec3& p0, const Vec3& p1, const Vec3& p2)
    : p0(p0)
    , p1(p1)
    , p2(p2)
  {}

  Intersection __device__ intersect(Ray& ray) const;
};

inline Triangle::Intersection
Triangle::intersect(Ray& ray) const
{
  auto e1 = p0 - p1;
  auto e2 = p2 - p0;
  auto n = cross(e1, e2);

  auto c = p0 - ray.org;
  auto r = cross(ray.dir, c);
  auto inv_det = 1.0f / dot(n, ray.dir);

  auto u = dot(r, e2) * inv_det;
  auto v = dot(r, e1) * inv_det;
  auto w = 1.0f - u - v;

  // These comparisons are designed to return false
  // when one of t, u, or v is a NaN
  if (u >= 0 && v >= 0 && w >= 0) {
    auto t = dot(n, c) * inv_det;
    if (t >= ray.tmin && t <= ray.tmax) {
      ray.tmax = t;
      return Intersection{ t, u, v };
    }
  }

  return Intersection{};
}

} // namespace Qx
