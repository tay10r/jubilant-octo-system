#pragma once

#ifdef __CUDACC__
#define DEV_FUNC __device
#else
#define DEV_FUNC
#endif

#include "small_stack.hpp"

#include <algorithm>
#include <array>
#include <iostream>
#include <limits>
#include <numeric>
#include <tuple>
#include <utility>
#include <vector>

#include <cassert>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>

struct Vec3
{
  float values[3];

  Vec3() = default;
  Vec3(float x, float y, float z)
    : values{ x, y, z }
  {}
  explicit Vec3(float x)
    : Vec3(x, x, x)
  {}

  float& operator[](int i) { return values[i]; }
  float operator[](int i) const { return values[i]; }
};

inline Vec3
operator+(const Vec3& a, const Vec3& b)
{
  return Vec3(a[0] + b[0], a[1] + b[1], a[2] + b[2]);
}

inline Vec3
operator-(const Vec3& a, const Vec3& b)
{
  return Vec3(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}

inline Vec3
operator*(const Vec3& a, const Vec3& b)
{
  return Vec3(a[0] * b[0], a[1] * b[1], a[2] * b[2]);
}

inline Vec3
operator*(const Vec3& a, float b)
{
  return Vec3(a[0] * b, a[1] * b, a[2] * b);
}

inline Vec3
operator/(const Vec3& a, const Vec3& b)
{
  return Vec3(a[0] / b[0], a[1] / b[1], a[2] / b[2]);
}

inline Vec3
operator*(float a, const Vec3& b)
{
  return b * a;
}

inline Vec3
min(const Vec3& a, const Vec3& b)
{
  return Vec3(std::min(a[0], b[0]), std::min(a[1], b[1]), std::min(a[2], b[2]));
}

inline Vec3
max(const Vec3& a, const Vec3& b)
{
  return Vec3(std::max(a[0], b[0]), std::max(a[1], b[1]), std::max(a[2], b[2]));
}

inline float
dot(const Vec3& a, const Vec3& b)
{
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

inline float
length(const Vec3& a)
{
  return std::sqrt(dot(a, a));
}

inline Vec3
normalize(const Vec3& a)
{
  return a * (1.0f / length(a));
}

inline Vec3
cross(const Vec3& a, const Vec3& b)
{
  return Vec3(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]);
}

struct BBox
{
  Vec3 min, max;

  BBox() = default;
  BBox(const Vec3& min, const Vec3& max)
    : min(min)
    , max(max)
  {}
  explicit BBox(const Vec3& point)
    : BBox(point, point)
  {}

  BBox& extend(const Vec3& point) { return extend(BBox(point)); }

  BBox& extend(const BBox& other)
  {
    min = ::min(min, other.min);
    max = ::max(max, other.max);
    return *this;
  }

  Vec3 diagonal() const { return max - min; }

  int largest_axis() const
  {
    auto d = diagonal();
    int axis = 0;
    if (d[axis] < d[1])
      axis = 1;
    if (d[axis] < d[2])
      axis = 2;
    return axis;
  }

  float half_area() const
  {
    auto d = diagonal();
    return (d[0] + d[1]) * d[2] + d[0] * d[1];
  }

  static BBox empty()
  {
    return BBox(Vec3(+std::numeric_limits<float>::max()), Vec3(-std::numeric_limits<float>::max()));
  }
};

inline float
robust_min(float a, float b)
{
  return a < b ? a : b;
}
inline float
robust_max(float a, float b)
{
  return a > b ? a : b;
}
inline float
safe_inverse(float x)
{
  return std::fabs(x) <= std::numeric_limits<float>::epsilon()
           ? std::copysign(1.0f / std::numeric_limits<float>::epsilon(), x)
           : 1.0f / x;
}

struct Ray
{
  Vec3 org, dir;

  float tmin, tmax;

  Ray(const Vec3& org_, const Vec3& dir_, float tmin_ = 0, float tmax_ = std::numeric_limits<float>::infinity())
    : org(org_)
    , dir(dir_)
    , tmin(tmin_)
    , tmax(tmax_)
  {}

  Vec3 inv_dir() const { return Vec3(safe_inverse(dir[0]), safe_inverse(dir[1]), safe_inverse(dir[2])); }
};

template<typename Prim>
struct Hit
{
  using Intersection = typename Prim::Intersection;

  uint32_t prim_index;

  Intersection intersection;

  operator bool() const { return prim_index != static_cast<uint32_t>(-1); }

  static Hit none() { return Hit{ static_cast<uint32_t>(-1) }; }
};

struct Node
{
  BBox bbox;
  uint32_t prim_count;
  uint32_t first_index;

  Node() = default;
  Node(const BBox& bbox, uint32_t prim_count, uint32_t first_index)
    : bbox(bbox)
    , prim_count(prim_count)
    , first_index(first_index)
  {}

  bool is_leaf() const { return prim_count != 0; }

  struct Intersection
  {
    float tmin;
    float tmax;
    operator bool() const { return tmin <= tmax; }
  };

  Intersection intersect(const Ray& ray) const
  {
    auto inv_dir = ray.inv_dir();
    auto tmin = (bbox.min - ray.org) * inv_dir;
    auto tmax = (bbox.max - ray.org) * inv_dir;
    std::tie(tmin, tmax) = std::make_pair(min(tmin, tmax), max(tmin, tmax));
    return Intersection{ robust_max(tmin[0], robust_max(tmin[1], robust_max(tmin[2], ray.tmin))),
                         robust_min(tmax[0], robust_min(tmax[1], robust_min(tmax[2], ray.tmax))) };
  }
};

struct CAllocator final
{
  void* alloc(size_t size) { return std::malloc(size); }

  void release(void* addr) { std::free(addr); }
};

struct Bvh
{
  std::vector<Node> nodes;
  std::vector<size_t> prim_indices;

  Bvh() = default;

  static Bvh build(const BBox* bboxes, const Vec3* centers, size_t prim_count);

  size_t depth(size_t node_index = 0) const
  {
    auto& node = nodes[node_index];
    return node.is_leaf() ? 1 : 1 + std::max(depth(node.first_index), depth(node.first_index + 1));
  }

  template<typename Prim>
  Hit<Prim> traverse(Ray& ray, const Prim* prims) const;
};

struct Morton
{
  using Value = uint32_t;
  static constexpr int log_bits = 5;
  static constexpr size_t grid_dim = 1024;

  static Value split(Value x)
  {
    uint64_t mask = (UINT64_C(1) << (1 << log_bits)) - 1;
    for (int i = log_bits, n = 1 << log_bits; i > 0; --i, n >>= 1) {
      mask = (mask | (mask << n)) & ~(mask << (n / 2));
      x = (x | (x << n)) & mask;
    }
    return x;
  }

  static Value encode(Value x, Value y, Value z) { return split(x) | (split(y) << 1) | (split(z) << 2); }
};

struct TriangleIntersection final
{
  float t = std::numeric_limits<float>::infinity();
  float u = 0;
  float v = 0;

  constexpr float distance() const noexcept { return t; }

  constexpr operator bool() const noexcept { return t != std::numeric_limits<float>::infinity(); }
};

struct Triangle final
{
  using Intersection = TriangleIntersection;

  Vec3 p0, p1, p2;

  Triangle() = default;

  Triangle(const Vec3& p0, const Vec3& p1, const Vec3& p2)
    : p0(p0)
    , p1(p1)
    , p2(p2)
  {}

  Intersection intersect(Ray& ray) const;
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

template<typename Prim>
Hit<Prim>
Bvh::traverse(Ray& ray, const Prim* prims) const
{
  auto hit = Hit<Prim>::none();
  SmallStack<uint32_t, 32> stack;
  stack.push_back(0);
  while (!stack.empty()) {
    auto& node = nodes[stack.pop_back()];
    if (!node.intersect(ray))
      continue;

    if (node.is_leaf()) {
      for (size_t i = 0; i < node.prim_count; ++i) {
        auto prim_index = prim_indices[node.first_index + i];
        auto isect = prims[prim_index].intersect(ray);
        if (isect && (!hit || (isect.distance() < hit.intersection.distance()))) {
          hit.prim_index = prim_index;
          hit.intersection = isect;
        }
      }
    } else {
      stack.push_back(node.first_index);
      stack.push_back(node.first_index + 1);
    }
  }
  return hit;
}
