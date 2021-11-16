#pragma once

#include <Qx/array.hpp>
#include <Qx/bbox.hpp>
#include <Qx/hit.hpp>
#include <Qx/ray.hpp>
#include <Qx/small_stack.hpp>
#include <Qx/vec3.hpp>

namespace Qx {

struct Node final
{
  BBox bbox;
  std::uint32_t prim_count = 0;
  std::uint32_t first_index = 0;

  constexpr Node() = default;

  constexpr Node(const BBox& bbox, std::uint32_t prim_count, std::uint32_t first_index)
    : bbox(bbox)
    , prim_count(prim_count)
    , first_index(first_index)
  {}

  constexpr bool __device__ is_leaf() const { return prim_count != 0; }

  struct Intersection final
  {
    float tmin = 1;
    float tmax = 0;
    __device__ operator bool() const { return tmin <= tmax; }
  };

  Intersection __device__ intersect(const Ray& ray) const
  {
    auto inv_dir = ray.inv_dir();
    auto tmin = (bbox.min - ray.org) * inv_dir;
    auto tmax = (bbox.max - ray.org) * inv_dir;
    tmin = min(tmin, tmax);
    tmax = max(tmin, tmax);
    return Intersection{ max(tmin[0], max(tmin[1], max(tmin[2], ray.tmin))),
                         min(tmax[0], min(tmax[1], min(tmax[2], ray.tmax))) };
  }
};

template<bool DeviceFlag>
struct Bvh final
{
  Array<Node, DeviceFlag> nodes;

  Array<size_t, DeviceFlag> prim_indices;

  Bvh(size_t prim_count)
    : nodes((2 * prim_count) - 1)
    , prim_indices(prim_count)
  {}

  template<typename Prim>
  Hit<Prim> __device__ traverse(Ray& ray, const Array<Prim, DeviceFlag>& prims) const;
};

using HostBvh = Bvh<false>;

HostBvh
build_bvh(const BBox* bboxes, const Vec3* centers, size_t prim_count);

template<bool DeviceFlag>
template<typename Prim>
Hit<Prim> __device__
Bvh<DeviceFlag>::traverse(Ray& ray, const Array<Prim, DeviceFlag>& prims) const
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

} // namespace Qx
