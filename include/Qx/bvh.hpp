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
    auto tmin_0 = (bbox.min - ray.org) * inv_dir;
    auto tmax_0 = (bbox.max - ray.org) * inv_dir;
    auto tmin_1 = min(tmin_0, tmax_0);
    auto tmax_1 = max(tmin_0, tmax_0);
    return Intersection{ max(tmin_1[0], max(tmin_1[1], max(tmin_1[2], ray.tmin))),
                         min(tmax_1[0], min(tmax_1[1], min(tmax_1[2], ray.tmax))) };
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
  static Hit<Prim> __device__ traverse(Ray& ray, const Prim* prims, const Node* nodes);
};

template<typename Prim>
Array<Prim, false>
permute(const Bvh<false>& bvh, const Prim* primitives)
{
  Array<Prim, false> result(bvh.prim_indices.size());

  for (size_t i = 0; i < bvh.prim_indices.size(); i++)
    result[i] = primitives[bvh.prim_indices[i]];

  return result;
}

using HostBvh = Bvh<false>;

HostBvh
build_bvh(const BBox* bboxes, const Vec3* centers, size_t prim_count);

template<bool DeviceFlag>
template<typename Prim>
Hit<Prim> __device__
Bvh<DeviceFlag>::traverse(Ray& ray, const Prim* prims, const Node* nodes)
{
  auto hit = Hit<Prim>::none();

  SmallStack<const Node*, 128> stack;

  const Node* node = &nodes[0];

  while (true) {

    auto first_child = node->first_index;
    const Node* lNode = &nodes[first_child + 0];
    const Node* rNode = &nodes[first_child + 1];
    const auto lHit = lNode->intersect(ray);
    const auto rHit = rNode->intersect(ray);

    if (lHit) {
      if (lNode->is_leaf()) {
        const auto isect = prims[lNode->first_index].intersect(ray);
        if (isect && (!hit || (isect.distance() < hit.intersection.distance()))) {
          hit.prim_index = lNode->first_index;
          hit.intersection = isect;
        }
        lNode = nullptr;
      }
    } else {
      lNode = nullptr;
    }

    if (rHit) {
      if (rNode->is_leaf()) {
        const auto isect = prims[rNode->first_index].intersect(ray);
        if (isect && (!hit || (isect.distance() < hit.intersection.distance()))) {
          hit.prim_index = rNode->first_index;
          hit.intersection = isect;
        }
        rNode = nullptr;
      }
    } else {
      rNode = nullptr;
    }

    if ((lNode != nullptr) ^ (rNode != nullptr)) {
      node = (lNode != nullptr) ? lNode : rNode;
    } else if ((lNode != nullptr) & (rNode != nullptr)) {
      if (lHit.tmin > rHit.tmin) {
        const Node* tmp = lNode;
        lNode = rNode;
        rNode = tmp;
      }
      stack.push_back(rNode);
      node = lNode;
    } else {
      if (stack.empty())
        break;
      node = stack.pop_back();
    }
  }

  return hit;
}

} // namespace Qx
