#include "bvh.cuh"

#include "array.cuh"

DeviceBvh
make_device_bvh(const HostBvh& bvh)
{
  DeviceBvh device_bvh(bvh.prim_indices.size());

  hostToDevice(bvh.nodes, device_bvh.nodes);

  hostToDevice(bvh.prim_indices, device_bvh.prim_indices);

  return device_bvh;
}
