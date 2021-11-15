#pragma once

#include <Qx/cuda_macros.hpp>
#include <Qx/array.hpp>

#include <cstdio>

namespace Qx {

template<>
struct Allocator<true> final
{
  static void* alloc(size_t size)
  {
    void* result = nullptr;

    const cudaError_t error = cudaMalloc(&result, size);

    if (error != cudaSuccess)
      fprintf(stderr, "%s\n", cudaGetErrorString(error));

    assert(error == cudaSuccess);

    return result;
  }

  static void release(void* addr) { cudaFree(addr); }
};

template<typename Element>
using DeviceArray = Array<Element, true>;

template<typename Element>
void
hostToDevice(const Array<Element, false>& src, Array<Element, true>& dst)
{
  cudaMemcpy(&dst[0], &src[0], src.size() * sizeof(Element), cudaMemcpyHostToDevice);
}

template<typename Element>
void
deviceToHost(const Array<Element, true>& src, Array<Element, false>& dst)
{
  cudaMemcpy(&dst[0], &src[0], src.size() * sizeof(Element), cudaMemcpyDeviceToHost);
}

} // namespace Qx
