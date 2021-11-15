#pragma once

#include <Qx/cuda_macros.hpp>

#include <cassert>
#include <cstdlib>
#include <cstring>

namespace Qx {

template<bool UseDeviceMemory>
struct Allocator final
{};

template<>
struct Allocator<false> final
{
  static void* alloc(size_t size) { return std::malloc(size); }

  static void release(void* addr) { std::free(addr); }
};

template<typename Element, bool IsDeviceArray>
class Array final
{
public:
  using ArrayAllocator = Qx::Allocator<IsDeviceArray>;

  Array(const Array&) = delete;

  Array(Array&& other)
    : m_data(other.m_data)
    , m_size(other.m_size)
  {
    other.m_data = nullptr;
    other.m_size = 0;
  }

  Array(size_t size)
    : m_data(static_cast<Element*>(ArrayAllocator::alloc(size * sizeof(Element))))
    , m_size(size)
  {}

  ~Array()
  {
    ArrayAllocator::release(m_data);
    m_data = nullptr;
    m_size = 0;
  }

  __device__ __host__ const Element& operator[](size_t index) const noexcept { return m_data[index]; }

  __device__ __host__ Element& operator[](size_t index) noexcept { return m_data[index]; }

  __device__ __host__ size_t size() const noexcept { return m_size; }

  __device__ __host__ Element* begin() noexcept { return m_data; }

  __device__ __host__ Element* end() noexcept { return m_data + m_size; }

  const Element* begin() const noexcept { return m_data; }

  const Element* end() const noexcept { return m_data + m_size; }

private:
  Element* m_data = nullptr;

  size_t m_size = 0;
};

template<typename Element>
using HostArray = Array<Element, false>;

} // namespace Qx
