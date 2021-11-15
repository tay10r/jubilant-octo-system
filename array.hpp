#pragma once

#include <cstdlib>

template<bool UseDeviceMemory>
struct Allocator final
{
  static void* alloc(size_t size) { return std::malloc(size); }

  static void release(void* addr) { std::free(addr); }
};

#ifdef __CUDACC__

template<>
struct Allocator<true>
{
  static void* alloc(size_t size) { return cudaMalloc(size); }

  static void release(void* addr) { cudaFree(addr); }
};

#endif // __CUDACC__

template<typename Element, bool IsDeviceArray = false>
class Array final
{
public:
  using Allocator = ::Allocator<IsDeviceArray>;

  Array(const Array&) = delete;

  Array(Array&& other)
    : m_data(other.m_data)
    , m_size(other.m_size)
  {
    other.m_data = nullptr;
    other.m_size = 0;
  }

  Array(size_t size)
    : m_data(static_cast<Element*>(Allocator::alloc(size * sizeof(Element))))
    , m_size(size)
  {}

  ~Array()
  {
    Allocator::release(m_data);
    m_data = nullptr;
    m_size = 0;
  }
  const Element& operator[](size_t index) const noexcept { return m_data[index]; }

  Element& operator[](size_t index) noexcept { return m_data[index]; }

  size_t size() const noexcept { return m_size; }

  Element* begin() noexcept { return m_data; }

  Element* end() noexcept { return m_data + m_size; }

  const Element* begin() const noexcept { return m_data; }

  const Element* end() const noexcept { return m_data + m_size; }

private:
  Element* m_data = nullptr;

  size_t m_size = 0;
};

#ifdef __CUDACC__

template<typename Element>
void
transfer(const Array<Element, false>& src, Array<Element, true>& dst)
{
  cudaMemcpy(&dst[0], &src[0], src.size() * sizeof(Element), cudaMemcpyHostToDevice);
}

template<typename Element>
void
transfer(const Array<Element, true>& src, Array<Element, false>& dst)
{
  cudaMemcpy(&dst[0], &src[0], src.size() * sizeof(Element), cudaMemcpyDeviceToHost);
}

#endif // __CUDACC__
