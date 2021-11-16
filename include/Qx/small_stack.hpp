#pragma once

#include <Qx/cuda_macros.hpp>

namespace Qx {

template<typename Element, int Max>
class SmallStack final
{
public:
  constexpr bool __device__ empty() const noexcept { return m_size == 0; }

  constexpr int __device__ size() const noexcept { return m_size; }

  Element __device__ pop_back()
  {
    m_size--;
    return m_elements[m_size];
  }

  void __device__ push_back_safely(Element element)
  {
    if ((m_size + 1) < Max)
      push_back(element);
  }

  void __device__ push_back(Element element)
  {
    m_elements[m_size] = element;

    m_size++;
  }

private:
  Element m_elements[Max];

  int m_size = 0;
};

} // namespace Qx
