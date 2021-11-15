#pragma once

namespace Qx {

template<typename Element, int Max>
class SmallStack final
{
public:
  constexpr bool empty() const noexcept { return m_size == 0; }

  constexpr int size() const noexcept { return m_size; }

  Element pop_back()
  {
    m_size--;
    return m_elements[m_size];
  }

  void push_back_safely(Element element)
  {
    if ((m_size + 1) < Max)
      push_back(element);
  }

  void push_back(Element element)
  {
    m_elements[m_size] = element;

    m_size++;
  }

private:
  Element m_elements[Max];

  int m_size = 0;
};

} // namespace Qx
