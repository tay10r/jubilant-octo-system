#pragma once

#include <cstdint>

class Pcg final
{
public:
  constexpr Pcg(std::uint32_t initial_state)
    : m_state(initial_state)
  {}

  constexpr std::uint32_t operator()() noexcept
  {
    const std::uint32_t state = m_state;
    m_state = m_state * 747796405u + 2891336453u;
    const std::uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
  }

private:
  std::uint32_t m_state;
};

template<typename Rng>
std::uint32_t
random_int(Rng& rng)
{
  return rng();
}

float
floatbits(std::uint32_t in)
{
  union
  {
    std::uint32_t i;
    float f;
  } v;

  static_assert(sizeof(v) == 4);

  v.i = in;

  return v.f;
}

template<typename Rng>
float
random_float(Rng& rng)
{
  std::uint32_t value = random_int(rng);

  return floatbits((value & 0x007f'ffff) | 0x3f80'0000) - 1.0f;
}

template<typename Rng>
float
random_float_2(Rng& rng)
{
  std::uint32_t value = random_int(rng);

  return floatbits((value & 0x007f'ffff) | 0x4000'0000) - 3.0f;
}
