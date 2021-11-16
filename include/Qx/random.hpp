#pragma once

#include <Qx/cuda_macros.hpp>
#include <Qx/scalar.hpp>

#include <cstdint>

namespace Qx {

class Pcg final
{
public:
  __device__ constexpr Pcg(std::uint32_t initial_state)
    : m_state(initial_state)
  {}

  __device__ constexpr std::uint32_t operator()() noexcept
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
std::uint32_t __device__
random_int(Rng& rng)
{
  return rng();
}

template<typename Rng>
float __device__
random_float(Rng& rng)
{
  std::uint32_t value = random_int(rng);

  return floatbits((value & 0x007f'ffff) | 0x3f80'0000) - 1.0f;
}

template<typename Rng>
float __device__
random_float_2(Rng& rng)
{
  std::uint32_t value = random_int(rng);

  return floatbits((value & 0x007f'ffff) | 0x4000'0000) - 3.0f;
}

} // namespace Qx
