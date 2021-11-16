#pragma once

#include <Qx/cuda_macros.hpp>

#include <cmath>
#include <cstdint>

namespace Qx {

template<typename Scalar>
constexpr Scalar __device__ __host__
min(Scalar a, Scalar b)
{
  return (a < b) ? a : b;
}

template<typename Scalar>
constexpr Scalar __device__ __host__
max(Scalar a, Scalar b)
{
  return (a > b) ? a : b;
}

template<typename Scalar>
constexpr Scalar __device__ __host__
clamp(Scalar x, Scalar min_value, Scalar max_value)
{
  return min(max(x, min_value), max_value);
}

inline float __device__ __host__
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

inline float __device__ __host__
infinity()
{
  return floatbits(0x7f800000);
}

inline float __device__ __host__
safe_inverse(float x)
{
  return std::fabs(x) <= std::numeric_limits<float>::epsilon()
           ? std::copysign(1.0f / std::numeric_limits<float>::epsilon(), x)
           : 1.0f / x;
}

} // namespace Qx
