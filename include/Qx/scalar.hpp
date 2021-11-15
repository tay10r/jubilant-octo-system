#pragma once

#include <Qx/cuda_macros.hpp>

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

} // namespace Qx
