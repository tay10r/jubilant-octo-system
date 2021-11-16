#pragma once

#include <cstdint>

namespace Qx {

template<typename Primitive>
struct Hit final
{
  using Intersection = typename Primitive::Intersection;

  std::uint32_t prim_index = static_cast<uint32_t>(-1);

  Intersection intersection;

  constexpr __device__ operator bool() const { return prim_index != static_cast<uint32_t>(-1); }

  static constexpr Hit __device__ none() { return Hit{ static_cast<uint32_t>(-1), Intersection{} }; }
};

} // namespace Qx
