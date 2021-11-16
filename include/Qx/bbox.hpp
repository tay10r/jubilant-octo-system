#pragma once

#include <Qx/scalar.hpp>
#include <Qx/vec3.hpp>

namespace Qx {

struct BBox final
{
  Vec3 min;
  Vec3 max;

  BBox() = default;

  constexpr BBox(const Vec3& min, const Vec3& max)
    : min(min)
    , max(max)
  {}

  explicit constexpr BBox(const Vec3& point)
    : BBox(point, point)
  {}

  constexpr BBox& extend(const Vec3& point) { return extend(BBox(point)); }

  constexpr BBox& extend(const BBox& other)
  {
    min = Qx::min(min, other.min);
    max = Qx::max(max, other.max);
    return *this;
  }

  constexpr Vec3 diagonal() const { return max - min; }

  constexpr int largest_axis() const
  {
    auto d = diagonal();
    int axis = 0;
    if (d[axis] < d[1])
      axis = 1;
    if (d[axis] < d[2])
      axis = 2;
    return axis;
  }

  constexpr float half_area() const
  {
    const Vec3 d = diagonal();
    return (d[0] + d[1]) * d[2] + d[0] * d[1];
  }

  static constexpr BBox empty()
  {
    return BBox(Vec3(+std::numeric_limits<float>::max()), Vec3(-std::numeric_limits<float>::max()));
  }
};

} // namespace Qx
