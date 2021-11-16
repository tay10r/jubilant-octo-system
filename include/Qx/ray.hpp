#pragma once

#include <Qx/scalar.hpp>
#include <Qx/vec3.hpp>

namespace Qx {

struct Ray
{
  Vec3 org, dir;

  float tmin, tmax;

  constexpr __device__ Ray(const Vec3& org_, const Vec3& dir_, float tmin_ = 0, float tmax_ = infinity())
    : org(org_)
    , dir(dir_)
    , tmin(tmin_)
    , tmax(tmax_)
  {}

  Vec3 __device__ inv_dir() const { return Vec3(safe_inverse(dir[0]), safe_inverse(dir[1]), safe_inverse(dir[2])); }
};

} // namespace Qx
