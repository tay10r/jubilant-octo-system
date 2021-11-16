#pragma once

#include <Qx/cuda_macros.hpp>
#include <Qx/scalar.hpp>

#include <algorithm>

#include <cmath>

namespace Qx {

struct Vec3 final
{
  float values[3];

  __device__ __host__ constexpr Vec3()
    : Vec3(0, 0, 0)
  {}

  __device__ __host__ constexpr Vec3(float x, float y, float z)
    : values{ x, y, z }
  {}

  __device__ __host__ explicit constexpr Vec3(float x)
    : Vec3(x, x, x)
  {}

  __device__ __host__ constexpr float& operator[](int i) { return values[i]; }

  __device__ __host__ constexpr float operator[](int i) const { return values[i]; }
};

constexpr Vec3 __device__ __host__
operator+(const Vec3& a, const Vec3& b)
{
  return Vec3(a[0] + b[0], a[1] + b[1], a[2] + b[2]);
}

constexpr Vec3 __device__ __host__
operator-(const Vec3& a, const Vec3& b)
{
  return Vec3(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}

constexpr Vec3 __device__ __host__
operator*(const Vec3& a, const Vec3& b)
{
  return Vec3(a[0] * b[0], a[1] * b[1], a[2] * b[2]);
}

constexpr Vec3 __device__ __host__
operator*(const Vec3& a, float b)
{
  return Vec3(a[0] * b, a[1] * b, a[2] * b);
}

constexpr Vec3 __device__ __host__
operator/(const Vec3& a, const Vec3& b)
{
  return Vec3(a[0] / b[0], a[1] / b[1], a[2] / b[2]);
}

constexpr Vec3 __device__ __host__
operator*(float a, const Vec3& b)
{
  return b * a;
}

constexpr Vec3 __device__ __host__
min(const Vec3& a, const Vec3& b)
{
  return Vec3(min(a[0], b[0]), min(a[1], b[1]), min(a[2], b[2]));
}

constexpr Vec3 __device__ __host__
max(const Vec3& a, const Vec3& b)
{
  return Vec3(max(a[0], b[0]), max(a[1], b[1]), max(a[2], b[2]));
}

constexpr float __device__ __host__
dot(const Vec3& a, const Vec3& b)
{
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

inline float __device__ __host__
length(const Vec3& a)
{
  return std::sqrt(dot(a, a));
}

inline Vec3 __device__ __host__
normalize(const Vec3& a)
{
  return a * (1.0f / length(a));
}

constexpr Vec3 __device__ __host__
cross(const Vec3& a, const Vec3& b)
{
  return Vec3(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]);
}

} // namespace Qx
