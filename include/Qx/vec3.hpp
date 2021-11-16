#pragma once

#include <Qx/cuda_macros.hpp>
#include <Qx/scalar.hpp>

#include <algorithm>

#include <cmath>

namespace Qx {

struct Vec3
{
  float values[3];

  __device__ __host__ Vec3() = default;

  __device__ __host__ Vec3(float x, float y, float z)
    : values{ x, y, z }
  {}

  __device__ __host__ explicit Vec3(float x)
    : Vec3(x, x, x)
  {}

  __device__ __host__ float& operator[](int i) { return values[i]; }

  __device__ __host__ float operator[](int i) const { return values[i]; }
};

__device__ __host__ inline Vec3
operator+(const Vec3& a, const Vec3& b)
{
  return Vec3(a[0] + b[0], a[1] + b[1], a[2] + b[2]);
}

__device__ __host__ inline Vec3
operator-(const Vec3& a, const Vec3& b)
{
  return Vec3(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}

__device__ __host__ inline Vec3
operator*(const Vec3& a, const Vec3& b)
{
  return Vec3(a[0] * b[0], a[1] * b[1], a[2] * b[2]);
}

inline Vec3
operator*(const Vec3& a, float b)
{
  return Vec3(a[0] * b, a[1] * b, a[2] * b);
}

inline Vec3
operator/(const Vec3& a, const Vec3& b)
{
  return Vec3(a[0] / b[0], a[1] / b[1], a[2] / b[2]);
}

inline Vec3
operator*(float a, const Vec3& b)
{
  return b * a;
}

__device__ __host__ inline Vec3
min(const Vec3& a, const Vec3& b)
{
  return Vec3(min(a[0], b[0]), min(a[1], b[1]), min(a[2], b[2]));
}

__device__ __host__ inline Vec3
max(const Vec3& a, const Vec3& b)
{
  return Vec3(max(a[0], b[0]), max(a[1], b[1]), max(a[2], b[2]));
}

inline float
dot(const Vec3& a, const Vec3& b)
{
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

inline float
length(const Vec3& a)
{
  return std::sqrt(dot(a, a));
}

inline Vec3
normalize(const Vec3& a)
{
  return a * (1.0f / length(a));
}

inline Vec3
cross(const Vec3& a, const Vec3& b)
{
  return Vec3(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]);
}

} // namespace Qx
