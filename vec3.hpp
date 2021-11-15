#pragma once

#include "cuda_macros.hpp"

#include <algorithm>

#include <cmath>

struct Vec3
{
  float values[3];

  DEVHOST_FUNC Vec3() = default;

  DEVHOST_FUNC Vec3(float x, float y, float z)
    : values{ x, y, z }
  {}

  DEVHOST_FUNC explicit Vec3(float x)
    : Vec3(x, x, x)
  {}

  DEVHOST_FUNC float& operator[](int i) { return values[i]; }

  DEVHOST_FUNC float operator[](int i) const { return values[i]; }
};

DEVHOST_FUNC inline Vec3
operator+(const Vec3& a, const Vec3& b)
{
  return Vec3(a[0] + b[0], a[1] + b[1], a[2] + b[2]);
}

DEVHOST_FUNC inline Vec3
operator-(const Vec3& a, const Vec3& b)
{
  return Vec3(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}

inline Vec3
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

inline Vec3
min(const Vec3& a, const Vec3& b)
{
  return Vec3(std::min(a[0], b[0]), std::min(a[1], b[1]), std::min(a[2], b[2]));
}

inline Vec3
max(const Vec3& a, const Vec3& b)
{
  return Vec3(std::max(a[0], b[0]), std::max(a[1], b[1]), std::max(a[2], b[2]));
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
