#pragma once

#include <memory>

namespace Qx {

template<bool>
struct Bvh;

struct Triangle;
struct Vec3;

class Renderer
{
public:
  static std::unique_ptr<Renderer> create(const Triangle* triangles, const Bvh<false>& bvh);

  virtual ~Renderer() = default;

  virtual void
  render(const Vec3& eye, const Vec3& dir, const Vec3& right, const Vec3& up, int width, int height, float* rgb) = 0;
};

} // namespace Qx
