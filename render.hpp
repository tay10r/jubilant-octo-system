struct Bvh;
struct Triangle;
struct Vec3;

void
render(const Vec3& eye,
       const Vec3& dir,
       const Vec3& right,
       const Vec3& up,
       const Bvh& bvh,
       const Triangle* triangles,
       int width,
       int height,
       float* rgb);
