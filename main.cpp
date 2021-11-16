#include <window_blit/window_blit.hpp>

#include <Qx/common.hpp>
#include <Qx/render.hpp>

#include <fstream>
#include <optional>

namespace obj {

inline void
remove_eol(char* ptr)
{
  int i = 0;
  while (ptr[i])
    i++;
  i--;
  while (i > 0 && std::isspace(ptr[i])) {
    ptr[i] = '\0';
    i--;
  }
}

inline char*
strip_spaces(char* ptr)
{
  while (std::isspace(*ptr))
    ptr++;
  return ptr;
}

inline std::optional<int>
read_index(char** ptr)
{
  char* base = *ptr;

  // Detect end of line (negative indices are supported)
  base = strip_spaces(base);
  if (!std::isdigit(*base) && *base != '-')
    return std::nullopt;

  int index = std::strtol(base, &base, 10);
  base = strip_spaces(base);

  if (*base == '/') {
    base++;

    // Handle the case when there is no texture coordinate
    if (*base != '/')
      std::strtol(base, &base, 10);

    base = strip_spaces(base);

    if (*base == '/') {
      base++;
      std::strtol(base, &base, 10);
    }
  }

  *ptr = base;
  return std::make_optional(index);
}

inline std::vector<Qx::Triangle>
load_from_stream(std::istream& is)
{
  static constexpr size_t max_line = 1024;
  char line[max_line];

  std::vector<Qx::Vec3> vertices;
  std::vector<Qx::Triangle> triangles;

  while (is.getline(line, max_line)) {
    char* ptr = strip_spaces(line);
    if (*ptr == '\0' || *ptr == '#')
      continue;
    remove_eol(ptr);
    if (*ptr == 'v' && std::isspace(ptr[1])) {
      auto x = std::strtof(ptr + 1, &ptr);
      auto y = std::strtof(ptr, &ptr);
      auto z = std::strtof(ptr, &ptr);
      vertices.emplace_back(x, y, z);
    } else if (*ptr == 'f' && std::isspace(ptr[1])) {
      Qx::Vec3 points[2];
      ptr += 2;
      for (size_t i = 0;; ++i) {
        if (auto index = read_index(&ptr)) {
          size_t j = *index < 0 ? vertices.size() + *index : *index - 1;
          assert(j < vertices.size());
          auto v = vertices[j];
          if (i >= 2) {
            triangles.emplace_back(points[0], points[1], v);
            points[1] = v;
          } else {
            points[i] = v;
          }
        } else {
          break;
        }
      }
    }
  }

  return triangles;
}

inline std::vector<Qx::Triangle>
load_from_file(const std::string& file)
{
  std::ifstream is(file);
  if (is)
    return load_from_stream(is);
  return std::vector<Qx::Triangle>();
}

} // namespace obj

namespace {

class App final : public window_blit::AppBase
{
public:
  App(std::vector<Qx::Triangle>&& triangles, Qx::HostBvh&& bvh, GLFWwindow* window)
    : window_blit::AppBase(window)
    , m_triangles(std::move(triangles))
    , m_bvh(std::move(bvh))
    , m_renderer(Qx::Renderer::create(m_triangles.data(), m_bvh))
  {}

  void render(float* rgb, int width, int height) override
  {
    const glm::vec3 eye = get_camera_position();

    const glm::vec3 dir = get_camera_rotation_transform() * glm::vec3(0, 0, -1);

    const glm::vec3 up = get_camera_rotation_transform() * glm::vec3(0, 1, 0);

    const glm::vec3 right = glm::normalize(glm::cross(dir, up));

    const Qx::Vec3 tmp_eye(eye.x, eye.y, eye.z);
    const Qx::Vec3 tmp_dir(dir.x, dir.y, dir.z);
    const Qx::Vec3 tmp_up(up.x, up.y, up.z);
    const Qx::Vec3 tmp_right(right.x, right.y, right.z);

    m_renderer->render(tmp_eye, tmp_dir, tmp_right, tmp_up, width, height, rgb);
  }

private:
  std::vector<Qx::Triangle> m_triangles;

  Qx::HostBvh m_bvh;

  std::unique_ptr<Qx::Renderer> m_renderer;
};

class AppFactory final : public window_blit::AppFactoryBase
{
public:
  AppFactory(std::vector<Qx::Triangle>&& triangles, Qx::HostBvh&& bvh)
    : m_triangles(std::move(triangles))
    , m_bvh(std::move(bvh))
  {}

  window_blit::App* create_app(GLFWwindow* window) { return new App(std::move(m_triangles), std::move(m_bvh), window); }

private:
  std::vector<Qx::Triangle> m_triangles;

  Qx::HostBvh m_bvh;
};

} // namespace

int
main(int argc, char** argv)
{
  if (argc < 2) {
    std::cerr << "Missing input file" << std::endl;
    return 1;
  }

  auto tris = obj::load_from_file(argv[1]);
  if (tris.empty()) {
    std::cerr << "No triangle was found in input OBJ file" << std::endl;
    return 1;
  }
  std::cout << "Loaded file with " << tris.size() << " triangle(s)" << std::endl;

  std::vector<Qx::BBox> bboxes(tris.size());
  std::vector<Qx::Vec3> centers(tris.size());
  for (size_t i = 0; i < tris.size(); ++i) {
    bboxes[i] = Qx::BBox(tris[i].p0).extend(tris[i].p1).extend(tris[i].p2);
    centers[i] = (tris[i].p0 + tris[i].p1 + tris[i].p2) * (1.0f / 3.0f);
  }
  auto bvh = Qx::build_bvh(bboxes.data(), centers.data(), tris.size());

  AppFactory appFactory(std::move(tris), std::move(bvh));

  return window_blit::run_glfw_window(std::move(appFactory));
}
