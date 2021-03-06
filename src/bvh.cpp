#include <Qx/bvh.hpp>

#include <algorithm>
#include <numeric>
#include <vector>

namespace Qx {

namespace {

struct Morton
{
  using Value = uint32_t;
  static constexpr int log_bits = 5;
  static constexpr size_t grid_dim = 1024;

  static Value split(Value x)
  {
    uint64_t mask = (UINT64_C(1) << (1 << log_bits)) - 1;
    for (int i = log_bits, n = 1 << log_bits; i > 0; --i, n >>= 1) {
      mask = (mask | (mask << n)) & ~(mask << (n / 2));
      x = (x | (x << n)) & mask;
    }
    return x;
  }

  static Value encode(Value x, Value y, Value z) { return split(x) | (split(y) << 1) | (split(z) << 2); }
};

inline size_t
find_closest_node(const std::vector<Node>& nodes, size_t index)
{
  static size_t search_radius = 14;
  size_t begin = index > search_radius ? index - search_radius : 0;
  size_t end = index + search_radius + 1 < nodes.size() ? index + search_radius + 1 : nodes.size();
  auto& first_node = nodes[index];
  size_t best_index = 0;
  float best_distance = std::numeric_limits<float>::max();
  for (size_t i = begin; i < end; ++i) {
    if (i == index)
      continue;
    auto& second_node = nodes[i];
    auto distance = BBox(first_node.bbox).extend(second_node.bbox).half_area();
    if (distance < best_distance) {
      best_distance = distance;
      best_index = i;
    }
  }
  return best_index;
}

BBox compute_centroid_bounds(const Vec3* centers, size_t prim_count)
{
  BBox bounds = BBox::empty();

  for (size_t i = 0; i < prim_count; i++)
    bounds.extend(centers[i]);

  return bounds;
}

} // namespace

HostBvh
build_bvh(const BBox* bboxes, const Vec3* centers, size_t prim_count)
{
  HostBvh bvh(prim_count);

  auto center_bbox = compute_centroid_bounds(centers, prim_count);

  // Compute morton codes for each primitive
  std::vector<Morton::Value> mortons(prim_count);
  for (size_t i = 0; i < prim_count; ++i) {
    auto grid_pos =
      min(Vec3(Morton::grid_dim - 1),
          max(Vec3(0), (centers[i] - center_bbox.min) * (Vec3(Morton::grid_dim) / center_bbox.diagonal())));
    mortons[i] = Morton::encode(grid_pos[0], grid_pos[1], grid_pos[2]);
  }

  // Sort primitives according to their morton code
  std::iota(bvh.prim_indices.begin(), bvh.prim_indices.end(), 0);

  std::sort(
    bvh.prim_indices.begin(), bvh.prim_indices.end(), [&](size_t i, size_t j) { return mortons[i] < mortons[j]; });

  // Create leaves
  std::vector<Node> current_nodes(prim_count), next_nodes;
  std::vector<size_t> merge_index(prim_count);
  for (size_t i = 0; i < prim_count; ++i) {
    current_nodes[i].prim_count = 1;
    current_nodes[i].first_index = i;
    current_nodes[i].bbox = bboxes[bvh.prim_indices[i]];
  }

  // Merge nodes until there is only one left
  size_t insertion_index = bvh.nodes.size();

  while (current_nodes.size() > 1) {
    for (size_t i = 0; i < current_nodes.size(); ++i)
      merge_index[i] = find_closest_node(current_nodes, i);
    next_nodes.clear();
    for (size_t i = 0; i < current_nodes.size(); ++i) {
      auto j = merge_index[i];
      // The two nodes should be merged if they agree on their respective merge
      // indices.
      if (i == merge_index[j]) {
        // Since we only need to merge once, we only merge if the first index is
        // less than the second.
        if (i > j)
          continue;

        // Reserve space in the target array for the two children
        assert(insertion_index >= 2);
        insertion_index -= 2;
        bvh.nodes[insertion_index + 0] = current_nodes[i];
        bvh.nodes[insertion_index + 1] = current_nodes[j];

        // Create the parent node and place it in the array for the next
        // iteration
        Node parent;
        parent.bbox = BBox(current_nodes[i].bbox).extend(current_nodes[j].bbox);
        parent.first_index = insertion_index;
        parent.prim_count = 0;
        next_nodes.push_back(parent);
      } else {
        // The current node should be kept for the next iteration
        next_nodes.push_back(current_nodes[i]);
      }
    }
    std::swap(next_nodes, current_nodes);
  }
  assert(insertion_index == 1);

  // Copy root node into the destination array
  bvh.nodes[0] = current_nodes[0];
  return bvh;
}

} // namespace Qx
