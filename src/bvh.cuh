#pragma once

#include <Qx/bvh.hpp>

namespace Qx {

using DeviceBvh = Bvh<true>;

DeviceBvh
make_device_bvh(const HostBvh&);

} // namespace Qx
