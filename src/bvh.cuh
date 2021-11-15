#pragma once

#include <Qx/common.hpp>

namespace Qx {

using DeviceBvh = Bvh<true>;

DeviceBvh
make_device_bvh(const HostBvh&);

} // namespace Qx
