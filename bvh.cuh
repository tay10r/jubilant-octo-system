#pragma once

#include "common.hpp"

using DeviceBvh = Bvh<true>;

DeviceBvh
make_device_bvh(const HostBvh&);
