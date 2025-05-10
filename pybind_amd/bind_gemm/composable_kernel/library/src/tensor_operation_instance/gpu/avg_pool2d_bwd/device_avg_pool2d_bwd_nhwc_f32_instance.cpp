// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "device_avg_pool2d_bwd_nhwc_instance_common.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_avgpool_2D_bwd_nhwc_f32_instances(
    std::vector<std::unique_ptr<DeviceAvgPoolBwd<2, F32, F32, NHWC, NHWC>>>& instances)
{
    add_device_operation_instances(instances,
                                   device_avgpool_2D_bwd_nhwc_instances<F32, F32, F32>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
