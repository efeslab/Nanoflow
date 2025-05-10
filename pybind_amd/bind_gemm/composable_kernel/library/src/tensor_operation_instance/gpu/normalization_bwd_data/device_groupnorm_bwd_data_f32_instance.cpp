// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include "normalization_bwd_data_instance_common.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_groupnorm_bwd_data_f32_instances(
    std::vector<std::unique_ptr<DeviceNormalizationBwdData<F32, F32, F32, F32, F32, 5, 3>>>&
        instances)
{
    add_device_operation_instances(instances, device_groupnorm_bwd_data_f32_generic_instance{});
    add_device_operation_instances(instances, device_groupnorm_bwd_data_f32_instances{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
