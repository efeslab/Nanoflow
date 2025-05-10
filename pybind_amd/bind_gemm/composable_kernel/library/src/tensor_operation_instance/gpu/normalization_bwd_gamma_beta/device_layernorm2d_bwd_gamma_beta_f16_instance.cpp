// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include "normalization_bwd_gamma_beta_instance_common.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_layernorm2d_bwd_gamma_beta_f16_instances(
    std::vector<std::unique_ptr<DeviceNormalizationBwdGammaBeta<F16, F16, F16, F16, F16, 2, 1>>>&
        instances)
{
    add_device_operation_instances(instances,
                                   device_layernorm_bwd_gamma_beta_f16_generic_instance<2, 1>{});
    add_device_operation_instances(instances,
                                   device_layernorm_bwd_gamma_beta_f16_instances<2, 1>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
