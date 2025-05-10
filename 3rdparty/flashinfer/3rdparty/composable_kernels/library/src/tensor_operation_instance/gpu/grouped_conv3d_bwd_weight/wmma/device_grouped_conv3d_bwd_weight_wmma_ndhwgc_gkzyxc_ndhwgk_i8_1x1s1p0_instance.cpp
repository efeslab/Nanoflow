// SPDX-License-Identifier: MIT
// Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_conv_bwd_weight/device_grouped_conv_bwd_weight_wmma_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
void add_device_grouped_conv3d_bwd_weight_wmma_ndhwgc_gkzyxc_ndhwgk_i8_1x1s1p0_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdWeight<3,
                                                           NDHWGC,
                                                           GKZYXC,
                                                           NDHWGK,
                                                           int8_t,
                                                           int8_t,
                                                           int8_t,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances)
{
    add_device_operation_instances(
        instances,
        device_grouped_conv_bwd_weight_wmma_i8_instances<3,
                                                         NDHWGC,
                                                         GKZYXC,
                                                         NDHWGK,
                                                         ConvBwdWeightFilter1x1Stride1Pad0>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
