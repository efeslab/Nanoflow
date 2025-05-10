// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

// conv3d backward weight
#ifdef CK_ENABLE_FP16
void add_device_grouped_conv3d_bwd_weight_wmma_gndhwc_gkzyxc_gndhwk_f16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdWeight<3,
                                                           GNDHWC,
                                                           GKZYXC,
                                                           GNDHWK,
                                                           F16,
                                                           F16,
                                                           F16,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances);

void add_device_grouped_conv3d_bwd_weight_wmma_gndhwc_gkzyxc_gndhwk_f16_1x1s1p0_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdWeight<3,
                                                           GNDHWC,
                                                           GKZYXC,
                                                           GNDHWK,
                                                           F16,
                                                           F16,
                                                           F16,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances);

void add_device_grouped_conv3d_bwd_weight_wmma_ndhwgc_gkzyxc_ndhwgk_f16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdWeight<3,
                                                           NDHWGC,
                                                           GKZYXC,
                                                           NDHWGK,
                                                           F16,
                                                           F16,
                                                           F16,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances);

void add_device_grouped_conv3d_bwd_weight_wmma_ndhwgc_gkzyxc_ndhwgk_f16_1x1s1p0_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdWeight<3,
                                                           NDHWGC,
                                                           GKZYXC,
                                                           NDHWGK,
                                                           F16,
                                                           F16,
                                                           F16,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances);
#endif
#ifdef CK_ENABLE_INT8
void add_device_grouped_conv3d_bwd_weight_wmma_gndhwc_gkzyxc_gndhwk_i8_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdWeight<3,
                                                           GNDHWC,
                                                           GKZYXC,
                                                           GNDHWK,
                                                           int8_t,
                                                           int8_t,
                                                           int8_t,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances);

void add_device_grouped_conv3d_bwd_weight_wmma_gndhwc_gkzyxc_gndhwk_i8_1x1s1p0_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdWeight<3,
                                                           GNDHWC,
                                                           GKZYXC,
                                                           GNDHWK,
                                                           int8_t,
                                                           int8_t,
                                                           int8_t,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances);

void add_device_grouped_conv3d_bwd_weight_wmma_ndhwgc_gkzyxc_ndhwgk_i8_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdWeight<3,
                                                           NDHWGC,
                                                           GKZYXC,
                                                           NDHWGK,
                                                           int8_t,
                                                           int8_t,
                                                           int8_t,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances);

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
                                                           PassThrough>>>& instances);
#endif

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
