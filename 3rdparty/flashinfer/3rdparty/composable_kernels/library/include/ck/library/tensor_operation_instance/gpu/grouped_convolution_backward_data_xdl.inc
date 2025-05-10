// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

#ifdef CK_ENABLE_FP16
void add_device_grouped_conv2d_bwd_data_xdl_gnhwk_gkyxc_gnhwc_f16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<2,
                                                                  GNHWK,
                                                                  GKYXC,
                                                                  Empty_Tuple,
                                                                  GNHWC,
                                                                  F16,
                                                                  F16,
                                                                  Empty_Tuple,
                                                                  F16,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);

#endif
#ifdef CK_ENABLE_FP32
void add_device_grouped_conv2d_bwd_data_xdl_gnhwk_gkyxc_gnhwc_f32_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<2,
                                                                  GNHWK,
                                                                  GKYXC,
                                                                  Empty_Tuple,
                                                                  GNHWC,
                                                                  F32,
                                                                  F32,
                                                                  Empty_Tuple,
                                                                  F32,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);
#endif
#ifdef CK_ENABLE_BF16
void add_device_grouped_conv2d_bwd_data_xdl_gnhwk_gkyxc_gnhwc_bf16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<2,
                                                                  GNHWK,
                                                                  GKYXC,
                                                                  Empty_Tuple,
                                                                  GNHWC,
                                                                  BF16,
                                                                  BF16,
                                                                  Empty_Tuple,
                                                                  BF16,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);
#endif

#ifdef CK_ENABLE_FP16
void add_device_grouped_conv2d_bwd_data_xdl_nhwgk_gkyxc_nhwgc_f16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<2,
                                                                  NHWGK,
                                                                  GKYXC,
                                                                  Empty_Tuple,
                                                                  NHWGC,
                                                                  F16,
                                                                  F16,
                                                                  Empty_Tuple,
                                                                  F16,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);
#endif
#ifdef CK_ENABLE_FP32
void add_device_grouped_conv2d_bwd_data_xdl_nhwgk_gkyxc_nhwgc_f32_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<2,
                                                                  NHWGK,
                                                                  GKYXC,
                                                                  Empty_Tuple,
                                                                  NHWGC,
                                                                  F32,
                                                                  F32,
                                                                  Empty_Tuple,
                                                                  F32,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);
#endif
#ifdef CK_ENABLE_BF16
void add_device_grouped_conv2d_bwd_data_xdl_nhwgk_gkyxc_nhwgc_bf16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<2,
                                                                  NHWGK,
                                                                  GKYXC,
                                                                  Empty_Tuple,
                                                                  NHWGC,
                                                                  BF16,
                                                                  BF16,
                                                                  Empty_Tuple,
                                                                  BF16,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);
#endif

// conv3d backward data
#ifdef CK_ENABLE_FP16
void add_device_grouped_conv3d_bwd_data_xdl_gndhwk_gkzyxc_gndhwc_f16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<3,
                                                                  GNDHWK,
                                                                  GKZYXC,
                                                                  Empty_Tuple,
                                                                  GNDHWC,
                                                                  F16,
                                                                  F16,
                                                                  Empty_Tuple,
                                                                  F16,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);
#endif
#ifdef CK_ENABLE_FP32
void add_device_grouped_conv3d_bwd_data_xdl_gndhwk_gkzyxc_gndhwc_f32_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<3,
                                                                  GNDHWK,
                                                                  GKZYXC,
                                                                  Empty_Tuple,
                                                                  GNDHWC,
                                                                  F32,
                                                                  F32,
                                                                  Empty_Tuple,
                                                                  F32,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);
#endif
#ifdef CK_ENABLE_BF16
void add_device_grouped_conv3d_bwd_data_xdl_gndhwk_gkzyxc_gndhwc_bf16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<3,
                                                                  GNDHWK,
                                                                  GKZYXC,
                                                                  Empty_Tuple,
                                                                  GNDHWC,
                                                                  BF16,
                                                                  BF16,
                                                                  Empty_Tuple,
                                                                  BF16,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);
#endif
#ifdef CK_ENABLE_FP16
void add_device_grouped_conv3d_bwd_data_xdl_ndhwgk_gkzyxc_ndhwgc_f16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<3,
                                                                  NDHWGK,
                                                                  GKZYXC,
                                                                  Empty_Tuple,
                                                                  NDHWGC,
                                                                  F16,
                                                                  F16,
                                                                  Empty_Tuple,
                                                                  F16,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);
#endif
#ifdef CK_ENABLE_FP32
void add_device_grouped_conv3d_bwd_data_xdl_ndhwgk_gkzyxc_ndhwgc_f32_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<3,
                                                                  NDHWGK,
                                                                  GKZYXC,
                                                                  Empty_Tuple,
                                                                  NDHWGC,
                                                                  F32,
                                                                  F32,
                                                                  Empty_Tuple,
                                                                  F32,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);
#endif
#ifdef CK_ENABLE_BF16
void add_device_grouped_conv3d_bwd_data_xdl_ndhwgk_gkzyxc_ndhwgc_bf16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<3,
                                                                  NDHWGK,
                                                                  GKZYXC,
                                                                  Empty_Tuple,
                                                                  NDHWGC,
                                                                  BF16,
                                                                  BF16,
                                                                  Empty_Tuple,
                                                                  BF16,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);
#endif
#if defined CK_ENABLE_FP16 && defined CK_ENABLE_FP8 && defined CK_ENABLE_BF8
void add_device_grouped_conv3d_bwd_data_xdl_ndhwgk_gkzyxc_ndhwgc_input_f16_comp_bf8f8_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<3,
                                                                  NDHWGK,
                                                                  GKZYXC,
                                                                  Empty_Tuple,
                                                                  NDHWGC,
                                                                  F16,
                                                                  F16,
                                                                  Empty_Tuple,
                                                                  F16,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  BF8,
                                                                  F8>>>& instances);
#endif

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
