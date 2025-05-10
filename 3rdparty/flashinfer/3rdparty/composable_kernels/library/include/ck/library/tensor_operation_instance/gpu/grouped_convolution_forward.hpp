// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <vector>
#include <memory>
#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_conv_fwd_multiple_abd.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"

#ifdef DL_KERNELS
#include "grouped_convolution_forward_dl.inc"
#endif
#ifdef CK_USE_XDL
#include "grouped_convolution_forward_xdl.inc"
#endif
#ifdef CK_USE_WMMA
#include "grouped_convolution_forward_wmma.inc"
#endif

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

template <ck::index_t NumDimSpatial,
          typename InLayout,
          typename WeiLayout,
          typename OutLayout,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename AComputeType,
          typename BComputeType>
struct DeviceOperationInstanceFactory<ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<
    NumDimSpatial,
    InLayout,
    WeiLayout,
    Empty_Tuple,
    OutLayout,
    InDataType,
    WeiDataType,
    Empty_Tuple,
    OutDataType,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough,
    AComputeType,
    BComputeType>>
{
    using DeviceOp =
        DeviceGroupedConvFwdMultipleABD<NumDimSpatial,
                                        InLayout,
                                        WeiLayout,
                                        Empty_Tuple,
                                        OutLayout,
                                        InDataType,
                                        WeiDataType,
                                        Empty_Tuple,
                                        OutDataType,
                                        ck::tensor_operation::element_wise::PassThrough,
                                        ck::tensor_operation::element_wise::PassThrough,
                                        ck::tensor_operation::element_wise::PassThrough,
                                        AComputeType,
                                        BComputeType>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;

#ifdef DL_KERNELS
        if constexpr(NumDimSpatial == 2 && is_same_v<InLayout, GNHWC> &&
                     is_same_v<WeiLayout, GKYXC> && is_same_v<OutLayout, GNHWK>)
        {
#ifdef CK_ENABLE_FP32
            if constexpr(is_same_v<InDataType, float> && is_same_v<WeiDataType, float> &&
                         is_same_v<OutDataType, float> && is_same_v<AComputeType, float> &&
                         is_same_v<BComputeType, float>)
            {
                add_device_grouped_conv2d_fwd_dl_gnhwc_gkyxc_gnhwk_f32_instances(op_ptrs);
            }
#endif
#ifdef CK_ENABLE_FP16
            if constexpr(is_same_v<InDataType, half_t> && is_same_v<WeiDataType, half_t> &&
                         is_same_v<OutDataType, half_t> && is_same_v<AComputeType, half_t> &&
                         is_same_v<BComputeType, half_t>)
            {
                add_device_grouped_conv2d_fwd_dl_gnhwc_gkyxc_gnhwk_f16_instances(op_ptrs);
            }
#endif
        }
        if constexpr(NumDimSpatial == 2 && is_same_v<InLayout, NHWGC> &&
                     is_same_v<WeiLayout, GKYXC> && is_same_v<OutLayout, NHWGK>)
        {

#ifdef CK_ENABLE_FP32
            if constexpr(is_same_v<InDataType, float> && is_same_v<WeiDataType, float> &&
                         is_same_v<OutDataType, float> && is_same_v<AComputeType, float> &&
                         is_same_v<BComputeType, float>)
            {
                add_device_grouped_conv2d_fwd_dl_nhwgc_gkyxc_nhwgk_f32_instances(op_ptrs);
            }
#endif
#ifdef CK_ENABLE_FP16
            if constexpr(is_same_v<InDataType, half_t> && is_same_v<WeiDataType, half_t> &&
                         is_same_v<OutDataType, half_t> && is_same_v<AComputeType, half_t> &&
                         is_same_v<BComputeType, half_t>)
            {
                add_device_grouped_conv2d_fwd_dl_nhwgc_gkyxc_nhwgk_f16_instances(op_ptrs);
            }
#endif
        }
#endif // DL_KERNELS

#ifdef CK_USE_XDL
        if constexpr(NumDimSpatial == 1 && is_same_v<InLayout, GNWC> &&
                     is_same_v<WeiLayout, GKXC> && is_same_v<OutLayout, GNWK>)
        {
#ifdef CK_ENABLE_FP32
            if constexpr(is_same_v<InDataType, float> && is_same_v<WeiDataType, float> &&
                         is_same_v<OutDataType, float> && is_same_v<AComputeType, float> &&
                         is_same_v<BComputeType, float>)
            {
                add_device_grouped_conv1d_fwd_xdl_gnwc_gkxc_gnwk_f32_instances(op_ptrs);
            }
#endif
#ifdef CK_ENABLE_FP16
            if constexpr(is_same_v<InDataType, half_t> && is_same_v<WeiDataType, half_t> &&
                         is_same_v<OutDataType, half_t> && is_same_v<AComputeType, half_t> &&
                         is_same_v<BComputeType, half_t>)
            {
                add_device_grouped_conv1d_fwd_xdl_gnwc_gkxc_gnwk_f16_instances(op_ptrs);
            }
#endif
#ifdef CK_ENABLE_BF16
            if constexpr(is_same_v<InDataType, ck::bhalf_t> &&
                         is_same_v<WeiDataType, ck::bhalf_t> &&
                         is_same_v<OutDataType, ck::bhalf_t> &&
                         is_same_v<AComputeType, ck::bhalf_t> &&
                         is_same_v<BComputeType, ck::bhalf_t>)
            {
                add_device_grouped_conv1d_fwd_xdl_gnwc_gkxc_gnwk_bf16_instances(op_ptrs);
            }
#endif
#ifdef CK_ENABLE_INT8
            if constexpr(is_same_v<InDataType, int8_t> && is_same_v<WeiDataType, int8_t> &&
                         is_same_v<OutDataType, int8_t> && is_same_v<AComputeType, int8_t> &&
                         is_same_v<BComputeType, int8_t>)
            {
                add_device_grouped_conv1d_fwd_xdl_gnwc_gkxc_gnwk_int8_instances(op_ptrs);
            }
#endif
        }

        if constexpr(NumDimSpatial == 2 && is_same_v<InLayout, GNHWC> &&
                     is_same_v<WeiLayout, GKYXC> && is_same_v<OutLayout, GNHWK>)
        {
#ifdef CK_ENABLE_FP32
            if constexpr(is_same_v<InDataType, float> && is_same_v<WeiDataType, float> &&
                         is_same_v<OutDataType, float> && is_same_v<AComputeType, float> &&
                         is_same_v<BComputeType, float>)
            {
                add_device_grouped_conv2d_fwd_xdl_gnhwc_gkyxc_gnhwk_f32_instances(op_ptrs);
            }
#endif
#ifdef CK_ENABLE_FP16
            if constexpr(is_same_v<InDataType, half_t> && is_same_v<WeiDataType, half_t> &&
                         is_same_v<OutDataType, half_t> && is_same_v<AComputeType, half_t> &&
                         is_same_v<BComputeType, half_t>)
            {
                add_device_grouped_conv2d_fwd_xdl_gnhwc_gkyxc_gnhwk_f16_instances(op_ptrs);
            }
#endif
#ifdef CK_ENABLE_BF16
            if constexpr(is_same_v<InDataType, ck::bhalf_t> &&
                         is_same_v<WeiDataType, ck::bhalf_t> &&
                         is_same_v<OutDataType, ck::bhalf_t> &&
                         is_same_v<AComputeType, ck::bhalf_t> &&
                         is_same_v<BComputeType, ck::bhalf_t>)
            {
                add_device_grouped_conv1d_fwd_xdl_gnhwc_gkyxc_gnhwk_bf16_instances(op_ptrs);
            }
#endif
        }

        if constexpr(NumDimSpatial == 2 && is_same_v<InLayout, NHWGC> &&
                     is_same_v<WeiLayout, GKYXC> && is_same_v<OutLayout, NHWGK>)
        {
#ifdef CK_ENABLE_FP32
            if constexpr(is_same_v<InDataType, float> && is_same_v<WeiDataType, float> &&
                         is_same_v<OutDataType, float> && is_same_v<AComputeType, float> &&
                         is_same_v<BComputeType, float>)
            {
                add_device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_f32_instances(op_ptrs);
            }
#endif
#ifdef CK_ENABLE_FP16
            if constexpr(is_same_v<InDataType, half_t> && is_same_v<WeiDataType, half_t> &&
                         is_same_v<OutDataType, half_t> && is_same_v<AComputeType, half_t> &&
                         is_same_v<BComputeType, half_t>)
            {
                add_device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_f16_instances(op_ptrs);
            }
#endif
#ifdef CK_ENABLE_BF16
            if constexpr(is_same_v<InDataType, ck::bhalf_t> &&
                         is_same_v<WeiDataType, ck::bhalf_t> &&
                         is_same_v<OutDataType, ck::bhalf_t> &&
                         is_same_v<AComputeType, ck::bhalf_t> &&
                         is_same_v<BComputeType, ck::bhalf_t>)
            {
                add_device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_bf16_instances(op_ptrs);
            }
#endif
        }

        if constexpr(NumDimSpatial == 3 && is_same_v<InLayout, GNDHWC> &&
                     is_same_v<WeiLayout, GKZYXC> && is_same_v<OutLayout, GNDHWK>)
        {
#ifdef CK_ENABLE_FP32
            if constexpr(is_same_v<InDataType, float> && is_same_v<WeiDataType, float> &&
                         is_same_v<OutDataType, float> && is_same_v<AComputeType, float> &&
                         is_same_v<BComputeType, float>)
            {
                add_device_grouped_conv3d_fwd_xdl_gndhwc_gkzyxc_gndhwk_f32_instances(op_ptrs);
            }
#endif
#ifdef CK_ENABLE_FP16
            if constexpr(is_same_v<InDataType, half_t> && is_same_v<WeiDataType, half_t> &&
                         is_same_v<OutDataType, half_t> && is_same_v<AComputeType, half_t> &&
                         is_same_v<BComputeType, half_t>)
            {
                add_device_grouped_conv3d_fwd_xdl_gndhwc_gkzyxc_gndhwk_f16_instances(op_ptrs);
            }
#endif
#ifdef CK_ENABLE_BF16
            if constexpr(is_same_v<InDataType, ck::bhalf_t> &&
                         is_same_v<WeiDataType, ck::bhalf_t> &&
                         is_same_v<OutDataType, ck::bhalf_t> &&
                         is_same_v<AComputeType, ck::bhalf_t> &&
                         is_same_v<BComputeType, ck::bhalf_t>)
            {
                add_device_grouped_conv3d_fwd_xdl_gndhwc_gkzyxc_gndhwk_bf16_instances(op_ptrs);
            }
#endif
#ifdef CK_ENABLE_INT8
            if constexpr(is_same_v<InDataType, int8_t> && is_same_v<WeiDataType, int8_t> &&
                         is_same_v<OutDataType, int8_t> && is_same_v<AComputeType, int8_t> &&
                         is_same_v<BComputeType, int8_t>)
            {
                add_device_grouped_conv3d_fwd_xdl_gndhwc_gkzyxc_gndhwk_int8_instances(op_ptrs);
            }
#endif
        }

        if constexpr(NumDimSpatial == 3 && is_same_v<InLayout, NDHWGC> &&
                     is_same_v<WeiLayout, GKZYXC> && is_same_v<OutLayout, NDHWGK>)
        {
#ifdef CK_ENABLE_FP32
            if constexpr(is_same_v<InDataType, float> && is_same_v<WeiDataType, float> &&
                         is_same_v<OutDataType, float> && is_same_v<AComputeType, float> &&
                         is_same_v<BComputeType, float>)
            {
                add_device_grouped_conv3d_fwd_xdl_ndhwgc_gkzyxc_ndhwgk_f32_instances(op_ptrs);
            }
#endif

#ifdef CK_ENABLE_FP8
            if constexpr(is_same_v<InDataType, half_t> && is_same_v<WeiDataType, half_t> &&
                         is_same_v<OutDataType, half_t> && is_same_v<AComputeType, ck::f8_t> &&
                         is_same_v<BComputeType, ck::f8_t>)
            {
                add_device_grouped_conv3d_fwd_xdl_ndhwgc_gkzyxc_ndhwgk_f16_comp_f8_instances(
                    op_ptrs);
            }

            if constexpr(is_same_v<InDataType, ck::f8_t> && is_same_v<WeiDataType, ck::f8_t> &&
                         is_same_v<OutDataType, ck::f8_t> && is_same_v<AComputeType, ck::f8_t> &&
                         is_same_v<BComputeType, ck::f8_t>)
            {
                add_device_grouped_conv3d_fwd_xdl_ndhwgc_gkzyxc_ndhwgk_f8_instances(op_ptrs);
            }
#endif
#ifdef CK_ENABLE_BF8
            if constexpr(is_same_v<InDataType, ck::bf8_t> && is_same_v<WeiDataType, ck::bf8_t> &&
                         is_same_v<OutDataType, ck::f8_t> && is_same_v<AComputeType, ck::bf8_t> &&
                         is_same_v<BComputeType, ck::bf8_t>)
            {
                add_device_grouped_conv3d_fwd_xdl_ndhwgc_gkzyxc_ndhwgk_bf8_instances(op_ptrs);
            }
#endif
#if(defined(CK_ENABLE_FP8) && defined(CK_ENABLE_BF8))
            if constexpr(is_same_v<InDataType, ck::f8_t> && is_same_v<WeiDataType, ck::bf8_t> &&
                         is_same_v<OutDataType, ck::f8_t> && is_same_v<AComputeType, ck::f8_t> &&
                         is_same_v<BComputeType, ck::bf8_t>)
            {
                add_device_grouped_conv3d_fwd_xdl_ndhwgc_gkzyxc_ndhwgk_f8_bf8_instances(op_ptrs);
            }
#endif
#if(defined(CK_ENABLE_FP8) && defined(CK_ENABLE_BF8))
            if constexpr(is_same_v<InDataType, ck::bf8_t> && is_same_v<WeiDataType, ck::f8_t> &&
                         is_same_v<OutDataType, ck::f8_t> && is_same_v<AComputeType, ck::bf8_t> &&
                         is_same_v<BComputeType, ck::f8_t>)
            {
                add_device_grouped_conv3d_fwd_xdl_ndhwgc_gkzyxc_ndhwgk_bf8_f8_instances(op_ptrs);
            }
#endif
#ifdef CK_ENABLE_FP16
            if constexpr(is_same_v<InDataType, half_t> && is_same_v<WeiDataType, half_t> &&
                         is_same_v<OutDataType, half_t> && is_same_v<AComputeType, half_t> &&
                         is_same_v<BComputeType, half_t>)
            {
                add_device_grouped_conv3d_fwd_xdl_ndhwgc_gkzyxc_ndhwgk_f16_instances(op_ptrs);
            }
#endif
#ifdef CK_ENABLE_BF16
            if constexpr(is_same_v<InDataType, ck::bhalf_t> &&
                         is_same_v<WeiDataType, ck::bhalf_t> &&
                         is_same_v<OutDataType, ck::bhalf_t> &&
                         is_same_v<AComputeType, ck::bhalf_t> &&
                         is_same_v<BComputeType, ck::bhalf_t>)
            {
                add_device_grouped_conv3d_fwd_xdl_ndhwgc_gkzyxc_ndhwgk_bf16_instances(op_ptrs);
            }
#endif
#ifdef CK_ENABLE_INT8
            if constexpr(is_same_v<InDataType, int8_t> && is_same_v<WeiDataType, int8_t> &&
                         is_same_v<OutDataType, int8_t> && is_same_v<AComputeType, int8_t> &&
                         is_same_v<BComputeType, int8_t>)
            {
                add_device_grouped_conv3d_fwd_xdl_ndhwgc_gkzyxc_ndhwgk_int8_instances(op_ptrs);
            }
#endif
        }
#endif

#ifdef CK_USE_WMMA
        if constexpr(NumDimSpatial == 2 && is_same_v<InLayout, GNHWC> &&
                     is_same_v<WeiLayout, GKYXC> && is_same_v<OutLayout, GNHWK>)
        {
#ifdef CK_ENABLE_FP16
            if constexpr(is_same_v<InDataType, half_t> && is_same_v<WeiDataType, half_t> &&
                         is_same_v<OutDataType, half_t> && is_same_v<AComputeType, half_t> &&
                         is_same_v<BComputeType, half_t>)
            {
                add_device_grouped_conv2d_fwd_wmma_gnhwc_gkyxc_gnhwk_f16_instances(op_ptrs);
                add_device_grouped_conv2d_fwd_wmma_gnhwc_gkyxc_gnhwk_f16_1x1p0_instances(op_ptrs);
                add_device_grouped_conv2d_fwd_wmma_gnhwc_gkyxc_gnhwk_f16_1x1s1p0_instances(op_ptrs);
                add_device_grouped_conv2d_fwd_wmma_gnhwc_gkyxc_gnhwk_f16_oddc_instances(op_ptrs);
            }
#endif
#ifdef CK_ENABLE_INT8
            if constexpr(is_same_v<InDataType, int8_t> && is_same_v<WeiDataType, int8_t> &&
                         is_same_v<OutDataType, int8_t> && is_same_v<AComputeType, int8_t> &&
                         is_same_v<BComputeType, int8_t>)
            {
                add_device_grouped_conv2d_fwd_wmma_gnhwc_gkyxc_gnhwk_i8_instances(op_ptrs);
                add_device_grouped_conv2d_fwd_wmma_gnhwc_gkyxc_gnhwk_i8_1x1p0_instances(op_ptrs);
                add_device_grouped_conv2d_fwd_wmma_gnhwc_gkyxc_gnhwk_i8_1x1s1p0_instances(op_ptrs);
                add_device_grouped_conv2d_fwd_wmma_gnhwc_gkyxc_gnhwk_i8_oddc_instances(op_ptrs);
            }
#endif
        }

        if constexpr(NumDimSpatial == 2 && is_same_v<InLayout, NHWGC> &&
                     is_same_v<WeiLayout, GKYXC> && is_same_v<OutLayout, NHWGK>)
        {
#ifdef CK_ENABLE_INT8
            if constexpr(is_same_v<InDataType, int8_t> && is_same_v<WeiDataType, int8_t> &&
                         is_same_v<OutDataType, int8_t> && is_same_v<AComputeType, int8_t> &&
                         is_same_v<BComputeType, int8_t>)
            {
                add_device_grouped_conv2d_fwd_wmma_nhwgc_gkyxc_nhwgk_i8_instances(op_ptrs);
                add_device_grouped_conv2d_fwd_wmma_nhwgc_gkyxc_nhwgk_i8_1x1p0_instances(op_ptrs);
                add_device_grouped_conv2d_fwd_wmma_nhwgc_gkyxc_nhwgk_i8_1x1s1p0_instances(op_ptrs);
                add_device_grouped_conv2d_fwd_wmma_nhwgc_gkyxc_nhwgk_i8_oddc_instances(op_ptrs);
            }
#endif
        }

        if constexpr(NumDimSpatial == 3 && is_same_v<InLayout, GNDHWC> &&
                     is_same_v<WeiLayout, GKZYXC> && is_same_v<OutLayout, GNDHWK>)
        {
#ifdef CK_ENABLE_FP16
            if constexpr(is_same_v<InDataType, half_t> && is_same_v<WeiDataType, half_t> &&
                         is_same_v<OutDataType, half_t> && is_same_v<AComputeType, half_t> &&
                         is_same_v<BComputeType, half_t>)
            {
                add_device_grouped_conv3d_fwd_wmma_gndhwc_gkzyxc_gndhwk_f16_instances(op_ptrs);
                add_device_grouped_conv3d_fwd_wmma_gndhwc_gkzyxc_gndhwk_f16_1x1p0_instances(
                    op_ptrs);
                add_device_grouped_conv3d_fwd_wmma_gndhwc_gkzyxc_gndhwk_f16_1x1s1p0_instances(
                    op_ptrs);
                add_device_grouped_conv3d_fwd_wmma_gndhwc_gkzyxc_gndhwk_f16_oddc_instances(op_ptrs);
            }
#endif
#ifdef CK_ENABLE_INT8
            if constexpr(is_same_v<InDataType, int8_t> && is_same_v<WeiDataType, int8_t> &&
                         is_same_v<OutDataType, int8_t> && is_same_v<AComputeType, int8_t> &&
                         is_same_v<BComputeType, int8_t>)
            {
                add_device_grouped_conv3d_fwd_wmma_gndhwc_gkzyxc_gndhwk_i8_instances(op_ptrs);
                add_device_grouped_conv3d_fwd_wmma_gndhwc_gkzyxc_gndhwk_i8_1x1p0_instances(op_ptrs);
                add_device_grouped_conv3d_fwd_wmma_gndhwc_gkzyxc_gndhwk_i8_1x1s1p0_instances(
                    op_ptrs);
                add_device_grouped_conv3d_fwd_wmma_gndhwc_gkzyxc_gndhwk_i8_oddc_instances(op_ptrs);
            }
#endif
        }

        if constexpr(NumDimSpatial == 3 && is_same_v<InLayout, NDHWGC> &&
                     is_same_v<WeiLayout, GKZYXC> && is_same_v<OutLayout, NDHWGK>)
        {
#ifdef CK_ENABLE_FP16
            if constexpr(is_same_v<InDataType, half_t> && is_same_v<WeiDataType, half_t> &&
                         is_same_v<OutDataType, half_t> && is_same_v<AComputeType, half_t> &&
                         is_same_v<BComputeType, half_t>)
            {
                add_device_grouped_conv3d_fwd_wmma_ndhwgc_gkzyxc_ndhwgk_f16_instances(op_ptrs);
                add_device_grouped_conv3d_fwd_wmma_ndhwgc_gkzyxc_ndhwgk_f16_1x1p0_instances(
                    op_ptrs);
                add_device_grouped_conv3d_fwd_wmma_ndhwgc_gkzyxc_ndhwgk_f16_1x1s1p0_instances(
                    op_ptrs);
                add_device_grouped_conv3d_fwd_wmma_ndhwgc_gkzyxc_ndhwgk_f16_oddc_instances(op_ptrs);
            }
#endif
#ifdef CK_ENABLE_INT8
            if constexpr(is_same_v<InDataType, int8_t> && is_same_v<WeiDataType, int8_t> &&
                         is_same_v<OutDataType, int8_t> && is_same_v<AComputeType, int8_t> &&
                         is_same_v<BComputeType, int8_t>)
            {
                add_device_grouped_conv3d_fwd_wmma_ndhwgc_gkzyxc_ndhwgk_i8_instances(op_ptrs);
                add_device_grouped_conv3d_fwd_wmma_ndhwgc_gkzyxc_ndhwgk_i8_1x1p0_instances(op_ptrs);
                add_device_grouped_conv3d_fwd_wmma_ndhwgc_gkzyxc_ndhwgk_i8_1x1s1p0_instances(
                    op_ptrs);
                add_device_grouped_conv3d_fwd_wmma_ndhwgc_gkzyxc_ndhwgk_i8_oddc_instances(op_ptrs);
            }
#endif
        }
#endif

        return op_ptrs;
    }
};

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
