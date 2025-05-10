// SPDX-License-Identifier: MIT
// Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.

#include "common.hpp"

#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_weight_wmma_cshuffle.hpp"

using InDataType  = F16;
using WeiDataType = F16;
using OutDataType = F16;
using AccDataType = F32;

using InElementOp  = PassThrough;
using WeiElementOp = PassThrough;
using OutElementOp = PassThrough;

template <ck::index_t NDimSpatial>
using DeviceConvBwdWeightInstance =
    ck::tensor_operation::device::DeviceGroupedConvBwdWeight_Wmma_CShuffle<
        NDimSpatial,
        ck::tensor_layout::convolution::GNDHWC,
        ck::tensor_layout::convolution::GKZYXC,
        ck::tensor_layout::convolution::GNDHWK,
        InDataType,           // InDataType
        WeiDataType,          // WeiDataType
        OutDataType,          // OutDataType
        AccDataType,          // AccDataType
        InElementOp,          // InElementwiseOperation
        WeiElementOp,         // WeiElementwiseOperation
        OutElementOp,         // OutElementwiseOperation
        ConvBwdWeightDefault, // ConvolutionBackwardWeightSpecialization
        256,                  // BlockSize
        128,                  // MPerBlock
        128,                  // NPerBlock
        4,                    // K0PerBlock
        8,                    // K1
        16,                   // MPerWMMA
        16,                   // NPerWMMA
        4,                    // MRepeat
        2,                    // NRepeat
        S<4, 64, 1>,          // ABlockTransferThreadClusterLengths_AK0_M_AK1
        S<0, 2, 1>,           // ABlockTransferThreadClusterArrangeOrder
        S<0, 2, 1>,           // ABlockTransferSrcAccessOrder
        1,                    // ABlockTransferSrcVectorDim
        1,                    // ABlockTransferSrcScalarPerVector
        8,                    // ABlockTransferDstScalarPerVector_AK1
        true,                 // ABlockLdsExtraM
        S<4, 64, 1>,          // BBlockTransferThreadClusterLengths_BK0_N_BK1
        S<0, 2, 1>,           // BBlockTransferThreadClusterArrangeOrder
        S<0, 2, 1>,           // BBlockTransferSrcAccessOrder
        1,                    // BBlockTransferSrcVectorDim
        1,                    // BBlockTransferSrcScalarPerVector
        8,                    // BBlockTransferDstScalarPerVector_BK1
        true,                 // BBlockLdsExtraN
        4,
        2,
        S<1, 32, 1, 8>,
        1>;

template <ck::index_t NDimSpatial>
using HostConvBwdWeightInstance = ck::tensor_operation::host::ReferenceConvBwdWeight<NDimSpatial,
                                                                                     InDataType,
                                                                                     WeiDataType,
                                                                                     OutDataType,
                                                                                     InElementOp,
                                                                                     WeiElementOp,
                                                                                     OutElementOp>;

#include "run_grouped_conv_bwd_weight_example.inc"

int main(int argc, char* argv[])
{
    ExecutionConfig config;
    ck::utils::conv::ConvParam conv_param = DefaultConvParam;

    if(!parse_cmd_args(argc, argv, config, conv_param))
    {
        return 1;
    }

    switch(conv_param.num_dim_spatial_)
    {
    case 3: return !run_grouped_conv_bwd_weight<3>(config, conv_param);
    default: break;
    }

    return 1;
}
