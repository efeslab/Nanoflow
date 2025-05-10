// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/tensor_operation/gpu/device/impl/device_pool3d_fwd_ndhwc_ndhwc.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename InDataType,
          typename OutDataType,
          typename IndexDataType, // enable if OutputIndex == true
          typename ComputeDataType,
          ck::ReduceTensorOp ReduceOpId,
          bool OutputIndex,
          ck::index_t BlockSize,
          ck::index_t ReduceMThreadClusterSize,
          ck::index_t ReduceKThreadClusterSize,
          ck::index_t ReduceMThreadSliceSize,
          ck::index_t ReduceKThreadSliceSize,
          ck::index_t InSrcOutDstVectorSize>
struct DevicePool2dFwd_NHWC_NHWC : public DevicePool3dFwd_NDHWC_NDHWC<InDataType,
                                                                      OutDataType,
                                                                      IndexDataType,
                                                                      ComputeDataType,
                                                                      ReduceOpId,
                                                                      OutputIndex,
                                                                      BlockSize,
                                                                      ReduceMThreadClusterSize,
                                                                      ReduceKThreadClusterSize,
                                                                      ReduceMThreadSliceSize,
                                                                      ReduceKThreadSliceSize,
                                                                      InSrcOutDstVectorSize>
{
    using DevicePool3D = DevicePool3dFwd_NDHWC_NDHWC<InDataType,
                                                     OutDataType,
                                                     IndexDataType,
                                                     ComputeDataType,
                                                     ReduceOpId,
                                                     OutputIndex,
                                                     BlockSize,
                                                     ReduceMThreadClusterSize,
                                                     ReduceKThreadClusterSize,
                                                     ReduceMThreadSliceSize,
                                                     ReduceKThreadSliceSize,
                                                     InSrcOutDstVectorSize>;

    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_in_dev,
                        void* p_out_dev,
                        void* p_out_indices_dev,
                        std::vector<ck::index_t> input_lengths,
                        std::vector<ck::index_t> window_lengths,
                        std::vector<ck::index_t> output_lengths,
                        std::vector<ck::index_t> input_stride,
                        std::vector<ck::index_t> output_stride,
                        std::vector<ck::index_t> indices_stride,
                        std::vector<ck::index_t> window_strides,
                        std::vector<ck::index_t> window_dilations,
                        std::vector<ck::index_t> input_left_pads,
                        std::vector<ck::index_t> input_right_pads,
                        std::vector<ck::index_t> pooling_dims) override
    {
        static constexpr index_t InOutRank  = 4;
        static constexpr index_t WindowRank = 2;

        if(input_lengths.size() != InOutRank || window_lengths.size() != WindowRank ||
           input_lengths.size() != InOutRank || window_strides.size() != WindowRank ||
           window_dilations.size() != WindowRank || input_left_pads.size() != WindowRank ||
           input_right_pads.size() != WindowRank)
            throw std::runtime_error("dimension is incorrect");

        if(pooling_dims != std::vector<ck::index_t>{2, 3})
            throw std::runtime_error("pooling_dims only support {2, 3} in pool2d so far");

        // NCHW to NCDHW
        input_lengths.insert(input_lengths.begin() + 2, 1);
        output_lengths.insert(output_lengths.begin() + 2, 1);
        input_stride.insert(input_stride.begin() + 2, 0);
        output_stride.insert(output_stride.begin() + 2, 0);
        indices_stride.insert(indices_stride.begin() + 2, 0);

        // YX to ZYX
        window_lengths.insert(window_lengths.begin(), 1);
        window_strides.insert(window_strides.begin(), 0);
        window_dilations.insert(window_dilations.begin(), 0);
        input_left_pads.insert(input_left_pads.begin(), 0);
        input_right_pads.insert(input_right_pads.begin(), 0);

        pooling_dims = {2, 3, 4};

        return DevicePool3D::MakeArgumentPointer(p_in_dev,
                                                 p_out_dev,
                                                 p_out_indices_dev,
                                                 input_lengths,
                                                 window_lengths,
                                                 output_lengths,
                                                 input_stride,
                                                 output_stride,
                                                 indices_stride,
                                                 window_strides,
                                                 window_dilations,
                                                 input_left_pads,
                                                 input_right_pads,
                                                 pooling_dims);
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
