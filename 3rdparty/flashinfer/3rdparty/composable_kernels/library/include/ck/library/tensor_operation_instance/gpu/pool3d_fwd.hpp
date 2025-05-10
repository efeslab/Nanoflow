// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_pool_fwd.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

static constexpr auto InOutRank  = 5;
static constexpr auto WindowRank = 3;

static constexpr auto MaxOp = ck::ReduceTensorOp::MAX;
static constexpr auto AvgOp = ck::ReduceTensorOp::AVG;
#ifdef CK_ENABLE_FP16
// FP16
void add_device_pool3d_fwd_ndhwc_f16_instances(
    std::vector<std::unique_ptr<
        DevicePoolFwd<InOutRank, WindowRank, F16, F16, I32, NDHWC, NDHWC, MaxOp, false>>>&);

void add_device_pool3d_fwd_ndhwc_f16_instances(
    std::vector<std::unique_ptr<
        DevicePoolFwd<InOutRank, WindowRank, F16, F16, I32, NDHWC, NDHWC, AvgOp, false>>>&);

// FP16 - return index
void add_device_pool3d_fwd_ndhwc_index_f16_instances(
    std::vector<std::unique_ptr<
        DevicePoolFwd<InOutRank, WindowRank, F16, F16, I32, NDHWC, NDHWC, MaxOp, true>>>&);
#endif
#ifdef CK_ENABLE_BF16
// BF16
void add_device_pool3d_fwd_ndhwc_bf16_instances(
    std::vector<std::unique_ptr<
        DevicePoolFwd<InOutRank, WindowRank, BF16, BF16, I32, NDHWC, NDHWC, MaxOp, false>>>&);

void add_device_pool3d_fwd_ndhwc_bf16_instances(
    std::vector<std::unique_ptr<
        DevicePoolFwd<InOutRank, WindowRank, BF16, BF16, I32, NDHWC, NDHWC, AvgOp, false>>>&);

// BF16 - return index
void add_device_pool3d_fwd_ndhwc_index_bf16_instances(
    std::vector<std::unique_ptr<
        DevicePoolFwd<InOutRank, WindowRank, BF16, BF16, I32, NDHWC, NDHWC, MaxOp, true>>>&);
#endif
#ifdef CK_ENABLE_FP32
// FP32
void add_device_pool3d_fwd_ndhwc_f32_instances(
    std::vector<std::unique_ptr<
        DevicePoolFwd<InOutRank, WindowRank, F32, F32, I32, NDHWC, NDHWC, MaxOp, false>>>&);

void add_device_pool3d_fwd_ndhwc_f32_instances(
    std::vector<std::unique_ptr<
        DevicePoolFwd<InOutRank, WindowRank, F32, F32, I32, NDHWC, NDHWC, AvgOp, false>>>&);

// FP32 - return index
void add_device_pool3d_fwd_ndhwc_index_f32_instances(
    std::vector<std::unique_ptr<
        DevicePoolFwd<InOutRank, WindowRank, F32, F32, I32, NDHWC, NDHWC, MaxOp, true>>>&);
#endif
template <typename InDataType,
          typename OutDataType,
          typename IndexDataType,
          typename InLayout,
          typename OutLayout,
          ck::ReduceTensorOp ReduceOpId,
          bool OutputIndex>
struct DeviceOperationInstanceFactory<ck::tensor_operation::device::DevicePoolFwd<InOutRank,
                                                                                  WindowRank,
                                                                                  InDataType,
                                                                                  OutDataType,
                                                                                  IndexDataType,
                                                                                  InLayout,
                                                                                  OutLayout,
                                                                                  ReduceOpId,
                                                                                  OutputIndex>>
{
    using DeviceOp = DevicePoolFwd<InOutRank,
                                   WindowRank,
                                   InDataType,
                                   OutDataType,
                                   IndexDataType,
                                   InLayout,
                                   OutLayout,
                                   ReduceOpId,
                                   OutputIndex>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;
        if constexpr(is_same_v<InLayout, NDHWC> && is_same_v<OutLayout, NDHWC>)
        {
#ifdef CK_ENABLE_FP16
            if constexpr(is_same_v<InDataType, F16> && is_same_v<OutDataType, F16> &&
                         is_same_v<IndexDataType, I32>)
            {
                if constexpr(OutputIndex && ReduceOpId == MaxOp)
                {
                    add_device_pool3d_fwd_ndhwc_index_f16_instances(op_ptrs);
                }
                else
                {
                    add_device_pool3d_fwd_ndhwc_f16_instances(op_ptrs);
                }
            }
#endif
#ifdef CK_ENABLE_BF16
            else if constexpr(is_same_v<InDataType, BF16> && is_same_v<OutDataType, BF16> &&
                              is_same_v<IndexDataType, I32>)
            {
                if constexpr(OutputIndex && ReduceOpId == MaxOp)
                {
                    add_device_pool3d_fwd_ndhwc_index_bf16_instances(op_ptrs);
                }
                else
                {
                    add_device_pool3d_fwd_ndhwc_bf16_instances(op_ptrs);
                }
            }
#endif
#ifdef CK_ENABLE_FP32
            else if constexpr(is_same_v<InDataType, F32> && is_same_v<OutDataType, F32> &&
                              is_same_v<IndexDataType, I32>)
            {
                if constexpr(OutputIndex && ReduceOpId == MaxOp)
                {
                    add_device_pool3d_fwd_ndhwc_index_f32_instances(op_ptrs);
                }
                else
                {
                    add_device_pool3d_fwd_ndhwc_f32_instances(op_ptrs);
                }
            }
#endif
        }

        return op_ptrs;
    }
};

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
