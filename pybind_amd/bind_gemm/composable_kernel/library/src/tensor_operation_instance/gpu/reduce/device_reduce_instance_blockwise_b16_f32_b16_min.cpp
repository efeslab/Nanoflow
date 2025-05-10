// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include "ck/utility/reduction_enums.hpp"
#include "ck/library/tensor_operation_instance/gpu/reduce/device_reduce_instance_blockwise.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

// clang-format off
// InDataType | AccDataType | OutDataType | Rank | NumReduceDim | ReduceOperation | InElementwiseOp | AccElementwiseOp | PropagateNan | UseIndex 
template void add_device_reduce_instance_blockwise<BF16, F32, BF16, 4, 3, ReduceMin, PassThrough, PassThrough, false, false>(std::vector<DeviceReducePtr<BF16, F32, BF16, 4, 3, ReduceMin, PassThrough, PassThrough, false, false>>&);
template void add_device_reduce_instance_blockwise<BF16, F32, BF16, 4, 4, ReduceMin, PassThrough, PassThrough, false, false>(std::vector<DeviceReducePtr<BF16, F32, BF16, 4, 4, ReduceMin, PassThrough, PassThrough, false, false>>&);
template void add_device_reduce_instance_blockwise<BF16, F32, BF16, 4, 1, ReduceMin, PassThrough, PassThrough, false, false>(std::vector<DeviceReducePtr<BF16, F32, BF16, 4, 1, ReduceMin, PassThrough, PassThrough, false, false>>&);
template void add_device_reduce_instance_blockwise<BF16, F32, BF16, 2, 1, ReduceMin, PassThrough, PassThrough, false, false>(std::vector<DeviceReducePtr<BF16, F32, BF16, 2, 1, ReduceMin, PassThrough, PassThrough, false, false>>&);
template void add_device_reduce_instance_blockwise<BF16, F32, BF16, 4, 3, ReduceMin, PassThrough, PassThrough, false, true>(std::vector<DeviceReducePtr<BF16, F32, BF16, 4, 3, ReduceMin, PassThrough, PassThrough, false, true>>&);
template void add_device_reduce_instance_blockwise<BF16, F32, BF16, 4, 4, ReduceMin, PassThrough, PassThrough, false, true>(std::vector<DeviceReducePtr<BF16, F32, BF16, 4, 4, ReduceMin, PassThrough, PassThrough, false, true>>&);
template void add_device_reduce_instance_blockwise<BF16, F32, BF16, 4, 1, ReduceMin, PassThrough, PassThrough, false, true>(std::vector<DeviceReducePtr<BF16, F32, BF16, 4, 1, ReduceMin, PassThrough, PassThrough, false, true>>&);
template void add_device_reduce_instance_blockwise<BF16, F32, BF16, 2, 1, ReduceMin, PassThrough, PassThrough, false, true>(std::vector<DeviceReducePtr<BF16, F32, BF16, 2, 1, ReduceMin, PassThrough, PassThrough, false, true>>&);
// clang-format on

} // namespace instance
} // namespace device
} // namespace tensor_operation

} // namespace ck
