// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include "ck/utility/reduction_enums.hpp"
#include "ck/library/tensor_operation_instance/gpu/reduce/device_reduce_instance_multiblock_atomic_add.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

// clang-format off
// InDataType | AccDataType | OutDataType | Rank | NumReduceDim | ReduceOperation | InElementwiseOp | AccElementwiseOp | PropagateNan | UseIndex 
template void add_device_reduce_instance_multiblock_atomic_add<BF16, F32, F32, 4, 3, ReduceAdd, PassThrough, UnaryDivide, false, false>(std::vector<DeviceReducePtr<BF16, F32, F32, 4, 3, ReduceAdd, PassThrough, UnaryDivide, false, false>>&); 
template void add_device_reduce_instance_multiblock_atomic_add<BF16, F32, F32, 4, 4, ReduceAdd, PassThrough, UnaryDivide, false, false>(std::vector<DeviceReducePtr<BF16, F32, F32, 4, 4, ReduceAdd, PassThrough, UnaryDivide, false, false>>&); 
template void add_device_reduce_instance_multiblock_atomic_add<BF16, F32, F32, 4, 1, ReduceAdd, PassThrough, UnaryDivide, false, false>(std::vector<DeviceReducePtr<BF16, F32, F32, 4, 1, ReduceAdd, PassThrough, UnaryDivide, false, false>>&); 
template void add_device_reduce_instance_multiblock_atomic_add<BF16, F32, F32, 2, 1, ReduceAdd, PassThrough, UnaryDivide, false, false>(std::vector<DeviceReducePtr<BF16, F32, F32, 2, 1, ReduceAdd, PassThrough, UnaryDivide, false, false>>&);
// clang-format on

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
