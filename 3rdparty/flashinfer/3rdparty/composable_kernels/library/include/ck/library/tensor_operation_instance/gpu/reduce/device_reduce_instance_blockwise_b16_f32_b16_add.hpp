// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/data_type.hpp"
#include "ck/utility/reduction_enums.hpp"

#include "ck/library/tensor_operation_instance/gpu/reduce/device_reduce_instance_blockwise.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

// clang-format off
// InDataType | AccDataType | OutDataType | Rank | NumReduceDim | ReduceOperation | InElementwiseOp | AccElementwiseOp | PropagateNan | UseIndex 
extern template void add_device_reduce_instance_blockwise<BF16, F32, BF16, 4, 3, ReduceAdd, PassThrough, PassThrough, false, false>(std::vector<DeviceReducePtr<BF16, F32, BF16, 4, 3, ReduceAdd, PassThrough, PassThrough, false, false>>&); 
extern template void add_device_reduce_instance_blockwise<BF16, F32, BF16, 4, 4, ReduceAdd, PassThrough, PassThrough, false, false>(std::vector<DeviceReducePtr<BF16, F32, BF16, 4, 4, ReduceAdd, PassThrough, PassThrough, false, false>>&); 
extern template void add_device_reduce_instance_blockwise<BF16, F32, BF16, 4, 1, ReduceAdd, PassThrough, PassThrough, false, false>(std::vector<DeviceReducePtr<BF16, F32, BF16, 4, 1, ReduceAdd, PassThrough, PassThrough, false, false>>&); 
extern template void add_device_reduce_instance_blockwise<BF16, F32, BF16, 2, 1, ReduceAdd, PassThrough, PassThrough, false, false>(std::vector<DeviceReducePtr<BF16, F32, BF16, 2, 1, ReduceAdd, PassThrough, PassThrough, false, false>>&);
// clang-format on

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
