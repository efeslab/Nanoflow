// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "device_grouped_gemm_splitk.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          typename ADataType,
          typename BDataType,
          typename DsDataType,
          typename EDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation>
struct DeviceGroupedGemmFixedNK : DeviceGroupedGemmSplitK<ALayout,
                                                          BLayout,
                                                          DsLayout,
                                                          ELayout,
                                                          ADataType,
                                                          BDataType,
                                                          DsDataType,
                                                          EDataType,
                                                          AElementwiseOperation,
                                                          BElementwiseOperation,
                                                          CElementwiseOperation>
{
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
