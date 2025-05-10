// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <array>

#include "device_grouped_gemm.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <index_t NumDTensor = 0>
struct GroupedGemmKernelArgument
{
    const void* p_a_grid;
    const void* p_b_grid;
    std::array<const void*, NumDTensor> p_ds_grid;
    void* p_e_grid;

    index_t M;
    index_t N;
    index_t K;

    index_t StrideA;
    index_t StrideB;
    std::array<index_t, NumDTensor> StrideDs;
    index_t StrideE;
};

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
struct DeviceGroupedGemmFixedNK : DeviceGroupedGemm<ALayout,
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
    virtual void SetDeviceKernelArgs(BaseArgument* p_arg, const void* kernel_args) const = 0;
    virtual size_t GetDeviceKernelArgSize(const BaseArgument* p_arg) const               = 0;
    virtual void SetKBatch(BaseArgument* p_arg, index_t k_batch) const                   = 0;
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
