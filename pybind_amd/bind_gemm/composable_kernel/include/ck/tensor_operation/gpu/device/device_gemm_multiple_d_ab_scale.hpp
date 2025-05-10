// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <array>

#include "ck/tensor_operation/gpu/device/device_base.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

// GEMM:
//   input : A[M, K], B[K, N],
//   input : D0[M, N], D1[M, N], ...
//   output : E[M, N]
//   C = a_op(A) * b_op(B)
//   E = cde_op(C, D0, D1, ...)
// Assume:
//   D0, D1, ... and E have the same layout
template <typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          typename ADataType,
          typename AScaleType,
          typename BDataType,
          typename BScaleType,
          typename DsDataType,
          typename EDataType,
          index_t ScaleBlockM,
          index_t ScaleBlockN,
          index_t ScaleBlockK,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation>
struct DeviceGemmMultipleD_ABScale : public BaseOperator
{
    static constexpr index_t NumDTensor = DsDataType::Size();

    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_a,
                        const void* p_b,
                        std::array<const void*, NumDTensor> p_ds,
                        void* p_e,
                        const ck::index_t M,
                        const ck::index_t N,
                        const ck::index_t K,
                        const ck::index_t StrideA,
                        const ck::index_t StrideB,
                        const std::array<ck::index_t, NumDTensor> StrideDs,
                        const ck::index_t StrideE,
                        const void* p_a_scale,
                        const void* p_b_scale,
                        AElementwiseOperation a_element_op,
                        BElementwiseOperation b_element_op,
                        CDEElementwiseOperation cde_element_op) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
