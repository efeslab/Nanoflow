// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <array>

#include "ck/tensor_operation/gpu/device/device_base.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

// GEMM:
//   input : A0[M, K], B0[K, N],
//   input : D0[M, N], D1[M, N], ...
//   output : E[M, N]
//   C = a_op(A) * b_op(B)
//   E = cde_op(C, D0, D1, ...)
// Assume:
//   D0, D1, ... and E have the same layout
template <typename AsLayout,
          typename BsLayout,
          typename DsLayout,
          typename ELayout,
          typename AsDataType,
          typename BsDataType,
          typename DsDataType,
          typename EDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation>
struct DeviceGemmMultipleABD : public BaseOperator
{
    static constexpr index_t NumATensor = AsDataType::Size();
    static constexpr index_t NumBTensor = BsDataType::Size();
    static constexpr index_t NumDTensor = DsDataType::Size();

    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(std::array<const void*, NumATensor> p_as,
                        std::array<const void*, NumBTensor> p_bs,
                        std::array<const void*, NumDTensor> p_ds,
                        void* p_e,
                        ck::index_t M,
                        ck::index_t N,
                        ck::index_t K,
                        std::array<ck::index_t, NumATensor> StrideAs,
                        std::array<ck::index_t, NumBTensor> StrideBs,
                        std::array<ck::index_t, NumDTensor> StrideDs,
                        ck::index_t StrideE,
                        AElementwiseOperation a_element_op,
                        BElementwiseOperation b_element_op,
                        CDEElementwiseOperation cde_element_op) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
