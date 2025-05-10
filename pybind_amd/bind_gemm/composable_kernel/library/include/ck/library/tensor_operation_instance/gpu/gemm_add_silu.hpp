// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cstdlib>
#include <vector>
#include <memory>

#include "ck/ck.hpp"
#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm_multiple_d.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_gemm_add_silu_xdl_c_shuffle_f16_i8_f16_f16_mk_kn_mn_mn_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleD<Row,
                                                    Row,
                                                    Row_Tuple,
                                                    Row,
                                                    F16,
                                                    I8,
                                                    F16_Tuple,
                                                    F16,
                                                    PassThrough,
                                                    PassThrough,
                                                    AddSilu>>>&);

void add_device_gemm_add_silu_xdl_c_shuffle_bf16_i8_bf16_bf16_mk_kn_mn_mn_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleD<Row,
                                                    Row,
                                                    Row_Tuple,
                                                    Row,
                                                    BF16,
                                                    I8,
                                                    BF16_Tuple,
                                                    BF16,
                                                    PassThrough,
                                                    PassThrough,
                                                    AddSilu>>>&);

// GEMM + Add + Silu
template <typename ALayout,
          typename BLayout,
          typename D0Layout,
          typename ELayout,
          typename ADataType,
          typename BDataType,
          typename D0DataType,
          typename EDataType>
struct DeviceOperationInstanceFactory<
    ck::tensor_operation::device::DeviceGemmMultipleD<ALayout,
                                                      BLayout,
                                                      ck::Tuple<D0Layout>,
                                                      ELayout,
                                                      ADataType,
                                                      BDataType,
                                                      ck::Tuple<D0DataType>,
                                                      EDataType,
                                                      PassThrough,
                                                      PassThrough,
                                                      AddSilu>>
{
    using DeviceOp = DeviceGemmMultipleD<ALayout,
                                         BLayout,
                                         ck::Tuple<D0Layout>,
                                         ELayout,
                                         ADataType,
                                         BDataType,
                                         ck::Tuple<D0DataType>,
                                         EDataType,
                                         PassThrough,
                                         PassThrough,
                                         AddSilu>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;

#if defined(CK_ENABLE_INT8) && defined(CK_ENABLE_FP16)
        if constexpr(is_same_v<ADataType, half_t> && is_same_v<BDataType, int8_t> &&
                     is_same_v<D0DataType, half_t> && is_same_v<EDataType, half_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Row> &&
                         is_same_v<D0Layout, Row> && is_same_v<ELayout, Row>)
            {
                add_device_gemm_add_silu_xdl_c_shuffle_f16_i8_f16_f16_mk_kn_mn_mn_instances(
                    op_ptrs);
            }
        }
#endif

#if defined(CK_ENABLE_INT8) && defined(CK_ENABLE_BF16)
        if constexpr(is_same_v<ADataType, ck::bhalf_t> && is_same_v<BDataType, int8_t> &&
                     is_same_v<D0DataType, ck::bhalf_t> && is_same_v<EDataType, ck::bhalf_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Row> &&
                         is_same_v<D0Layout, Row> && is_same_v<ELayout, Row>)
            {
                add_device_gemm_add_silu_xdl_c_shuffle_bf16_i8_bf16_bf16_mk_kn_mn_mn_instances(
                    op_ptrs);
            }
        }
#endif

        return op_ptrs;
    }
};

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
