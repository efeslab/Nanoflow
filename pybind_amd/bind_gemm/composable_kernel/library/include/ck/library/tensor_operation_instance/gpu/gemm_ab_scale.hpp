// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <vector>
#include <memory>
#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle_v3_ab_scale.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
#if(defined(CK_ENABLE_BF16) || defined(CK_ENABLE_FP8))
void add_device_gemm_ab_scale_xdl_f8_f8_bf16_mk_nk_mn_128_128_128_comp_default_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleD_ABScale<Row,
                                                            Col,
                                                            Tuple<>,
                                                            Row,
                                                            F8,
                                                            F32,
                                                            F8,
                                                            F32,
                                                            Tuple<>,
                                                            BF16,
                                                            128,
                                                            128,
                                                            128,
                                                            PassThrough,
                                                            PassThrough,
                                                            PassThrough>>>& instances);

void add_device_gemm_ab_scale_xdl_f8_f8_bf16_mk_nk_mn_128_128_128_comp_kpadding_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleD_ABScale<Row,
                                                            Col,
                                                            Tuple<>,
                                                            Row,
                                                            F8,
                                                            F32,
                                                            F8,
                                                            F32,
                                                            Tuple<>,
                                                            BF16,
                                                            128,
                                                            128,
                                                            128,
                                                            PassThrough,
                                                            PassThrough,
                                                            PassThrough>>>& instances);

void add_device_gemm_ab_scale_xdl_f8_f8_bf16_mk_nk_mn_128_128_128_comp_mnpadding_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleD_ABScale<Row,
                                                            Col,
                                                            Tuple<>,
                                                            Row,
                                                            F8,
                                                            F32,
                                                            F8,
                                                            F32,
                                                            Tuple<>,
                                                            BF16,
                                                            128,
                                                            128,
                                                            128,
                                                            PassThrough,
                                                            PassThrough,
                                                            PassThrough>>>& instances);

void add_device_gemm_ab_scale_xdl_f8_f8_bf16_mk_nk_mn_128_128_128_comp_mnkpadding_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleD_ABScale<Row,
                                                            Col,
                                                            Tuple<>,
                                                            Row,
                                                            F8,
                                                            F32,
                                                            F8,
                                                            F32,
                                                            Tuple<>,
                                                            BF16,
                                                            128,
                                                            128,
                                                            128,
                                                            PassThrough,
                                                            PassThrough,
                                                            PassThrough>>>& instances);

void add_device_gemm_ab_scale_xdl_f8_f8_bf16_mk_nk_mn_128_128_128_mem_v1_default_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleD_ABScale<Row,
                                                            Col,
                                                            Tuple<>,
                                                            Row,
                                                            F8,
                                                            F32,
                                                            F8,
                                                            F32,
                                                            Tuple<>,
                                                            BF16,
                                                            128,
                                                            128,
                                                            128,
                                                            PassThrough,
                                                            PassThrough,
                                                            PassThrough>>>& instances);

void add_device_gemm_ab_scale_xdl_f8_f8_bf16_mk_nk_mn_128_128_128_mem_v1_kpadding_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleD_ABScale<Row,
                                                            Col,
                                                            Tuple<>,
                                                            Row,
                                                            F8,
                                                            F32,
                                                            F8,
                                                            F32,
                                                            Tuple<>,
                                                            BF16,
                                                            128,
                                                            128,
                                                            128,
                                                            PassThrough,
                                                            PassThrough,
                                                            PassThrough>>>& instances);

void add_device_gemm_ab_scale_xdl_f8_f8_bf16_mk_nk_mn_128_128_128_mem_v1_mnkpadding_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleD_ABScale<Row,
                                                            Col,
                                                            Tuple<>,
                                                            Row,
                                                            F8,
                                                            F32,
                                                            F8,
                                                            F32,
                                                            Tuple<>,
                                                            BF16,
                                                            128,
                                                            128,
                                                            128,
                                                            PassThrough,
                                                            PassThrough,
                                                            PassThrough>>>& instances);
#endif

template <typename A0DataType,
          typename A1DataType,
          typename B0DataType,
          typename B1DataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout>
struct DeviceOperationInstanceFactory<ck::tensor_operation::device::DeviceGemmMultipleD_ABScale<
    ALayout,
    BLayout,
    Tuple<>,
    CLayout,
    A0DataType,
    A1DataType,
    B0DataType,
    B1DataType,
    Tuple<>,
    CDataType,
    128,
    128,
    128,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough>>
{
    using DeviceOp = DeviceGemmMultipleD_ABScale<ALayout,
                                                 BLayout,
                                                 Tuple<>,
                                                 CLayout,
                                                 A0DataType,
                                                 A1DataType,
                                                 B0DataType,
                                                 B1DataType,
                                                 Tuple<>,
                                                 CDataType,
                                                 128,
                                                 128,
                                                 128,
                                                 ck::tensor_operation::element_wise::PassThrough,
                                                 ck::tensor_operation::element_wise::PassThrough,
                                                 ck::tensor_operation::element_wise::PassThrough>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;

#if(defined(CK_ENABLE_BF16) || defined(CK_ENABLE_FP8))
        if constexpr(is_same_v<A0DataType, f8_t> && is_same_v<B0DataType, f8_t> &&
                     is_same_v<CDataType, bhalf_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                         is_same_v<CLayout, Row>)
            {
                add_device_gemm_ab_scale_xdl_f8_f8_bf16_mk_nk_mn_128_128_128_comp_default_instances(
                    op_ptrs);
                add_device_gemm_ab_scale_xdl_f8_f8_bf16_mk_nk_mn_128_128_128_comp_kpadding_instances(
                    op_ptrs);
                add_device_gemm_ab_scale_xdl_f8_f8_bf16_mk_nk_mn_128_128_128_comp_mnpadding_instances(
                    op_ptrs);
                add_device_gemm_ab_scale_xdl_f8_f8_bf16_mk_nk_mn_128_128_128_comp_mnkpadding_instances(
                    op_ptrs);

                add_device_gemm_ab_scale_xdl_f8_f8_bf16_mk_nk_mn_128_128_128_mem_v1_default_instances(
                    op_ptrs);
                add_device_gemm_ab_scale_xdl_f8_f8_bf16_mk_nk_mn_128_128_128_mem_v1_kpadding_instances(
                    op_ptrs);
                add_device_gemm_ab_scale_xdl_f8_f8_bf16_mk_nk_mn_128_128_128_mem_v1_mnkpadding_instances(
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
