// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <vector>
#include <memory>
#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm_multiple_d.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
#if defined(CK_ENABLE_FP16) && defined(CK_USE_XDL)
void add_device_gemm_bilinear_xdl_c_shuffle_f16_f16_f16_f16_km_kn_mn_mn_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleD<Col,
                                                    Row,
                                                    Row_Tuple,
                                                    Row,
                                                    F16,
                                                    F16,
                                                    F16_Tuple,
                                                    F16,
                                                    PassThrough,
                                                    PassThrough,
                                                    Bilinear>>>& instances);

void add_device_gemm_bilinear_xdl_c_shuffle_f16_f16_f16_f16_km_nk_mn_mn_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleD<Col,
                                                    Col,
                                                    Row_Tuple,
                                                    Row,
                                                    F16,
                                                    F16,
                                                    F16_Tuple,
                                                    F16,
                                                    PassThrough,
                                                    PassThrough,
                                                    Bilinear>>>& instances);

void add_device_gemm_bilinear_xdl_c_shuffle_f16_f16_f16_f16_mk_kn_mn_mn_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleD<Row,
                                                    Row,
                                                    Row_Tuple,
                                                    Row,
                                                    F16,
                                                    F16,
                                                    F16_Tuple,
                                                    F16,
                                                    PassThrough,
                                                    PassThrough,
                                                    Bilinear>>>& instances);

void add_device_gemm_bilinear_xdl_c_shuffle_f16_f16_f16_f16_mk_nk_mn_mn_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleD<Row,
                                                    Col,
                                                    Row_Tuple,
                                                    Row,
                                                    F16,
                                                    F16,
                                                    F16_Tuple,
                                                    F16,
                                                    PassThrough,
                                                    PassThrough,
                                                    Bilinear>>>& instances);
#endif
#if defined(CK_ENABLE_INT8) && defined(CK_USE_WMMA)
void add_device_gemm_bilinear_wmma_c_shuffle_i8_i8_i8_i8_mk_kn_mn_mn_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleD<Row,
                                                    Row,
                                                    Row_Tuple,
                                                    Row,
                                                    I8,
                                                    I8,
                                                    I8_Tuple,
                                                    I8,
                                                    PassThrough,
                                                    PassThrough,
                                                    Bilinear>>>& instances);

void add_device_gemm_bilinear_wmma_c_shuffle_i8_i8_i8_i8_mk_nk_mn_mn_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleD<Row,
                                                    Col,
                                                    Row_Tuple,
                                                    Row,
                                                    I8,
                                                    I8,
                                                    I8_Tuple,
                                                    I8,
                                                    PassThrough,
                                                    PassThrough,
                                                    Bilinear>>>& instances);

void add_device_gemm_bilinear_wmma_c_shuffle_i8_i8_i8_i8_km_kn_mn_mn_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleD<Col,
                                                    Row,
                                                    Row_Tuple,
                                                    Row,
                                                    I8,
                                                    I8,
                                                    I8_Tuple,
                                                    I8,
                                                    PassThrough,
                                                    PassThrough,
                                                    Bilinear>>>& instances);

void add_device_gemm_bilinear_wmma_c_shuffle_i8_i8_i8_i8_km_nk_mn_mn_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleD<Col,
                                                    Col,
                                                    Row_Tuple,
                                                    Row,
                                                    I8,
                                                    I8,
                                                    I8_Tuple,
                                                    I8,
                                                    PassThrough,
                                                    PassThrough,
                                                    Bilinear>>>& instances);
#endif
// GEMM + Bilinear
template <typename ALayout,
          typename BLayout,
          typename DLayout,
          typename ELayout,
          typename ADataType,
          typename BDataType,
          typename DDataType,
          typename EDataType>
struct DeviceOperationInstanceFactory<ck::tensor_operation::device::DeviceGemmMultipleD<
    ALayout,
    BLayout,
    ck::Tuple<DLayout>,
    ELayout,
    ADataType,
    BDataType,
    ck::Tuple<DDataType>,
    EDataType,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::Bilinear>>
{
    using DeviceOp = DeviceGemmMultipleD<ALayout,
                                         BLayout,
                                         ck::Tuple<DLayout>,
                                         ELayout,
                                         ADataType,
                                         BDataType,
                                         ck::Tuple<DDataType>,
                                         EDataType,
                                         ck::tensor_operation::element_wise::PassThrough,
                                         ck::tensor_operation::element_wise::PassThrough,
                                         ck::tensor_operation::element_wise::Bilinear>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;
#if defined(CK_ENABLE_FP16) && defined(CK_USE_XDL)
        if constexpr(is_same_v<ADataType, half_t> && is_same_v<BDataType, half_t> &&
                     is_same_v<DDataType, half_t> && is_same_v<EDataType, half_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Row> &&
                         is_same_v<DLayout, Row> && is_same_v<ELayout, Row>)
            {
                add_device_gemm_bilinear_xdl_c_shuffle_f16_f16_f16_f16_mk_kn_mn_mn_instances(
                    op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                              is_same_v<DLayout, Row> && is_same_v<ELayout, Row>)
            {
                add_device_gemm_bilinear_xdl_c_shuffle_f16_f16_f16_f16_mk_nk_mn_mn_instances(
                    op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Row> &&
                              is_same_v<DLayout, Row> && is_same_v<ELayout, Row>)
            {
                add_device_gemm_bilinear_xdl_c_shuffle_f16_f16_f16_f16_km_kn_mn_mn_instances(
                    op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Col> &&
                              is_same_v<DLayout, Row> && is_same_v<ELayout, Row>)
            {
                add_device_gemm_bilinear_xdl_c_shuffle_f16_f16_f16_f16_km_nk_mn_mn_instances(
                    op_ptrs);
            }
        }
#endif
#if defined(CK_ENABLE_INT8) && defined(CK_USE_WMMA)
        if constexpr(is_same_v<ADataType, std::int8_t> && is_same_v<BDataType, std::int8_t> &&
                     is_same_v<DDataType, std::int8_t> && is_same_v<EDataType, std::int8_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Row> &&
                         is_same_v<DLayout, Row> && is_same_v<ELayout, Row>)
            {
                add_device_gemm_bilinear_wmma_c_shuffle_i8_i8_i8_i8_mk_kn_mn_mn_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                              is_same_v<DLayout, Row> && is_same_v<ELayout, Row>)
            {
                add_device_gemm_bilinear_wmma_c_shuffle_i8_i8_i8_i8_mk_nk_mn_mn_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Row> &&
                              is_same_v<DLayout, Row> && is_same_v<ELayout, Row>)
            {
                add_device_gemm_bilinear_wmma_c_shuffle_i8_i8_i8_i8_km_kn_mn_mn_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Col> &&
                              is_same_v<DLayout, Row> && is_same_v<ELayout, Row>)
            {
                add_device_gemm_bilinear_wmma_c_shuffle_i8_i8_i8_i8_km_nk_mn_mn_instances(op_ptrs);
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
