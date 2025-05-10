// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "device_gemm_multiply_multiply_xdl_f8_f8_f16_mk_nk_mn.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_gemm_multiply_multiply_xdl_f8_f8_f16_mk_nk_mn_mem_v2_kpadding_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleDSplitK<Row,
                                                          Col,
                                                          Tuple<Row, Col>,
                                                          Row,
                                                          F8,
                                                          F8,
                                                          Tuple<F32, F32>,
                                                          F16,
                                                          PassThrough,
                                                          PassThrough,
                                                          MultiplyMultiply>>>& instances)
{
    add_device_operation_instances(
        instances,
        device_gemm_multiply_multiply_xdl_f8_f8_f16_mk_nk_mn_mem_instances<Interwave,
                                                                           GemmKPadding>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
