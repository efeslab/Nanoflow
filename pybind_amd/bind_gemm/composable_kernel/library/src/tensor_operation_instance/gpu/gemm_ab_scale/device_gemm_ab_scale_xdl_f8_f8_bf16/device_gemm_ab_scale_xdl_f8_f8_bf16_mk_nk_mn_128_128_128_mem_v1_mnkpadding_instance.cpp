// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "device_gemm_ab_scale_xdl_f8_f8_bf16_mk_nk_mn_128_128_128.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

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
                                                            PassThrough>>>& instances)
{
    add_device_operation_instances(
        instances,
        device_gemm_ab_scale_xdl_f8_f8_bf16_mk_nk_mn_128_128_128_mem_instances<Intrawave,
                                                                               GemmMNKPadding>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
