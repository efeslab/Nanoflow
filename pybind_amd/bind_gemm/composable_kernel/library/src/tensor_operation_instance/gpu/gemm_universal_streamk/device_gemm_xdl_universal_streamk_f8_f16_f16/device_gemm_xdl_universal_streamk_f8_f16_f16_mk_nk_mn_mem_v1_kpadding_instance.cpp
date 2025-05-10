// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#include "device_gemm_xdl_universal_streamk_f8_f16_f16_mk_nk_mn.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_gemm_xdl_universal_streamk_f8_f16_f16_mk_nk_mn_mem_v1_kpadding_instances(
    std::vector<std::unique_ptr<
        DeviceGemm_Streamk_V2<Row, Col, Row, F8, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances)
{
    add_device_operation_instances(
        instances,
        device_gemm_xdl_universal_streamk_f8_f16_f16_mk_nk_mn_mem_instances<Intrawave,
                                                                            GemmKPadding>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
