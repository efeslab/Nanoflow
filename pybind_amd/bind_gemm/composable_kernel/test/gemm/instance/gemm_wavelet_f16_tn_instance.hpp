// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include <memory>
#include <vector>

#include "include/ck/tensor_operation/gpu/device/device_base.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_gemm_wavelet_f16_tn_256x256(std::vector<std::unique_ptr<BaseOperator>>& instances);

void add_gemm_wavelet_f16_tn_256x128(std::vector<std::unique_ptr<BaseOperator>>& instances);

void add_gemm_wavelet_f16_tn_128x128(std::vector<std::unique_ptr<BaseOperator>>& instances);

void add_gemm_wavelet_f16_tn_128x64(std::vector<std::unique_ptr<BaseOperator>>& instances);

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
