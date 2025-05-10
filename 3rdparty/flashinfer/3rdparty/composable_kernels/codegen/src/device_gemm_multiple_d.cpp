
// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include "ck/host/device_gemm_multiple_d/problem.hpp"
#include "ck/host/device_gemm_multiple_d/operation.hpp"
#include "ck/host/utils.hpp"
#include <algorithm>

namespace ck {
namespace host {
namespace device_gemm_multiple_d {

std::string Problem::GetIncludeHeader() const
{
    return "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle.hpp";
}

std::vector<Solution> Problem::GetSolutions(const std::string& arch) const
{
    if(get_xdlop_archs().count(arch) == 0)
        return {};
    auto ops = ck::host::device_gemm_multiple_d::Operation_Xdl_CShuffle::CreateOperations(*this);
    std::vector<Solution> result;
    std::transform(ops.begin(), ops.end(), std::back_inserter(result), [&](const auto& op) {
        return op.ToSolution();
    });
    return result;
}

} // namespace device_gemm_multiple_d
} // namespace host
} // namespace ck