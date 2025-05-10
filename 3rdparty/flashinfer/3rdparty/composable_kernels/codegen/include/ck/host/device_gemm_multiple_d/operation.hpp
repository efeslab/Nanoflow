// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cstdlib>
#include <vector>
#include <string>
#include "ck/host/types.hpp"
#include "ck/host/operation/gemm.hpp"
#include "ck/host/device_gemm_multiple_d/problem.hpp"

namespace ck {
namespace host {
namespace device_gemm_multiple_d {

struct Operation_Xdl_CShuffle
{
    static std::vector<std::vector<Operation_Xdl_CShuffle>> CreateOperations();
    static std::vector<Operation_Xdl_CShuffle> CreateOperations(const Problem& prob);
    TensorDesc A{};
    TensorDesc B{};
    DataType acc               = DataType::Float;
    DataType cs_type           = DataType::Half;
    std::vector<TensorDesc> Ds = {};
    TensorDesc E{};
    std::string a_elem_op           = PassThrough;
    std::string b_elem_op           = PassThrough;
    std::string cde_elem_op         = Bilinear;
    std::string gemm_specialization = "ck::tensor_operation::device::GemmSpecialization::Default";
    operation::TileDesc tile_desc{};
    operation::BlockTransferDesc a_block_transfer{};
    operation::BlockTransferDesc b_block_transfer{};
    operation::CShuffleDesc cshuffle{};
    operation::CBlockTransferDesc c_block_transfer{};

    Solution ToSolution() const;
};

} // namespace device_gemm_multiple_d
} // namespace host
} // namespace ck
