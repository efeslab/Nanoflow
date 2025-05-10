// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cstdlib>
#include <vector>
#include <string>
#include "ck/host/types.hpp"

namespace ck {
namespace host {
namespace device_gemm_multiple_d {

struct Problem
{
    std::size_t M                    = 0;
    std::size_t N                    = 0;
    std::size_t K                    = 0;
    bool TransA                      = false;
    bool TransB                      = false;
    bool TransE                      = false;
    std::vector<bool> DsTrans        = {};
    DataType ADataType               = DataType::Half;
    DataType BDataType               = DataType::Half;
    DataType EDataType               = DataType::Half;
    std::vector<DataType> DsDataType = {};
    std::string AElementOp           = PassThrough;
    std::string BElementOp           = PassThrough;
    std::string CDEElementOp         = PassThrough;

    std::string GetIncludeHeader() const;

    std::vector<Solution> GetSolutions(const std::string& arch) const;
};

} // namespace device_gemm_multiple_d
} // namespace host
} // namespace ck
