// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

template <typename DataType_>
struct BlockSoftmax2DProblem
{
    using DataType = remove_cvref_t<DataType_>;
};

} // namespace ck_tile
