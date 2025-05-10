// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

template <typename XDataType_,
          typename ComputeDataType_,
          typename BlockShape_,
          bool kFastFDiv_,
          bool kWelford_>
struct BlockNormReduceProblem
{
    using XDataType                 = remove_cvref_t<XDataType_>;
    using ComputeDataType           = remove_cvref_t<ComputeDataType_>;
    using BlockShape                = remove_cvref_t<BlockShape_>;
    static constexpr bool kFastFDiv = kFastFDiv_;
    static constexpr bool kWelford  = kWelford_;
};

} // namespace ck_tile
