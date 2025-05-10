// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core/utility/type_traits.hpp"

namespace ck_tile {

template <typename DataType_,
          index_t kBlockSize_ = 256,
          index_t kMaxRanks_  = 8,
          bool KeepLastDim_   = false>
struct GenericPermuteProblem
{
    using DataType                      = remove_cvref_t<DataType_>;
    static constexpr index_t kBlockSize = kBlockSize_;
    static constexpr index_t kMaxRanks  = kMaxRanks_;
    /* KeepLastDim:
     *  if last dim keep the same? this can help enable vector load
     *   permute(0, 2, 4, 1, 3, 5) -> true
     *   permute(0, 3, 2, 1) -> false
     */
    static constexpr bool KeepLastDim = KeepLastDim_;
    // TODO: not used(?)
};

} // namespace ck_tile
