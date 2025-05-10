// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include <string>
#include <type_traits>

#define VectorLoadSize 16

namespace ck_tile {

template <typename InputType_,
          typename BlockTile,  // Sequence<...
          typename WarpTile,   // Sequence<...
          typename ThreadTile, // Sequence<...
          bool kPadM_ = true,
          bool kPadN_ = true>
struct BatchedTransposeProblem
{
    using InputType = remove_cvref_t<InputType_>;

    static constexpr index_t kMPerThread = ThreadTile::at(number<0>{});
    static constexpr index_t kNPerThread = ThreadTile::at(number<1>{});

    static constexpr index_t kMPerWarp = WarpTile::at(number<0>{});
    static constexpr index_t kNPerWarp = WarpTile::at(number<1>{});

    static constexpr index_t kMThreadPerWarp = kMPerWarp / kMPerThread;
    static constexpr index_t kNThreadPerWarp = kNPerWarp / kNPerThread;

    static constexpr index_t kMPerBlock = BlockTile::at(number<0>{});
    static constexpr index_t kNPerBlock = BlockTile::at(number<1>{});

    static constexpr index_t kMWarpPerBlock = kMPerBlock / kMPerWarp;
    static constexpr index_t kNWarpPerBlock = kNPerBlock / kNPerWarp;

    static constexpr index_t kBlockSize =
        kMThreadPerWarp * kNThreadPerWarp * kMWarpPerBlock * kNWarpPerBlock;

    static constexpr bool kPadM = kPadM_;
    static constexpr bool kPadN = kPadN_;

    static constexpr index_t AlignmentM = kPadM ? VectorLoadSize / sizeof(InputType) : 1; // TODO
    static constexpr index_t AlignmentN = kPadN ? VectorLoadSize / sizeof(InputType) : 1;
};
} // namespace ck_tile
