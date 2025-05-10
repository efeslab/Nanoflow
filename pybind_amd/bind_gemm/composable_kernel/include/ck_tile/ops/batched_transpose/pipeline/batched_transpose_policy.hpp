// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/softmax.hpp"
#include "ck_tile/ops/topk.hpp"

namespace ck_tile {

struct BatchedTransposePolicy
{
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeInputDistribution()
    {
        using S = Problem;
        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<>,
                tuple<sequence<S::kMWarpPerBlock, S::kMThreadPerWarp, S::kMPerThread>,
                      sequence<S::kNWarpPerBlock, S::kNThreadPerWarp, S::kNPerThread>>,
                tuple<sequence<1, 2>, sequence<1, 2>>,
                tuple<sequence<0, 0>, sequence<1, 1>>,
                sequence<1, 2>,
                sequence<2, 2>>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeOutputDistribution()
    {
        using S = Problem;
        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<>,
                tuple<sequence<S::kNWarpPerBlock, S::kNThreadPerWarp, S::kNPerThread>,
                      sequence<S::kMWarpPerBlock, S::kMThreadPerWarp, S::kMPerThread>>,
                tuple<sequence<2, 1>, sequence<2, 1>>,
                tuple<sequence<0, 0>, sequence<1, 1>>,
                sequence<2, 1>,
                sequence<2, 2>>{});
    }
};
} // namespace ck_tile
