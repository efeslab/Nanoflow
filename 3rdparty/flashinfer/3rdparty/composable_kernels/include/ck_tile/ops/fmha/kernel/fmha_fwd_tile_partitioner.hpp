// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

template <typename BlockFmhaShape_>
struct FmhaFwdTilePartitioner
{
    using BlockFmhaShape = ck_tile::remove_cvref_t<BlockFmhaShape_>;

    static constexpr ck_tile::index_t kM0 = BlockFmhaShape::kM0;
    static constexpr ck_tile::index_t kN0 = BlockFmhaShape::kN0;
    static constexpr ck_tile::index_t kK0 = BlockFmhaShape::kK0;
    static constexpr ck_tile::index_t kN1 = BlockFmhaShape::kN1;
    static constexpr ck_tile::index_t kK1 = BlockFmhaShape::kK1;

    static constexpr const char* name = "shb";

    CK_TILE_HOST static constexpr auto GridSize(ck_tile::index_t batch_size_,
                                                ck_tile::index_t nhead_,
                                                ck_tile::index_t seqlen_q_,
                                                ck_tile::index_t hdim_v_)
    {
        // TODO: this may need tuning
        return dim3(ck_tile::integer_divide_ceil(seqlen_q_, kM0) *
                        ck_tile::integer_divide_ceil(hdim_v_, kN1),
                    nhead_,
                    batch_size_);
    }

    CK_TILE_DEVICE auto operator()(ck_tile::index_t /*seqlen_q*/, ck_tile::index_t hdim_v)
    {
        // const index_t num_tile_m0 = seqlen_q / kM0;
        const index_t num_tile_n1 = ck_tile::integer_divide_ceil(hdim_v, kN1);

        const index_t i_block = blockIdx.x;
        const index_t i_nhead = blockIdx.y;
        const index_t i_batch = blockIdx.z;

        const auto f = [](index_t dividend, index_t divisor) {
            index_t quotient = dividend / divisor;
            index_t modulus  = dividend - quotient * divisor;
            return ck_tile::make_tuple(quotient, modulus);
        };

        const auto [i_tile_m, i_tile_n] = f(i_block, num_tile_n1);

        return ck_tile::make_tuple(i_tile_m, i_tile_n, i_nhead, i_batch);
    }
};

template <typename BlockFmhaShape_>
using FmhaFwdTilePartitioner_SHB = FmhaFwdTilePartitioner<BlockFmhaShape_>;

template <typename BlockFmhaShape_>
struct FmhaFwdTilePartitioner_HBS
{
    using BlockFmhaShape = ck_tile::remove_cvref_t<BlockFmhaShape_>;

    static constexpr ck_tile::index_t kM0 = BlockFmhaShape::kM0;
    static constexpr ck_tile::index_t kN0 = BlockFmhaShape::kN0;
    static constexpr ck_tile::index_t kK0 = BlockFmhaShape::kK0;
    static constexpr ck_tile::index_t kN1 = BlockFmhaShape::kN1;
    static constexpr ck_tile::index_t kK1 = BlockFmhaShape::kK1;

    static constexpr const char* name = "hbs";

    CK_TILE_HOST static constexpr auto GridSize(ck_tile::index_t batch_size_,
                                                ck_tile::index_t nhead_,
                                                ck_tile::index_t seqlen_q_,
                                                ck_tile::index_t hdim_v_)
    {
        // TODO: this may need tuning
        return dim3(nhead_,
                    batch_size_,
                    ck_tile::integer_divide_ceil(seqlen_q_, kM0) *
                        ck_tile::integer_divide_ceil(hdim_v_, kN1));
    }

    CK_TILE_DEVICE auto operator()(ck_tile::index_t /*seqlen_q*/, ck_tile::index_t hdim_v)
    {
        // const index_t num_tile_m0 = seqlen_q / kM0;
        const index_t num_tile_n1 = ck_tile::integer_divide_ceil(hdim_v, kN1);

        const index_t i_block = blockIdx.z;
        const index_t i_nhead = blockIdx.x;
        const index_t i_batch = blockIdx.y;

        const auto f = [](index_t dividend, index_t divisor) {
            index_t quotient = dividend / divisor;
            index_t modulus  = dividend - quotient * divisor;
            return ck_tile::make_tuple(quotient, modulus);
        };

        const auto [i_tile_m, i_tile_n] = f(i_block, num_tile_n1);

        return ck_tile::make_tuple(i_tile_m, i_tile_n, i_nhead, i_batch);
    }
};

} // namespace ck_tile
