// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

// this epilogue just store out a M*N matrix, row major

template <typename AccDataType_, typename ODataType_, bool kPadM_, bool kPadN_>
struct Default2DEpilogueProblem
{
    using AccDataType           = remove_cvref_t<AccDataType_>;
    using ODataType             = remove_cvref_t<ODataType_>;
    static constexpr bool kPadM = kPadM_;
    static constexpr bool kPadN = kPadN_;
};

template <typename Problem_, typename Policy_ = void>
struct Default2DEpilogue
{
    using Problem               = remove_cvref_t<Problem_>;
    using AccDataType           = remove_cvref_t<typename Problem::AccDataType>;
    using ODataType             = remove_cvref_t<typename Problem::ODataType>;
    static constexpr bool kPadM = Problem::kPadM;
    static constexpr bool kPadN = Problem::kPadN;

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize() { return 0; }

    // TODO: this function assume store out vector size is the same as OAccTile last dimension size
    //       how do we fix this ?
    template <typename ODramWindowTmp, typename OAccTile>
    CK_TILE_DEVICE auto operator()(ODramWindowTmp& o_dram_window_tmp, const OAccTile& o_acc_tile)
    {

        // TODO: this is ugly
        if constexpr(kPadM || kPadN)
        {
            store_tile_raw(o_dram_window_tmp, cast_tile<ODataType>(o_acc_tile));
            buffer_store_fence();
        }
        else
        {
            store_tile(o_dram_window_tmp, cast_tile<ODataType>(o_acc_tile));
        }
    }
};
} // namespace ck_tile
