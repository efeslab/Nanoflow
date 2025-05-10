// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/integral_constant.hpp"
#include "ck_tile/core/algorithm/coordinate_transform.hpp"
#include "ck_tile/core/container/container_helper.hpp"
#include "ck_tile/core/numeric/math.hpp"
#include "ck_tile/core/tensor/tile_window.hpp"
#include "ck_tile/core/utility/type_traits.hpp"
#include "ck_tile/core/tensor/tile_window.hpp"
#include "ck_tile/core/tensor/null_tile_window.hpp"
#include "ck_tile/core/tensor/null_tensor.hpp"

namespace ck_tile {

template <typename BottomTensorView_,
          typename WindowLengths_,
          typename TileDistribution_,
          index_t NumCoord,
          bool oob_conditional_check = true>
CK_TILE_DEVICE auto load_tile(const tile_window_with_static_distribution<BottomTensorView_,
                                                                         WindowLengths_,
                                                                         TileDistribution_,
                                                                         NumCoord>& tile_window,
                              bool_constant<oob_conditional_check> = {})
{
    return tile_window.load(bool_constant<oob_conditional_check>{});
}

template <typename T,
          typename BottomTensorView_,
          typename WindowLengths_,
          typename TileDistribution_,
          index_t NumCoord,
          bool oob_conditional_check = true>
CK_TILE_DEVICE auto load_tile_raw(T& tile,
                                  const tile_window_with_static_distribution<BottomTensorView_,
                                                                             WindowLengths_,
                                                                             TileDistribution_,
                                                                             NumCoord>& tile_window,
                                  bool_constant<oob_conditional_check> = {})
{
    tile_window.load_raw(tile, bool_constant<oob_conditional_check>{});
}

template <typename LdsTileWindow_,
          typename BottomTensorView_,
          typename WindowLengths_,
          typename TileDistribution_,
          index_t NumCoord>
CK_TILE_DEVICE auto
async_load_tile_raw(LdsTileWindow_&& lds_tile,
                    const tile_window_with_static_distribution<BottomTensorView_,
                                                               WindowLengths_,
                                                               TileDistribution_,
                                                               NumCoord>& tile_window)
{
    return tile_window.async_load(lds_tile);
}

CK_TILE_DEVICE auto async_load_fence(index_t cnt = 0)
{
    asm volatile("s_waitcnt vmcnt(%0)" : : "n"(cnt) : "memory");
}

template <typename WindowLengths>
CK_TILE_DEVICE auto load_tile(const null_tile_window<WindowLengths>&)
{
    return null_tensor{};
}

template <typename T, typename WindowLengths>
CK_TILE_DEVICE auto load_tile_raw(T& /*null_tile*/, const null_tile_window<WindowLengths>&)
{
}

} // namespace ck_tile
