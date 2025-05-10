// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

enum struct GenericAttentionMaskEnum
{
    NO_MASK = 0,

    // below enum could be causal, or sliding window
    MASK_FROM_TOP_LEFT     = 1,
    MASK_FROM_BOTTOM_RIGHT = 2,

    // this enum maybe not used by xformer/FA, since it's hard to
    // specify left/right window for varlen case. put it here for
    // debug purpose
    MASK_GENERIC,
};

// clang-format off
/*  generic Attention Mask Coordinate
    use x(horizontal axis), y(vertical axis) to describe mask.
    top-left corner is origin

    x=1/y=5(top-left)  x=4/y=5(botm-r)    x=6/y=5            x=8/y=5(no mask)
    1 * * * * * * *    1 1 1 1 * * * *    1 1 1 1 1 1 * *    1 1 1 1 1 1 1 1
    1 1 * * * * * *    1 1 1 1 1 * * *    1 1 1 1 1 1 1 *    1 1 1 1 1 1 1 1
    1 1 1 * * * * *    1 1 1 1 1 1 * *    1 1 1 1 1 1 1 1    1 1 1 1 1 1 1 1
    1 1 1 1 * * * *    1 1 1 1 1 1 1 *    1 1 1 1 1 1 1 1    1 1 1 1 1 1 1 1
    1 1 1 1 1 * * *    1 1 1 1 1 1 1 1    1 1 1 1 1 1 1 1    1 1 1 1 1 1 1 1
    l=7,-1/r=0(tl)     l=7,-1/r=0(br)

    x=1/y=2            x=4/y=2            x=6/y=2            x=8/y=2
    1 * * * * * * *    1 1 1 1 * * * *    1 1 1 1 1 1 * *    1 1 1 1 1 1 1 1
    1 1 * * * * * *    1 1 1 1 1 * * *    1 1 1 1 1 1 1 *    1 1 1 1 1 1 1 1
    * 1 1 * * * * *    * 1 1 1 1 1 * *    * 1 1 1 1 1 1 1    * 1 1 1 1 1 1 1
    * * 1 1 * * * *    * * 1 1 1 1 1 *    * * 1 1 1 1 1 1    * * 1 1 1 1 1 1
    * * * 1 1 * * *    * * * 1 1 1 1 1    * * * 1 1 1 1 1    * * * 1 1 1 1 1
    l=1/r=0(tl)        l=1/r=3(tl)        l=1/r=5(tl)        l=1/r=7(tl)
                       l=4/r=0(br)        l=4/r=2(br)        l=4/r=4(br)

                       x=4/y=-1           x=6/y=-1            x=8/y=-1
                       * * 1 1 * * * *    * * 1 1 1 1 * *    * * 1 1 1 1 1 1
                       * * * 1 1 * * *    * * * 1 1 1 1 *    * * * 1 1 1 1 1
                       * * * * 1 1 * *    * * * * 1 1 1 1    * * * * 1 1 1 1
                       * * * * * 1 1 *    * * * * * 1 1 1    * * * * * 1 1 1
                       * * * * * * 1 1    * * * * * * 1 1    * * * * * * 1 1

    x=-2/y=5           x=1/y=5(top-left)  x=0/y=5(botm-r)
    * * * * * * * *    1 * * *            * * * *
    * * * * * * * *    1 1 * *            1 * * *
    * * * * * * * *    1 1 1 *            1 1 * *
    1 * * * * * * *    1 1 1 1            1 1 1 *
    1 1 * * * * * *    1 1 1 1            1 1 1 1

    Validations:
        x + y > 1 (x + y >= 2)

    Note:
        y = seq_q, x = 1 -> top-left
        y = seq_q, x = seq_k - seq_q + 1 -> bottom-right
        y < seq_q, x < seq_k -> local-attn
        y = seq_q, x = seq_k -> no mask

*/
namespace impl {
    template <bool IsMasking_, bool IsLocal_> struct MaskName;
    template<> struct MaskName<false, false> { static constexpr const char * name = "mn"; };
    template<> struct MaskName<false, true> { static constexpr const char * name = "mn"; };
    template<> struct MaskName<true, false> { static constexpr const char * name = "mc"; };
    template<> struct MaskName<true, true> { static constexpr const char * name = "mg"; };
}
// clang-format on

template <bool IsMasking_ = true, bool IsLocal_ = false>
struct GenericAttentionMask
{
    static constexpr bool IsMasking = IsMasking_; // false will disable masking
    static constexpr bool IsLocal   = IsLocal_;   // if true, upper/lower area could have mask,
                                                  // else only upper-right could have mask

    static constexpr const char* name = impl::MaskName<IsMasking, IsLocal>::name;

    CK_TILE_HOST_DEVICE GenericAttentionMask(index_t y_total_, index_t x_total_)
        : GenericAttentionMask(0, 0, y_total_, x_total_)
    {
    }

    CK_TILE_HOST_DEVICE
    GenericAttentionMask(index_t y_, index_t x_, index_t y_total_, index_t x_total_)
        : y(y_), x(x_), y_total(y_total_), x_total(x_total_)
    {
    }
    template <typename MaskCoordinates>
    CK_TILE_HOST_DEVICE GenericAttentionMask(const MaskCoordinates& mask_coord)
        : y(mask_coord.at(number<0>{})),
          x(mask_coord.at(number<1>{})),
          y_total(mask_coord.at(number<2>{})),
          x_total(mask_coord.at(number<3>{}))
    {
    }

    // to get the loop length along X axis, return index:[start, end), end-start=length
    // use this if need loop over X axis tile by tile (like k-seqlen loopover)
    // TODO: x_end still could be negative, so end-start could be negative(need check)
    template <index_t YTile, index_t XTile>
    CK_TILE_HOST_DEVICE constexpr auto
    GetTileRangeAlongX(index_t i_y, number<YTile>, number<XTile>) const
    {
        if constexpr(!IsMasking)
        {
            return ck_tile::make_tuple(0, x_total);
        }
        else
        {
            // get the tile start/end range assum we loop over along X tile by tile
            index_t x_start = [&]() {
                if constexpr(IsLocal)
                {
                    index_t tmp = max(-y + i_y + 1, 0);
                    return (tmp / XTile) * XTile; // round to tile aligned
                }
                else
                {
                    return 0;
                }
            }();

            // TODO: end could be negative, we ignore clamp here, and let caller to check
            //      ... in which case end-start is negative
            index_t x_end = [&]() {
                index_t tmp = min(i_y + YTile - 1 + x, x_total);
                return ((tmp + XTile - 1) / XTile) * XTile;
            }();

            return ck_tile::make_tuple(x_start, x_end);
        }
    }

    // per-pixel check if out-of-bound, if true, need mask a value(like -INF)
    CK_TILE_HOST_DEVICE constexpr auto IsOutOfBound(index_t i_y, index_t i_x) const
    {
        if constexpr(!IsMasking)
        {
            return i_x >= x_total;
        }
        else
        {
            // no need to do min/max here, since i_x will never be < 0 or >= x_total
            index_t x_start = -y + i_y + 1;
            index_t x_end   = min(i_y + x, x_total);

            if constexpr(IsLocal)
            {
                return i_x < x_start || i_x >= x_end;
            }
            else
            {
                return i_x >= x_end;
            }
        }
    }

    // if current tile is at the edge, means need per-pixel mask check.
    // otherwise no need to check per-pixel
    // Attention! assume the idex passed in this function is with in range of GetTileRangeAlongX()
    // can be used as a fast-path to decide if do per-pixel check or not
    template <index_t TileHeight, index_t TileWidth>
    CK_TILE_HOST_DEVICE constexpr auto
    IsEdgeTile(index_t i_tile_top, index_t i_tile_left, number<TileHeight>, number<TileWidth>) const
    {
        if constexpr(IsLocal)
        {
            // check top-right corner > x or left-borrom corner < x
            index_t i_tile_right  = i_tile_left + TileWidth;
            index_t i_tile_bottom = i_tile_top + TileHeight;
            index_t x_end         = min(i_tile_top + x, x_total);

            bool top_right_edge          = i_tile_right > (i_tile_top + x);
            bool bottom_left_edge        = i_tile_bottom > (i_tile_left + y);
            bool is_partial_out_of_bound = i_tile_right > x_end; // only consider right-pad for now

            return top_right_edge || bottom_left_edge || is_partial_out_of_bound;
        }
        else
        {
            // only need to check top-right corner > x
            index_t i_tile_right = i_tile_left + TileWidth;
            index_t x_end        = min(i_tile_top + x, x_total);

            bool top_right_edge = i_tile_right > x_end;
            return top_right_edge;
        }
    }

    private:
    index_t y, x;
    index_t y_total, x_total;
};

// clang-format off
namespace impl {
    template <bool IsMasking_> struct SimplifiedMaskName;
    template<> struct SimplifiedMaskName<false> { static constexpr const char * name = "nomask"; };
    template<> struct SimplifiedMaskName<true> { static constexpr const char * name = "mask"; };
}
// clang-format on

// this version only have 2 variation: masking and non-masking
// This is more friendly to codegen (e.g. need generate less kernel)
// ... with the trade-off that may have more instruction in causal mode
template <bool IsMasking_ = true>
struct SimplifiedGenericAttentionMask
{
    static constexpr bool IsMasking = IsMasking_; // false will disable masking

    static constexpr const char* name = impl::SimplifiedMaskName<IsMasking>::name;

    CK_TILE_HOST_DEVICE SimplifiedGenericAttentionMask(index_t y_total_, index_t x_total_)
        : SimplifiedGenericAttentionMask(0, 0, y_total_, x_total_)
    {
    }

    CK_TILE_HOST_DEVICE
    SimplifiedGenericAttentionMask(index_t y_, index_t x_, index_t y_total_, index_t x_total_)
        : y(y_), x(x_), y_total(y_total_), x_total(x_total_)
    {
    }
    template <typename MaskCoordinates>
    CK_TILE_HOST_DEVICE SimplifiedGenericAttentionMask(const MaskCoordinates& mask_coord)
        : y(mask_coord.at(number<0>{})),
          x(mask_coord.at(number<1>{})),
          y_total(mask_coord.at(number<2>{})),
          x_total(mask_coord.at(number<3>{}))
    {
    }

    // to get the loop length along X axis, return index:[start, end), end-start=length
    // use this if need loop over X axis tile by tile (like k-seqlen loopover)
    // TODO: x_end still could be negative, so end-start could be negative(need check)
    template <index_t YTile, index_t XTile>
    CK_TILE_HOST_DEVICE constexpr auto
    GetTileRangeAlongX(index_t i_y, number<YTile>, number<XTile>) const
    {
        if constexpr(!IsMasking)
        {
            return ck_tile::make_tuple(0, x_total);
        }
        else
        {
            // get the tile start/end range assum we loop over along X tile by tile
            index_t x_start = [&]() {
                index_t tmp = max(-y + i_y + 1, 0);
                return (tmp / XTile) * XTile; // round to tile aligned
            }();

            // TODO: end could be negative, we ignore clamp here, and let caller to check
            //      ... in which case end-start is negative
            index_t x_end = [&]() {
                index_t tmp = min(i_y + YTile - 1 + x, x_total);
                return ((tmp + XTile - 1) / XTile) * XTile;
            }();

            return ck_tile::make_tuple(x_start, x_end);
        }
    }

    // per-pixel check if out-of-bound, if true, need mask a value(like -INF)
    CK_TILE_HOST_DEVICE constexpr auto IsOutOfBound(index_t i_y, index_t i_x) const
    {
        if constexpr(!IsMasking)
        {
            // the only case that need do following compare is under kPadSeqLenK
            // ... for non-masking kernel.
            return i_x >= x_total;
        }
        else
        {
            index_t x_start = -y + i_y + 1;          // this could be negative, but it's fine
            index_t x_end   = min(i_y + x, x_total); // need min in case x is padded

            return i_x < x_start || i_x >= x_end;
        }
    }

    // if current tile is at the edge, means need per-pixel mask check.
    // otherwise no need to check per-pixel
    // Attention! assume the idex passed in this function is with in range of GetTileRangeAlongX()
    // can be used as a fast-path to decide if do per-pixel check or not
    template <index_t TileHeight, index_t TileWidth>
    CK_TILE_HOST_DEVICE constexpr auto
    IsEdgeTile(index_t i_y, index_t i_x, number<TileHeight>, number<TileWidth>) const
    {
        if constexpr(!IsMasking)
        {
            // the only case that need do following compare is under kPadSeqLenK
            // ... for non-masking kernel.
            // return (i_x < x_total) && ((i_x + TileWidth) > x_total);

            // TODO: no need to check begin
            return (i_x + TileWidth) > x_total;
        }
        else
        {
            // check top-right corner > x or left-borrom corner < x
            index_t i_x_end = i_x + TileWidth;
            index_t i_y_end = i_y + TileHeight;
            // index_t x_end    = min(i_y + x, x_total);

            bool top_right_edge   = i_x_end > min(i_y + x, x_total); // consider right pad
            bool bottom_left_edge = i_y_end > (i_x + y);
            // bool is_partial_out_of_bound = i_x_end > x_end; // only consider right-pad for now

            return top_right_edge || bottom_left_edge;
        }
    }

    private:
    index_t y, x;
    index_t y_total, x_total;
};

// TODO: prefer use this function in host code
// can convert from the FA style left/right to our generic coordinate
// if left_size < 0 && right_size = 0, it is normal causal mask
// local is left_size >=0 or right_size >=0
CK_TILE_HOST_DEVICE constexpr auto
make_generic_attention_mask_coordinates_from_lr_window(index_t left_size,
                                                       index_t right_size,
                                                       index_t y_total,
                                                       index_t x_total,
                                                       bool is_top_left = true)
{
    // TODO: below should all use sgpr arithmetic
    index_t left_size_tmp  = is_top_left ? y_total - 1 : x_total - 1;
    index_t right_size_tmp = is_top_left ? x_total - 1 : y_total - 1;

    left_size  = left_size < 0 ? left_size_tmp : left_size;
    right_size = right_size < 0 ? right_size_tmp : right_size;

    index_t x_tmp = is_top_left ? 0 : x_total - y_total;
    index_t y_tmp = is_top_left ? 0 : y_total - x_total;

    index_t x = 1 + right_size + x_tmp;
    index_t y = 1 + left_size + y_tmp;

    return ck_tile::make_tuple(y, x, y_total, x_total);
}

template <typename MaskType>
CK_TILE_HOST_DEVICE constexpr auto
make_generic_attention_mask_from_lr_window(index_t left_size,
                                           index_t right_size,
                                           index_t y_total,
                                           index_t x_total,
                                           bool is_top_left = true)
{
    auto r = make_generic_attention_mask_coordinates_from_lr_window(
        left_size, right_size, y_total, x_total, is_top_left);
    return MaskType{r.at(ck_tile::number<0>{}), r.at(ck_tile::number<1>{}), y_total, x_total};
}
} // namespace ck_tile
