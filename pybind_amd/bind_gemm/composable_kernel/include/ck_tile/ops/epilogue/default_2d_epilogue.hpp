// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"

namespace ck_tile {

// this epilogue just store out a M*N matrix, row major

template <typename AccDataType_,
          typename ODataType_,
          bool kPadM_,
          bool kPadN_,
          bool UseRawStore_ = true>
struct Default2DEpilogueProblem
{
    using AccDataType                 = remove_cvref_t<AccDataType_>;
    using ODataType                   = remove_cvref_t<ODataType_>;
    static constexpr bool kPadM       = kPadM_;
    static constexpr bool kPadN       = kPadN_;
    static constexpr bool UseRawStore = UseRawStore_;
};

template <typename AccDataType_,
          typename ODataType_,
          typename CLayout_,
          bool kPadM_,
          bool kPadN_,
          index_t kMPerXdl_,
          index_t kNPerXdl_,
          index_t kKPerXdl_,
          bool isCTransposed_,
          bool UseRawStore_ = true>
struct DefaultGemm2DEpilogueProblem
    : public Default2DEpilogueProblem<AccDataType_, ODataType_, kPadM_, kPadN_, UseRawStore_>
{
    using CLayout                          = remove_cvref_t<CLayout_>;
    static constexpr index_t kMPerXdl      = kMPerXdl_;
    static constexpr index_t kNPerXdl      = kNPerXdl_;
    static constexpr index_t kKPerXdl      = kKPerXdl_;
    static constexpr index_t isCTransposed = isCTransposed_;
};

template <typename Problem_, typename Policy_ = void>
struct Default2DEpilogue
{
    using Problem                     = remove_cvref_t<Problem_>;
    using AccDataType                 = remove_cvref_t<typename Problem::AccDataType>;
    using ODataType                   = remove_cvref_t<typename Problem::ODataType>;
    static constexpr bool kPadM       = Problem::kPadM;
    static constexpr bool kPadN       = Problem::kPadN;
    static constexpr bool UseRawStore = Problem::UseRawStore;

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize() { return 0; }

    // TODO: this function assume store out vector size is the same as OAccTile last dimension size
    //       how do we fix this ?
    template <typename ODramWindowTmp,
              typename OAccTile,
              memory_operation_enum out_memory_data_op = memory_operation_enum::set>
    CK_TILE_DEVICE auto
    operator()(ODramWindowTmp& o_dram_window_tmp, const OAccTile& o_acc_tile, void* = nullptr)
    {

        // TODO: this is ugly
        if constexpr(UseRawStore && (kPadM || kPadN))
        {
            if constexpr(out_memory_data_op == memory_operation_enum::set)
            {
                store_tile_raw(o_dram_window_tmp, cast_tile<ODataType>(o_acc_tile));
            }
            else
            {
                update_tile_raw(o_dram_window_tmp, cast_tile<ODataType>(o_acc_tile));
            }
            buffer_store_fence();
        }
        else
        {
            if constexpr(out_memory_data_op == memory_operation_enum::set)
            {
                store_tile(o_dram_window_tmp, cast_tile<ODataType>(o_acc_tile));
            }
            else
            {
                update_tile(o_dram_window_tmp, cast_tile<ODataType>(o_acc_tile));
            }
        }
    }
};

template <typename Problem_, typename Policy_ = void>
struct DefaultGemm2DEpilogue : public Default2DEpilogue<Problem_, Policy_>
{
    using Problem                          = remove_cvref_t<Problem_>;
    using AccDataType                      = remove_cvref_t<typename Problem::AccDataType>;
    using ODataType                        = remove_cvref_t<typename Problem::ODataType>;
    using CLayout                          = remove_cvref_t<typename Problem::CLayout>;
    static constexpr index_t kMPerXdl      = Problem::kMPerXdl;
    static constexpr index_t kNPerXdl      = Problem::kNPerXdl;
    static constexpr index_t kKPerXdl      = Problem::kKPerXdl;
    static constexpr index_t isCTransposed = Problem::isCTransposed;

    using WG = WarpGemmMfmaDispatcher<ODataType,
                                      ODataType,
                                      AccDataType,
                                      kMPerXdl,
                                      kNPerXdl,
                                      kKPerXdl,
                                      isCTransposed>;

    using CWarpDstr = typename WG::CWarpDstr;

    CK_TILE_HOST_DEVICE static constexpr auto GetVectorSizeC()
    {
        // N is contiguous dimension
        if constexpr(std::is_same_v<CLayout, tensor_layout::gemm::RowMajor>)
        {
            if constexpr(isCTransposed)
            {
                // In this case each thread has multiple consecutive elements in
                // N dimension, however consecutive threads' elements have stride.
                constexpr index_t NDimY = CWarpDstr::NDimY;
                constexpr auto c_warp_y_lengths =
                    CWarpDstr{}.get_ys_to_d_descriptor().get_lengths();
                static_assert(WG::WarpGemmAttribute::Impl::kCM1PerLane ==
                              c_warp_y_lengths.get(number<NDimY - 1>{}));
                return c_warp_y_lengths.get(number<NDimY - 1>{});
            }
            else
            {
                // In this case each thread has just a single item in Ndim
                return WG::WarpGemmAttribute::Impl::kCNLane / WG::kN;
            }
        }
        // M is contiguous dimension
        else if constexpr(std::is_same_v<CLayout, tensor_layout::gemm::ColumnMajor>)
        {
            if constexpr(isCTransposed)
            {
                // In this case each thread has just a single item in Mdim
                return WG::WarpGemmAttribute::Impl::kCNLane / WG::kN;
            }
            else
            {
                // In this case each thread has multiple consecutive elements in
                // M dimension, however consecutive threads' elements have stride.
                constexpr index_t NDimY = CWarpDstr::NDimY;
                constexpr auto c_warp_y_lengths =
                    CWarpDstr{}.get_ys_to_d_descriptor().get_lengths();
                static_assert(WG::WarpGemmAttribute::Impl::kCM1PerLane ==
                              c_warp_y_lengths.get(number<NDimY - 1>{}));
                return c_warp_y_lengths.get(number<NDimY - 1>{});
            }
        }
        else
        {
            static_assert(false, "Unsupported CLayout!");
        }
    }
};

} // namespace ck_tile
