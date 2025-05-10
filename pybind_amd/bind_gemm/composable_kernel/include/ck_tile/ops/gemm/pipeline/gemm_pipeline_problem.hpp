// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_scheduler.hpp"

namespace ck_tile {

template <typename ADataType_,
          typename BDataType_,
          typename CDataType_,
          typename BlockGemmShape_,
          typename Traits_>
struct GemmPipelineProblemBase
{
    using Traits = remove_cvref_t<Traits_>;

    using ADataType = remove_cvref_t<ADataType_>;
    using BDataType = remove_cvref_t<BDataType_>;
    using CDataType = remove_cvref_t<CDataType_>;

    using BlockGemmShape = remove_cvref_t<BlockGemmShape_>;

    using ALayout = remove_cvref_t<typename Traits::ALayout>;
    using BLayout = remove_cvref_t<typename Traits::BLayout>;
    using CLayout = remove_cvref_t<typename Traits::CLayout>;

    static constexpr bool TransposeC = Traits::TransposeC;

    static constexpr index_t kBlockSize = BlockGemmShape::NumWarps * get_warp_size();

    static constexpr bool kPadM = Traits::kPadM;
    static constexpr bool kPadN = Traits::kPadN;
    static constexpr bool kPadK = Traits::kPadK;

    static constexpr auto Scheduler = GemmPipelineScheduler::Default;

    static constexpr index_t VectorLoadSize = Traits::_VectorSize;
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentA()
    {
        if constexpr(std::is_same_v<ALayout, ck_tile::tensor_layout::gemm::ColumnMajor>)
        {
            constexpr index_t pixels_per_thread =
                BlockGemmShape::kM * BlockGemmShape::kK / kBlockSize;
            return pixels_per_thread < VectorLoadSize / sizeof(ADataType)
                       ? pixels_per_thread
                       : VectorLoadSize / sizeof(ADataType);
        }
        else
        {
            return VectorLoadSize / sizeof(ADataType);
        }
    }

    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentB()
    {
        if constexpr(std::is_same_v<BLayout, ck_tile::tensor_layout::gemm::RowMajor>)
        {
            constexpr index_t pixels_per_thread =
                BlockGemmShape::kN * BlockGemmShape::kK / kBlockSize;
            return pixels_per_thread < VectorLoadSize / sizeof(BDataType)
                       ? pixels_per_thread
                       : VectorLoadSize / sizeof(BDataType);
        }
        else
        {
            return VectorLoadSize / sizeof(BDataType);
        }
    }

    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentC()
    {
        if constexpr(std::is_same_v<CLayout, ck_tile::tensor_layout::gemm::ColumnMajor>)
        {
            constexpr index_t N1 = kBlockSize / get_warp_size();
            constexpr index_t N2 = std::min(BlockGemmShape::kN / N1, get_warp_size());
            constexpr index_t M0 = get_warp_size() / N2;
            constexpr index_t M1 = BlockGemmShape::kM / M0;

            return std::min(M1, static_cast<index_t>(VectorLoadSize / sizeof(CDataType)));
        }
        else
        {
            constexpr index_t M1 = kBlockSize / get_warp_size();
            constexpr index_t M2 = std::min(BlockGemmShape::kM / M1, get_warp_size());
            constexpr index_t N0 = get_warp_size() / M2;
            constexpr index_t N1 = BlockGemmShape::kN / N0;

            return std::min(N1, static_cast<index_t>(VectorLoadSize / sizeof(CDataType)));
        }
    }

    static constexpr index_t VectorSizeA = []() {
        if constexpr(std::is_same_v<ALayout, tensor_layout::gemm::RowMajor>)
        {
            return kPadK ? 1 : GetAlignmentA();
        }
        else
        {
            return kPadM ? 1 : GetAlignmentA();
        }
    }();

    static constexpr index_t VectorSizeB = []() {
        if constexpr(std::is_same_v<BLayout, tensor_layout::gemm::ColumnMajor>)
        {
            return kPadN ? 1 : GetAlignmentB();
        }
        else
        {
            return kPadK ? 1 : GetAlignmentB();
        }
    }();
    static constexpr index_t VectorSizeC = []() {
        if constexpr(std::is_same_v<CLayout, tensor_layout::gemm::RowMajor>)
        {
            return kPadN ? 1 : GetAlignmentC();
        }
        else
        {
            return kPadM ? 1 : GetAlignmentC();
        }
    }();
};

// Alias for GemmPipelineProblem
template <typename ADataType_,
          typename BDataType_,
          typename CDataType_,
          typename BlockGemmShape_,
          typename Traits_>
using GemmPipelineProblem =
    GemmPipelineProblemBase<ADataType_, BDataType_, CDataType_, BlockGemmShape_, Traits_>;

template <typename ADataType_,
          typename BDataType_,
          typename CDataType_,
          typename BlockGemmShape_,
          typename Traits_,
          GemmPipelineScheduler Scheduler_ = GemmPipelineScheduler::Intrawave,
          bool HasHotLoop_                 = true,
          TailNumber TailNum_              = TailNumber::Full>
struct UniversalGemmPipelineProblem
{
    using Traits = remove_cvref_t<Traits_>;

    using ADataType = remove_cvref_t<ADataType_>;
    using BDataType = remove_cvref_t<BDataType_>;
    using CDataType = remove_cvref_t<CDataType_>;

    using BlockGemmShape = remove_cvref_t<BlockGemmShape_>;

    using ALayout = remove_cvref_t<typename Traits::ALayout>;
    using BLayout = remove_cvref_t<typename Traits::BLayout>;
    using CLayout = remove_cvref_t<typename Traits::CLayout>;

    static constexpr index_t kBlockSize = BlockGemmShape::NumWarps * get_warp_size();

    static constexpr bool kPadM = Traits::kPadM;
    static constexpr bool kPadN = Traits::kPadN;
    static constexpr bool kPadK = Traits::kPadK;

    static constexpr auto Scheduler  = Scheduler_;
    static constexpr auto HasHotLoop = HasHotLoop_;
    static constexpr auto TailNum    = TailNum_;

    static constexpr bool TransposeC = Traits::TransposeC;
};

} // namespace ck_tile
