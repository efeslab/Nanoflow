// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

template <typename QDataType_,
          typename KDataType_,
          typename VDataType_,
          typename SaccDataType_,
          typename SMPLComputeDataType_,
          typename BiasDataType_,
          typename RandValOutputDataType_,
          typename LSEDataType_,
          typename PDataType_,
          typename OaccDataType_,
          typename ODataType_,
          typename BlockFmhaShape_,
          bool kIsGroupMode_,
          typename FmhaMask_,
          typename Traits_>
struct BlockFmhaPipelineProblem
{
    using QDataType             = remove_cvref_t<QDataType_>;
    using KDataType             = remove_cvref_t<KDataType_>;
    using VDataType             = remove_cvref_t<VDataType_>;
    using SaccDataType          = remove_cvref_t<SaccDataType_>;
    using SMPLComputeDataType   = remove_cvref_t<SMPLComputeDataType_>;
    using BiasDataType          = remove_cvref_t<BiasDataType_>;
    using RandValOutputDataType = remove_cvref_t<RandValOutputDataType_>;
    using LSEDataType           = remove_cvref_t<LSEDataType_>;
    using PDataType             = remove_cvref_t<PDataType_>;
    using OaccDataType          = remove_cvref_t<OaccDataType_>;
    using ODataType             = remove_cvref_t<ODataType_>;
    using BlockFmhaShape        = remove_cvref_t<BlockFmhaShape_>;
    using FmhaMask              = remove_cvref_t<FmhaMask_>;
    using Traits                = remove_cvref_t<Traits_>;

    static constexpr index_t kNumGemm0Warps = BlockFmhaShape::NumGemm0Warps;
    static constexpr index_t kNumGemm1Warps = BlockFmhaShape::NumGemm1Warps;
    static constexpr index_t kBlockSize     = BlockFmhaShape::NumWarps * get_warp_size();

    static constexpr bool kIsGroupMode = kIsGroupMode_;

    // attributes from traits
    static constexpr bool kPadSeqLenQ       = Traits::kPadSeqLenQ;
    static constexpr bool kPadSeqLenK       = Traits::kPadSeqLenK;
    static constexpr bool kPadHeadDimQ      = Traits::kPadHeadDimQ;
    static constexpr bool kPadHeadDimV      = Traits::kPadHeadDimV;
    static constexpr auto BiasEnum          = Traits::BiasEnum;
    static constexpr bool kStoreLSE         = Traits::kStoreLSE;
    static constexpr bool kHasDropout       = Traits::kHasDropout;
    static constexpr bool kDoFp8StaticQuant = Traits::kDoFp8StaticQuant;
    static constexpr index_t kBlockPerCu    = Traits::kBlockPerCu;
};

template <typename QDataType_,
          typename KDataType_,
          typename VDataType_,
          typename SaccDataType_,
          typename SMPLComputeDataType_,
          typename BiasDataType_,
          typename LSEDataType_,
          typename PDataType_,
          typename OaccDataType_,
          typename ODataType_,
          typename BlockFmhaShape_,
          bool kIsGroupMode_,
          typename FmhaMask_,
          typename Traits_>
struct BlockFmhaFwdSplitKVPipelineProblem
{
    using QDataType           = remove_cvref_t<QDataType_>;
    using KDataType           = remove_cvref_t<KDataType_>;
    using VDataType           = remove_cvref_t<VDataType_>;
    using SaccDataType        = remove_cvref_t<SaccDataType_>;
    using SMPLComputeDataType = remove_cvref_t<SMPLComputeDataType_>;
    using BiasDataType        = remove_cvref_t<BiasDataType_>;
    using LSEDataType         = remove_cvref_t<LSEDataType_>;
    using PDataType           = remove_cvref_t<PDataType_>;
    using OaccDataType        = remove_cvref_t<OaccDataType_>;
    using ODataType           = remove_cvref_t<ODataType_>;
    using BlockFmhaShape      = remove_cvref_t<BlockFmhaShape_>;
    using FmhaMask            = remove_cvref_t<FmhaMask_>;
    using Traits              = remove_cvref_t<Traits_>;

    static constexpr index_t kNumGemm0Warps = BlockFmhaShape::NumGemm0Warps;
    static constexpr index_t kNumGemm1Warps = BlockFmhaShape::NumGemm1Warps;
    static constexpr index_t kBlockSize     = BlockFmhaShape::NumWarps * get_warp_size();

    static constexpr bool kIsGroupMode = kIsGroupMode_;

    // attributes from traits
    static constexpr bool kPadSeqLenQ                = Traits::kPadSeqLenQ;
    static constexpr bool kPadSeqLenK                = Traits::kPadSeqLenK;
    static constexpr bool kPadHeadDimQ               = Traits::kPadHeadDimQ;
    static constexpr bool kPadHeadDimV               = Traits::kPadHeadDimV;
    static constexpr auto BiasEnum                   = Traits::BiasEnum;
    static constexpr bool kStoreLSE                  = Traits::kStoreLSE;
    static constexpr bool kDoFp8StaticQuant          = Traits::kDoFp8StaticQuant;
    static constexpr bool kIsPagedKV                 = Traits::kIsPagedKV;
    static constexpr bool kHasUnevenSplits           = kIsGroupMode || Traits::kHasUnevenSplits;
    static constexpr bool kMergeNumHeadGroupsSeqLenQ = Traits::kMergeNumHeadGroupsSeqLenQ;
    static constexpr index_t kBlockPerCu             = Traits::kBlockPerCu;
};

// extract tile size attributes to remove dependency on traits
template <typename OaccDataType_, ck_tile::index_t kN1_>
struct BlockFmhaSplitKVCombinePipelineTileSizes
{
    static constexpr index_t MaxVectorSize = 16 / sizeof(OaccDataType_);

    static constexpr index_t kN1      = kN1_;
    static constexpr index_t NThreads = kN1 / MaxVectorSize;
    static constexpr index_t kM0      = get_warp_size() / NThreads; // MThreadPerWarp
};

template <typename LSEDataType_,
          typename OaccDataType_,
          typename ODataType_,
          index_t HeadDimV_,
          bool kIsGroupMode_,
          ck_tile::index_t kN1_,
          typename Traits_>
struct BlockFmhaSplitKVCombinePipelineProblem
    : BlockFmhaSplitKVCombinePipelineTileSizes<OaccDataType_, kN1_>
{
    using BaseType = BlockFmhaSplitKVCombinePipelineTileSizes<OaccDataType_, kN1_>;

    using LSEDataType  = remove_cvref_t<LSEDataType_>;
    using OaccDataType = remove_cvref_t<OaccDataType_>;
    using ODataType    = remove_cvref_t<ODataType_>;
    using Traits       = remove_cvref_t<Traits_>;

    static_assert(std::is_same_v<LSEDataType, OaccDataType>);

    static constexpr index_t kHeadDimV = HeadDimV_;
    static constexpr bool kIsGroupMode = kIsGroupMode_;

    using BaseType::kM0;
    using BaseType::kN1;

    static_assert(kN1 <= kHeadDimV && kHeadDimV % kN1 == 0);

    // attributes from traits
    static constexpr bool kPadSeqLenQ       = Traits::kPadSeqLenQ;
    static constexpr bool kPadHeadDimV      = Traits::kPadHeadDimV;
    static constexpr bool kStoreLSE         = Traits::kStoreLSE;
    static constexpr bool kDoFp8StaticQuant = Traits::kDoFp8StaticQuant;
    static constexpr index_t kBlockPerCu    = Traits::kBlockPerCu;
    static constexpr index_t kMaxSplits     = Traits::kMaxSplits;
    static_assert(8 <= kMaxSplits);

    static constexpr index_t kNumWarps  = 4; // always use 4 warps for each workgroup
    static constexpr index_t kBlockSize = kNumWarps * get_warp_size();

    static_assert(get_warp_size() <= (kM0 * kMaxSplits) &&
                  (kM0 * kMaxSplits) % get_warp_size() == 0);
};

template <typename QDataType_,
          typename KDataType_,
          typename VDataType_,
          index_t kM0_,
          index_t kN0_,
          index_t kK0_,
          index_t kN1_,
          bool kIsVLayoutRowMajor_,
          RotaryEmbeddingEnum RotaryEnum_,
          bool kIsPagedKV_,
          typename Traits_>
struct BlockFmhaFwdAppendKVPipelineProblem
{
    using QDataType = remove_cvref_t<QDataType_>;
    using KDataType = remove_cvref_t<KDataType_>;
    using VDataType = remove_cvref_t<VDataType_>;
    using Traits    = remove_cvref_t<Traits_>;

    static constexpr index_t kBlockSize = 256;

    static constexpr index_t kM0 = kM0_;
    static constexpr index_t kN0 = kN0_;
    static constexpr index_t kK0 = kK0_;
    static constexpr index_t kN1 = kN1_;

    using VLayout = std::conditional_t<kIsVLayoutRowMajor_,
                                       ck_tile::tensor_layout::gemm::RowMajor,
                                       ck_tile::tensor_layout::gemm::ColumnMajor>;

    static constexpr auto RotaryEnum = RotaryEnum_;
    static constexpr bool kIsPagedKV = kIsPagedKV_;

    // attributes from traits
    static constexpr bool kPadSeqLenQ    = Traits::kPadSeqLenQ;
    static constexpr bool kPadSeqLenK    = Traits::kPadSeqLenK;
    static constexpr bool kPadHeadDimQ   = Traits::kPadHeadDimQ;
    static constexpr bool kPadHeadDimV   = Traits::kPadHeadDimV;
    static constexpr index_t kBlockPerCu = Traits::kBlockPerCu;
};

} // namespace ck_tile
