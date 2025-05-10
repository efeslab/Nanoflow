// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_problem.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_problem.hpp"
#include "ck_tile/ops/gemm/pipeline/tile_gemm_shape.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_asmem_bsmem_creg_v1_custom_policy.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_asmem_bsmem_creg_v1.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_areg_bsmem_creg_v1_custom_policy.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_areg_bsmem_creg_v2_custom_policy.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_areg_bsmem_creg_v2.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_areg_bsmem_creg_one_warp_v1.hpp"

// TODO: remove this
#define K_LDS_LOAD_USE_OFFSET_TRANSFORM 0

namespace ck_tile {

template <bool QLoadOnce_>
struct BlockFmhaPipelineQXCustomPolicy;

template <>
struct BlockFmhaPipelineQXCustomPolicy</* QLoadOnce = */ true>
{
    static constexpr bool QLoadOnce = true;

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSizeQ()
    {
        return 0;
    }

    // TODO: GetAlignment*() currently didn't consider if need padding or not
    //       so in pipeline still need check padding requirement
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentQ()
    {
        constexpr index_t MaxVectorSize = 16 / sizeof(typename Problem::QDataType);

        using BlockGemm       = remove_cvref_t<decltype(GetQKBlockGemm<Problem>())>;
        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG              = remove_cvref_t<decltype(config.template at<0>())>;

        return min(MaxVectorSize, WG::kK / WG::WarpGemmAttribute::Impl::kABKLane);
    }

    template <typename Problem, typename BlockGemm>
    CK_TILE_HOST_DEVICE static constexpr auto MakeQDramTileDistribution()
    {
        return BlockGemm::template MakeABlockTileDistribution<
            Problem::BlockFmhaShape::kM0,
            Problem::BlockFmhaShape::kSubQKHeaddim>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetQKBlockGemm()
    {
        using GemmProblem =
            BlockGemmProblem<typename Problem::QDataType,
                             typename Problem::KDataType,
                             typename Problem::SaccDataType,
                             Problem::kNumGemm0Warps * get_warp_size(),
                             TileGemmShape<sequence<Problem::BlockFmhaShape::kM0,
                                                    Problem::BlockFmhaShape::kN0,
                                                    Problem::BlockFmhaShape::kK0>,
                                           typename Problem::BlockFmhaShape::Gemm0BlockWarps,
                                           typename Problem::BlockFmhaShape::Gemm0WarpTile>>;

        constexpr auto warp_gemm = []() {
            constexpr index_t WarpGemmM = Problem::BlockFmhaShape::Gemm0WarpTile::at(number<0>{});
            static_assert(WarpGemmM == 4 || WarpGemmM == 16 || WarpGemmM == 32);

            if constexpr(std::is_same_v<typename Problem::QDataType, half_t> &&
                         std::is_same_v<typename Problem::KDataType, half_t> &&
                         std::is_same_v<typename Problem::SaccDataType, float>)
            {
                if constexpr(WarpGemmM == 32)
                    return WarpGemmMfmaF16F16F32M32N32K16SwizzleBTransposedCDistribution{};
                else if constexpr(WarpGemmM == 16)
                    return WarpGemmMfmaF16F16F32M16N16K16TransposedCDistribution{};
                else // WarpGemmM == 4
                    return WarpGemmMfmaF16F16F32M4N64K16{};
            }
            else if constexpr(std::is_same_v<typename Problem::QDataType, bf16_t> &&
                              std::is_same_v<typename Problem::KDataType, bf16_t> &&
                              std::is_same_v<typename Problem::SaccDataType, float>)
            {
                if constexpr(WarpGemmM == 32)
                    return WarpGemmMfmaBf16Bf16F32M32N32K16SwizzleBTransposedCDistribution{};
                else if constexpr(WarpGemmM == 16)
                    return WarpGemmMfmaBf16Bf16F32M16N16K16TransposedCDistribution{};
                else // WarpGemmM == 4
                    return WarpGemmMfmaBf16Bf16F32M4N64K16{};
            }
            else if constexpr(std::is_same_v<typename Problem::QDataType, fp8_t> &&
                              std::is_same_v<typename Problem::KDataType, fp8_t> &&
                              std::is_same_v<typename Problem::SaccDataType, float>)
            {
                static_assert(WarpGemmM == 32);

                // TODO: hard coded here. Otherwise, it may incorrect result
                constexpr index_t swizzle_factor = 4;
                return WarpGemmMfmaFp8Fp8F32M32N32K16SwizzleBTransposedCDistribution<
                    swizzle_factor>{};
            } // TODO - bf8_t
        }();

        using BlockGemmPolicy =
            BlockGemmARegBSmemCRegV2CustomPolicy<typename Problem::QDataType,
                                                 typename Problem::KDataType,
                                                 typename Problem::SaccDataType,
                                                 typename Problem::BlockFmhaShape::Gemm0BlockWarps,
                                                 decltype(warp_gemm)>;

        if constexpr(1 < Problem::kNumGemm0Warps)
            return BlockGemmARegBSmemCRegV2<GemmProblem, BlockGemmPolicy>{};
        else
            return BlockGemmARegBSmemCRegOneWarpV1<GemmProblem, BlockGemmPolicy>{};
    }
};

template <>
struct BlockFmhaPipelineQXCustomPolicy</* QLoadOnce = */ false>
{
    static constexpr bool QLoadOnce = false;

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSizeQ()
    {
        constexpr index_t lds_alignment = 16; // optional
        constexpr index_t q_smem_size =
            ck_tile::integer_divide_ceil(
                sizeof(typename Problem::QDataType) *
                    MakeQLdsBlockDescriptor<Problem>().get_element_space_size(),
                lds_alignment) *
            lds_alignment;
        return q_smem_size;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentQ()
    {
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK0;

        constexpr index_t MaxVectorSize = 16 / sizeof(typename Problem::QDataType);

        // this should align with MakeQDramTileDistribution()
        constexpr index_t ElemPerThread = (kMPerBlock * kKPerBlock) / kBlockSize;
        static_assert(0 < ElemPerThread);
        return min(ElemPerThread, MaxVectorSize);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeQDramTileDistribution()
    {
        using QDataType = remove_cvref_t<typename Problem::QDataType>;

        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK0;

        constexpr index_t MaxVectorSize = 16 / sizeof(QDataType);

        constexpr index_t ElemPerThread = (kMPerBlock * kKPerBlock) / kBlockSize;
        static_assert(0 < ElemPerThread);
        constexpr index_t kMaxVecLoad = min(ElemPerThread, MaxVectorSize);

        constexpr index_t KPerThread     = kMaxVecLoad;
        constexpr index_t KThreads       = kKPerBlock / KPerThread;
        constexpr index_t MThreadPerWarp = get_warp_size() / KThreads;
        constexpr index_t NumWarps       = kBlockSize / get_warp_size();
        constexpr index_t MPerThread     = kMPerBlock / (MThreadPerWarp * NumWarps);

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<1>,
                                       tuple<sequence<MPerThread, NumWarps, MThreadPerWarp>,
                                             sequence<KThreads, KPerThread>>,
                                       tuple<sequence<1>, sequence<1, 2>>,
                                       tuple<sequence<1>, sequence<2, 0>>,
                                       sequence<1, 2>,
                                       sequence<0, 1>>{});
    }

    // 3d + padding
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeQLdsBlockDescriptor()
    {
        using QDataType = remove_cvref_t<typename Problem::QDataType>;

        constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK0;
        constexpr index_t kKPack     = 16 / sizeof(QDataType);

        constexpr auto q_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(number<kKPerBlock / kKPack>{}, number<kMPerBlock>{}, number<kKPack>{}),
            make_tuple(number<(kMPerBlock + 1) * kKPack>{}, number<kKPack>{}, number<1>{}),
            number<8>{},
            number<1>{});

        constexpr auto q_lds_block_desc = transform_tensor_descriptor(
            q_lds_block_desc_0,
            make_tuple(make_pass_through_transform(kMPerBlock),
                       make_merge_transform(make_tuple(kKPerBlock / kKPack, kKPack))),
            make_tuple(sequence<1>{}, sequence<0, 2>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        return q_lds_block_desc;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetQKBlockGemm()
    {
        using GemmProblem =
            BlockGemmProblem<typename Problem::QDataType,
                             typename Problem::KDataType,
                             typename Problem::SaccDataType,
                             Problem::kBlockSize,
                             TileGemmShape<sequence<Problem::BlockFmhaShape::kM0,
                                                    Problem::BlockFmhaShape::kN0,
                                                    Problem::BlockFmhaShape::kK0>,
                                           typename Problem::BlockFmhaShape::Gemm0BlockWarps,
                                           typename Problem::BlockFmhaShape::Gemm0WarpTile>>;

        constexpr index_t WarpGemmM = Problem::BlockFmhaShape::Gemm0WarpTile::at(number<0>{});
        static_assert(WarpGemmM == 4 || WarpGemmM == 16 || WarpGemmM == 32);

        constexpr auto warp_gemm = []() {
            if constexpr(std::is_same_v<typename Problem::QDataType, half_t> &&
                         std::is_same_v<typename Problem::KDataType, half_t> &&
                         std::is_same_v<typename Problem::SaccDataType, float>)
            {
                if constexpr(WarpGemmM == 32)
                    return WarpGemmMfmaF16F16F32M32N32K16SwizzleBTransposedCDistribution{};
                else if constexpr(WarpGemmM == 16)
                    return WarpGemmMfmaF16F16F32M16N16K16TransposedCDistribution{};
                else // WarpGemmM == 4
                    return WarpGemmMfmaF16F16F32M4N64K16{};
            }
            else if constexpr(std::is_same_v<typename Problem::QDataType, bf16_t> &&
                              std::is_same_v<typename Problem::KDataType, bf16_t> &&
                              std::is_same_v<typename Problem::SaccDataType, float>)
            {
                if constexpr(WarpGemmM == 32)
                    return WarpGemmMfmaBf16Bf16F32M32N32K16SwizzleBTransposedCDistribution{};
                else if constexpr(WarpGemmM == 16)
                    return WarpGemmMfmaBf16Bf16F32M16N16K16TransposedCDistribution{};
                else // WarpGemmM == 4
                    return WarpGemmMfmaBf16Bf16F32M4N64K16{};
            }
            else if constexpr(std::is_same_v<typename Problem::QDataType, fp8_t> &&
                              std::is_same_v<typename Problem::KDataType, fp8_t> &&
                              std::is_same_v<typename Problem::SaccDataType, float>)
            {
                // TODO: hard coded here. Otherwise, it may incorrect result
                constexpr index_t swizzle_factor = 4;
                return WarpGemmMfmaFp8Fp8F32M32N32K16SwizzleBTransposedCDistribution<
                    swizzle_factor>{};
            } // TODO - bf8_t
        }();

        using BlockGemmPolicy =
            BlockGemmASmemBSmemCRegV1CustomPolicy<typename Problem::QDataType,
                                                  typename Problem::KDataType,
                                                  typename Problem::SaccDataType,
                                                  typename Problem::BlockFmhaShape::Gemm0BlockWarps,
                                                  decltype(warp_gemm)>;

        return BlockGemmASmemBSmemCRegV1<GemmProblem, BlockGemmPolicy>{};
    }
};

// This pipeline is qkv all located in LDS
template <bool QLoadOnce_,
          bool AsyncCopyK_,
          bool AsyncCopyV_,
          index_t NumPrefetchK_,
          index_t NumPrefetchV_>
struct BlockFmhaPipelineQXKSVSCustomPolicy : BlockFmhaPipelineQXCustomPolicy<QLoadOnce_>
{
    static constexpr bool AsyncCopyK = AsyncCopyK_;
    static constexpr bool AsyncCopyV = AsyncCopyV_; // TODO: this not supported yet

    static constexpr index_t NumPrefetchK = NumPrefetchK_;
    static constexpr index_t NumPrefetchV = NumPrefetchK_;

    using QXPolicy = BlockFmhaPipelineQXCustomPolicy<QLoadOnce_>;

    template <index_t k_prefetches_, index_t v_prefetches_, index_t k_loops_, index_t v_loops_>
    struct LdsBufferSequence
    {
        static constexpr auto Make()
        {
            return transform_sequences(
                [&](auto i) {
                    if(i < k_loops_)
                        return i % k_prefetches_;
                    return (i - k_loops_) % v_prefetches_;
                },
                typename arithmetic_sequence_gen<0, k_loops_ + v_loops_, 1>::type{});
        };

        using type = remove_cvref_t<decltype(Make())>;
    };
    // clang-format off
    template<> struct
    LdsBufferSequence<3, 3, 4, 4> { using type = sequence<1, 2, 0, 1,   0, 1, 2, 0>; };

    template<> struct
    LdsBufferSequence<3, 3, 4, 2> { using type = sequence<1, 2, 0, 1,   2, 0>; };

    template<> struct
    LdsBufferSequence<3, 3, 2, 4> { using type = sequence<1, 2,         0, 1, 2, 0>; };

    template<> struct
    LdsBufferSequence<3, 3, 3, 3> { using type = sequence<1, 2, 0,      1, 2, 0>; };

    template<> struct
    LdsBufferSequence<3, 3, 3, 4> { using type = sequence<1, 2, 0,      0, 1, 2, 0>; };

    template<> struct
    LdsBufferSequence<3, 3, 2, 2> { using type = sequence<1, 2,         1, 0>;};
    // clang-format on

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetLdsBufferSequence()
    {
        using BlockFmhaShape = remove_cvref_t<typename Problem::BlockFmhaShape>;

        constexpr index_t kN0        = BlockFmhaShape::kN0;
        constexpr index_t kK0        = BlockFmhaShape::kK0;
        constexpr index_t kK1        = BlockFmhaShape::kK1;
        constexpr index_t kQKHeaddim = BlockFmhaShape::kQKHeaddim;

        constexpr index_t k0_loops = kQKHeaddim / kK0;
        constexpr index_t k1_loops = kN0 / kK1;

        return typename LdsBufferSequence<NumPrefetchK, NumPrefetchV, k0_loops, k1_loops>::type{};
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetSmemKPackK()
    {
        // TODO: this is for 3d layout
        using KDataType = remove_cvref_t<typename Problem::KDataType>;
        return 16 / sizeof(KDataType);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentK()
    {
        using KDataType = remove_cvref_t<typename Problem::KDataType>;
        if constexpr(AsyncCopyK)
        {
            return 4 / sizeof(KDataType);
        }
        else
        {
            return 16 / sizeof(KDataType);
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetSmemKPackV()
    {
        // TODO: this is for 3d layout
        using VDataType = remove_cvref_t<typename Problem::VDataType>;
        return 16 / sizeof(VDataType);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentV()
    {
        using VLayout   = remove_cvref_t<typename Problem::BlockFmhaShape::VLayout>;
        using VDataType = remove_cvref_t<typename Problem::VDataType>;
        if constexpr(std::is_same_v<VLayout, ck_tile::tensor_layout::gemm::RowMajor>)
        {
            constexpr index_t kBlockSize   = Problem::kBlockSize;
            constexpr index_t kNPerBlock   = Problem::BlockFmhaShape::kN1;
            constexpr index_t kKPerBlock   = Problem::BlockFmhaShape::kK1;
            constexpr index_t total_pixels = kNPerBlock * kKPerBlock / kBlockSize;
            constexpr index_t kMaxVecLoad =
                min(total_pixels, static_cast<index_t>(16 / sizeof(VDataType)));
            constexpr index_t kMinVecLoad = 4 / sizeof(VDataType);

            constexpr index_t kVecLoad = ((total_pixels / kMaxVecLoad) >= kMinVecLoad)
                                             ? kMaxVecLoad
                                             : (total_pixels / kMinVecLoad);

            return kVecLoad;
        }
        else
        {
            return 16 / sizeof(VDataType);
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentBias()
    {
        using BlockGemm = remove_cvref_t<decltype(QXPolicy::template GetQKBlockGemm<Problem>())>;
        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG              = remove_cvref_t<decltype(config.template at<0>())>;

        return WG::WarpGemmAttribute::Impl::kCM1PerLane;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentO()
    {
        using BlockGemm       = remove_cvref_t<decltype(GetKVBlockGemm<Problem>())>;
        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG              = remove_cvref_t<decltype(config.template at<0>())>;

        return WG::WarpGemmAttribute::Impl::kCM1PerLane;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetSingleSmemElementSpaceSize()
    {
        // this function assume K/V can share smem
        constexpr index_t SingleKSize = [&]() {
            if constexpr(!AsyncCopyK)
            {
                return MakeKLdsBlockDescriptor<Problem>().get_element_space_size();
            }
            else
            {
                constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0;
                constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK1;
                constexpr index_t NumWarps   = Problem::BlockFmhaShape::NumWarps;
                constexpr index_t warpSize   = ck_tile::get_warp_size();

                constexpr index_t KPack   = GetSmemKPackK<Problem>(); // this is for lds
                constexpr index_t KVector = GetAlignmentK<Problem>(); // this is for global load
                constexpr index_t kPad    = KPack;

                static_assert(warpSize * KVector >= kKPerBlock &&
                              warpSize * KVector % kKPerBlock == 0);
                constexpr index_t LanesPerK  = kKPerBlock / KVector;
                constexpr index_t LaneGroups = warpSize / LanesPerK;
                constexpr index_t NumIssues  = kNPerBlock / (LaneGroups * NumWarps);

                return NumIssues * NumWarps * (warpSize * KVector + kPad);
            }
        }();

        constexpr index_t SingleVSize = [&]() {
            using VDataType                = remove_cvref_t<typename Problem::VDataType>;
            constexpr index_t Banks        = 32; // TODO: need change based on arch
            constexpr index_t PixelsPerRow = Banks * 4 / sizeof(VDataType);
            constexpr index_t kKPack       = GetSmemKPackK<Problem>();
            static_assert(PixelsPerRow % kKPack == 0);
            constexpr index_t NPerRow    = PixelsPerRow / kKPack;
            constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN1;
            constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK1;
            static_assert(kNPerBlock % NPerRow == 0);
            static_assert(kKPerBlock % kKPack == 0);

            return (kKPerBlock / kKPack) * (kNPerBlock / NPerRow) * (PixelsPerRow + kKPack);
        }();

        return max(SingleKSize, SingleVSize);
    }

    // TODO: this is used for non async copy desc. unify in the future
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKLdsBlockDescriptor()
    {
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK0;
        constexpr index_t kKPack     = GetSmemKPackK<Problem>();

        constexpr auto k_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(number<kKPerBlock / kKPack>{}, number<kNPerBlock>{}, number<kKPack>{}),
            make_tuple(number<(kNPerBlock + 1) * kKPack>{}, number<kKPack>{}, number<1>{}),
            number<8>{},
            number<1>{});

        constexpr auto k_lds_block_desc = transform_tensor_descriptor(
            k_lds_block_desc_0,
            make_tuple(
                make_pass_through_transform(number<kNPerBlock>{}),
                make_merge_transform(make_tuple(number<kKPerBlock / kKPack>{}, number<kKPack>{}))),
            make_tuple(sequence<1>{}, sequence<0, 2>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        return k_lds_block_desc;
    }

    template <typename Problem, index_t IBuf = 0>
    CK_TILE_HOST_DEVICE static constexpr auto
        MakeKLdsStoreBlockDescriptor(number<IBuf> = number<0>{})
    {
        // K is always k-major, we use async-copy to load into LDS
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK1;
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t NumWarps   = Problem::BlockFmhaShape::NumWarps;
        constexpr index_t warpSize   = ck_tile::get_warp_size();

        constexpr index_t KPack   = GetSmemKPackK<Problem>(); // this is for lds
        constexpr index_t KVector = GetAlignmentK<Problem>(); // this is for global load
        constexpr index_t kPad =
            KPack; // for async-copy, this pad is between warps. Optimize this for lds_read speed

        static_assert(warpSize * KVector >= kKPerBlock && warpSize * KVector % kKPerBlock == 0);
        constexpr index_t LanesPerK =
            kKPerBlock / KVector; // how many lane (within a wave) to load K
        constexpr index_t LaneGroups =
            warpSize /
            LanesPerK; // how many groups (within a wave), they may load different N, but same K
        constexpr index_t NumIssues = kNPerBlock / (LaneGroups * NumWarps);
        static_assert(NumIssues == kNPerBlock * kKPerBlock / (kBlockSize * KVector));

        constexpr auto k_lds_block_desc_0 = make_naive_tensor_descriptor_with_offset(
            make_tuple(number<NumIssues>{},  // n0
                       number<LaneGroups>{}, // n1
                       number<NumWarps>{},   // n2
                       number<LanesPerK>{},  // k0
                       number<KVector>{}),   // k1
            make_tuple(number<NumWarps*(warpSize * KVector + kPad)>{},
                       number<kKPerBlock>{},
                       number<warpSize * KVector + kPad>{},
                       number<KVector>{},
                       number<1>{}),
            number<IBuf * GetSingleSmemElementSpaceSize<Problem>()>{},
            number<KVector>{},
            number<1>{});

        // TODO this layout is hard coded, and will be used in async copy buffer view load
        // in LDS the real layout is (bufs, N0, N2, N1*K0*K1)
        constexpr auto k_lds_block_desc_issues_warps_lanes = transform_tensor_descriptor(
            k_lds_block_desc_0,
            make_tuple(make_pass_through_transform(number<NumIssues>{}),
                       make_pass_through_transform(number<NumWarps>{}),
                       make_merge_transform(make_tuple(
                           number<LaneGroups>{}, number<LanesPerK>{}, number<KVector>{}))),
            make_tuple(sequence<0>{}, sequence<2>{}, sequence<1, 3, 4>{}),
            make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}));

        return k_lds_block_desc_issues_warps_lanes;
    }

#if K_LDS_LOAD_USE_OFFSET_TRANSFORM
    template <typename Problem, index_t IBuf = 0>
    CK_TILE_HOST_DEVICE static constexpr auto
        MakeKLdsLoadBlockDescriptor(number<IBuf> = number<0>{})
    {
        // K is always k-major, we use async-copy to load into LDS
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK1;
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t NumWarps   = Problem::BlockFmhaShape::NumWarps;
        constexpr index_t warpSize   = ck_tile::get_warp_size();

        constexpr index_t KPack   = GetSmemKPackK<Problem>(); // this is for lds
        constexpr index_t KVector = GetAlignmentK<Problem>(); // this is for global load
        constexpr index_t kPad    = KPack; // for async-copy, this pad is between warps

        static_assert(warpSize * KVector >= kKPerBlock && warpSize * KVector % kKPerBlock == 0);
        constexpr index_t LanesPerK  = kKPerBlock / KVector; // within a wave
        constexpr index_t LaneGroups = warpSize / LanesPerK; // within a wave
        constexpr index_t NumIssues  = kNPerBlock / (LaneGroups * NumWarps);
        static_assert(NumIssues == kNPerBlock * kKPerBlock / (kBlockSize * KVector));

        constexpr auto k_lds_block_desc_0 = make_naive_tensor_descriptor_with_offset(
            make_tuple(number<NumIssues>{},          // n0
                       number<NumWarps>{},           // n2
                       number<LaneGroups>{},         // n1
                       number<kKPerBlock / KPack>{}, // k0
                       number<KPack>{}),             // k1
            make_tuple(number<NumWarps*(warpSize * KVector + kPad)>{},
                       number<warpSize * KVector + kPad>{},
                       number<kKPerBlock>{},
                       number<KPack>{},
                       number<1>{}),
            number<IBuf * GetSingleSmemElementSpaceSize<Problem>()>{},
            number<KPack>{},
            number<1>{});

        constexpr auto k_lds_block_desc = transform_tensor_descriptor(
            k_lds_block_desc_0,
            make_tuple(
                make_merge_transform(
                    make_tuple(number<NumIssues>{}, number<LaneGroups>{}, number<NumWarps>{})),
                make_merge_transform(make_tuple(number<kKPerBlock / KPack>{}, number<KPack>{}))),
            make_tuple(sequence<0, 2, 1>{}, sequence<3, 4>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        return k_lds_block_desc;
    }
#else
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKLdsLoadBlockDescriptor()
    {
        // K is always k-major, we use async-copy to load into LDS
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK1;
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t NumWarps   = Problem::BlockFmhaShape::NumWarps;
        constexpr index_t warpSize   = ck_tile::get_warp_size();

        constexpr index_t KPack   = GetSmemKPackK<Problem>(); // this is for lds
        constexpr index_t KVector = GetAlignmentK<Problem>(); // this is for global load
        constexpr index_t kPad    = KPack; // for async-copy, this pad is between warps

        static_assert(warpSize * KVector >= kKPerBlock && warpSize * KVector % kKPerBlock == 0);
        constexpr index_t LanesPerK  = kKPerBlock / KVector; // within a wave
        constexpr index_t LaneGroups = warpSize / LanesPerK; // within a wave
        constexpr index_t NumIssues  = kNPerBlock / (LaneGroups * NumWarps);
        static_assert(NumIssues == kNPerBlock * kKPerBlock / (kBlockSize * KVector));
        // constexpr index_t SingleKSize = NumIssues * NumWarps * (warpSize * KVector + kPad);
        // constexpr index_t SingleVSize =
        // MakeVLdsBlockDescriptor<Problem>().get_element_space_size();
        constexpr index_t BufferSize =
            GetSingleSmemElementSpaceSize<Problem>(); //  max(SingleKSize, SingleVSize);

        constexpr auto k_lds_block_desc_0 =
            make_naive_tensor_descriptor(make_tuple(number<NumPrefetchK>{},       // num_buffers
                                                    number<NumIssues>{},          // n0
                                                    number<NumWarps>{},           // n2
                                                    number<LaneGroups>{},         // n1
                                                    number<kKPerBlock / KPack>{}, // k0
                                                    number<KPack>{}),             // k1
                                         make_tuple(number<BufferSize>{},
                                                    number<NumWarps*(warpSize * KVector + kPad)>{},
                                                    number<warpSize * KVector + kPad>{},
                                                    number<kKPerBlock>{},
                                                    number<KPack>{},
                                                    number<1>{}),
                                         number<KPack>{},
                                         number<1>{});

        constexpr auto k_lds_block_desc = transform_tensor_descriptor(
            k_lds_block_desc_0,
            make_tuple(
                make_merge_transform(make_tuple(number<NumPrefetchK>{},
                                                number<NumIssues>{},
                                                number<LaneGroups>{},
                                                number<NumWarps>{})),
                make_merge_transform(make_tuple(number<kKPerBlock / KPack>{}, number<KPack>{}))),
            make_tuple(sequence<0, 1, 3, 2>{}, sequence<4, 5>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        return k_lds_block_desc;
    }
#endif

    // 3d + padding
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeVLdsBlockDescriptor()
    {
        using VDataType                = remove_cvref_t<typename Problem::VDataType>;
        constexpr index_t Banks        = 32; // TODO: need change based on arch
        constexpr index_t PixelsPerRow = Banks * 4 / sizeof(VDataType);
        constexpr index_t kKPack       = GetSmemKPackV<Problem>();
        static_assert(PixelsPerRow % kKPack == 0);
        constexpr index_t NPerRow    = PixelsPerRow / kKPack;
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN1;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK1;
        static_assert(kNPerBlock % NPerRow == 0);
        static_assert(kKPerBlock % kKPack == 0);

        constexpr auto v_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(number<NumPrefetchV>{},
                       number<kKPerBlock / kKPack>{},
                       number<kNPerBlock / NPerRow>{},
                       number<NPerRow>{},
                       number<kKPack>{}),
            make_tuple(number<GetSingleSmemElementSpaceSize<Problem>()>{},
                       number<(kNPerBlock / NPerRow) * (PixelsPerRow + kKPack)>{},
                       number<PixelsPerRow + kKPack>{},
                       number<kKPack>{},
                       number<1>{}),
            number<kKPack>{},
            number<1>{});

        constexpr auto v_lds_block_desc = transform_tensor_descriptor(
            v_lds_block_desc_0,
            make_tuple(
                make_merge_transform(make_tuple(
                    number<NumPrefetchV>{}, number<kNPerBlock / NPerRow>{}, number<NPerRow>{})),
                make_merge_transform(make_tuple(number<kKPerBlock / kKPack>{}, number<kKPack>{}))),
            make_tuple(sequence<0, 2, 3>{}, sequence<1, 4>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        return v_lds_block_desc;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSizeKV()
    {
        // TODO: assume Q is in register
        // TODO: assume K/V has same data type
        constexpr index_t single_smem_size =
            GetSingleSmemElementSpaceSize<Problem>() * sizeof(typename Problem::KDataType);

        return QXPolicy::template GetSmemSizeQ<Problem>() +
               single_smem_size * max(NumPrefetchK, NumPrefetchV);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        if constexpr(AsyncCopyK)
        {
            return GetSmemSizeKV<Problem>() + GetSmemSizeDropout<Problem>(0);
        }
        else
        {
            return ck_tile::max(GetSmemSizeKV<Problem>(), GetSmemSizeDropout<Problem>(0));
        }
    }

    // this method is only available when Problem::kHasDropout is present
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr std::
        enable_if_t<std::is_convertible_v<decltype(Problem::kHasDropout), bool>, ck_tile::index_t>
        GetSmemSizeDropout(int)
    {
        if constexpr(Problem::kHasDropout)
        {
            constexpr auto gemm_0 = QXPolicy::template GetQKBlockGemm<Problem>();
            constexpr auto config =
                decltype(gemm_0)::Policy::template GetWarpGemmMWarpNWarp<Problem>();
            using WG                    = remove_cvref_t<decltype(config.template at<0>())>;
            constexpr index_t MWarp     = config.template at<1>();
            constexpr index_t kMPerStep = MWarp * WG::kM;
            constexpr index_t kNPerStep = WG::kN;

            return (kMPerStep + 1) * kNPerStep * sizeof(uint8_t);
        }
        else
        {
            return 0;
        }
    }

    // fallback version if Problem::kHasDropout is not exist
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSizeDropout(...)
    {
        return 0;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKDramTileDistribution()
    {
        if constexpr(!AsyncCopyK)
        {
            using KDataType = remove_cvref_t<typename Problem::KDataType>;

            constexpr index_t kBlockSize = Problem::kBlockSize;
            constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0;
            constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK0;

            constexpr index_t K1 = 16 / sizeof(KDataType);
            constexpr index_t K0 = kKPerBlock / K1;
            constexpr index_t N2 = get_warp_size() / K0;
            constexpr index_t N1 = kBlockSize / get_warp_size();
            constexpr index_t N0 = kNPerBlock / (N2 * N1);

            return make_static_tile_distribution(
                tile_distribution_encoding<sequence<1>,
                                           tuple<sequence<N0, N1, N2>, sequence<K0, K1>>,
                                           tuple<sequence<1>, sequence<1, 2>>,
                                           tuple<sequence<1>, sequence<2, 0>>,
                                           sequence<1, 2>,
                                           sequence<0, 1>>{});
        }
        else
        {
            constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0;
            constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK1;
            constexpr index_t kBlockSize = Problem::kBlockSize;
            constexpr index_t NumWarps   = Problem::BlockFmhaShape::NumWarps;
            constexpr index_t warpSize   = ck_tile::get_warp_size();

            constexpr index_t KVector = GetAlignmentK<Problem>(); // this is for global load

            static_assert(warpSize * KVector >= kKPerBlock && warpSize * KVector % kKPerBlock == 0);
            constexpr index_t LanesPerK  = kKPerBlock / KVector; // within a wave
            constexpr index_t LaneGroups = warpSize / LanesPerK; // within a wave
            constexpr index_t NumIssues  = kNPerBlock / (LaneGroups * NumWarps);
            static_assert(NumIssues == kNPerBlock * kKPerBlock / (kBlockSize * KVector));

            constexpr index_t N0 = NumIssues;
            constexpr index_t N1 = LaneGroups;
            constexpr index_t N2 = NumWarps;
            constexpr index_t K0 = LanesPerK;
            constexpr index_t K1 = KVector;

            return make_static_tile_distribution(
                tile_distribution_encoding<sequence<1>,
                                           tuple<sequence<N0, N1, N2>, sequence<K0, K1>>,
                                           tuple<sequence<1>, sequence<1, 2>>,
                                           tuple<sequence<2>, sequence<1, 0>>,
                                           sequence<1, 2>,
                                           sequence<0, 1>>{});
        }
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeVDramTileDistribution()
    {
        using VLayout = remove_cvref_t<typename Problem::BlockFmhaShape::VLayout>;

        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN1;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK1;

        if constexpr(std::is_same_v<VLayout, ck_tile::tensor_layout::gemm::RowMajor>)
        {
            constexpr index_t N1 = GetAlignmentV<Problem>();
            constexpr index_t N0 = kNPerBlock / N1; // P

            constexpr index_t total_pixels = kNPerBlock * kKPerBlock / kBlockSize;
            static_assert(total_pixels % N1 == 0); // TODO: this is not always true?
            constexpr index_t K3     = total_pixels / N1;
            constexpr index_t kKPack = GetSmemKPackV<Problem>();
            static_assert(kKPack % K3 == 0);
            constexpr index_t K2 = kKPack / K3; // TODO: this dimention could be outside single wave
            if constexpr(get_warp_size() % (K2 * N0) == 0)
            {
                constexpr index_t K1 = get_warp_size() / (K2 * N0);
                constexpr index_t K0 = kBlockSize / get_warp_size();
                static_assert(kKPerBlock == K0 * K1 * K2 * K3);
                return make_static_tile_distribution(
                    tile_distribution_encoding<sequence<1>,
                                               tuple<sequence<N0, N1>, sequence<K0, K1, K2, K3>>,
                                               tuple<sequence<2>, sequence<2, 1, 2>>,
                                               tuple<sequence<0>, sequence<1, 0, 2>>,
                                               sequence<2, 1>,
                                               sequence<3, 1>>{});
            }
            else
            {
                constexpr index_t K1   = (K2 * N0) / get_warp_size();
                constexpr index_t K2_m = K2 / K1;
                constexpr index_t K0   = kBlockSize / get_warp_size() / K1;
                static_assert(kKPerBlock == K0 * K1 * K2_m * K3);
                return make_static_tile_distribution(
                    tile_distribution_encoding<sequence<1>,
                                               tuple<sequence<N0, N1>, sequence<K0, K1, K2_m, K3>>,
                                               tuple<sequence<2, 2>, sequence<1, 2>>,
                                               tuple<sequence<0, 1>, sequence<0, 2>>,
                                               sequence<2, 1>,
                                               sequence<3, 1>>{});
            }
        }
        else
        {
            constexpr index_t K1 = GetAlignmentV<Problem>();
            constexpr index_t K0 = kKPerBlock / K1;
            constexpr index_t N2 = get_warp_size() / K0;
            constexpr index_t N1 = kBlockSize / get_warp_size();
            static_assert(N2 != 0, "N2 is zero, which will lead to a division by zero error.");
            static_assert(N1 != 0, "N1 is zero, which will lead to a division by zero error.");
            constexpr index_t N0 = kNPerBlock / (N2 * N1);
            static_assert(N0 != 0);

            return make_static_tile_distribution(
                tile_distribution_encoding<sequence<1>,
                                           tuple<sequence<N0, N1, N2>, sequence<K0, K1>>,
                                           tuple<sequence<1>, sequence<1, 2>>,
                                           tuple<sequence<1>, sequence<2, 0>>,
                                           sequence<1, 2>,
                                           sequence<0, 1>>{});
        }
    }

    template <typename BlockGemm>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBiasDramTileDistribution()
    {
        return BlockGemm::MakeCBlockTile().get_tile_distribution();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeShuffledVRegBlockDescriptor()
    {
        // This descriptor only used when V layout is seqlen * hdim
        using VLayout = remove_cvref_t<typename Problem::BlockFmhaShape::VLayout>;
        static_assert(std::is_same_v<VLayout, ck_tile::tensor_layout::gemm::RowMajor>);
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN1;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK1;

        constexpr index_t N1           = GetAlignmentV<Problem>();
        constexpr index_t N0           = kNPerBlock / N1;
        constexpr index_t total_pixels = kNPerBlock * kKPerBlock / kBlockSize;
        static_assert(total_pixels % N1 == 0); // TODO: this is not always true?
        constexpr index_t K3     = total_pixels / N1;
        constexpr index_t kKPack = GetSmemKPackV<Problem>();
        static_assert(kKPack % K3 == 0);
        constexpr index_t K2 = kKPack / K3; // TODO: this dimention could be outside single wave
        if constexpr(get_warp_size() % (K2 * N0) == 0)
        {
            constexpr index_t K1 = get_warp_size() / (K2 * N0);
            constexpr index_t K0 = kBlockSize / get_warp_size();

            return make_static_tile_distribution(
                tile_distribution_encoding<sequence<1>,
                                           tuple<sequence<N0, N1>, sequence<K0, K1, K2, K3>>,
                                           tuple<sequence<2>, sequence<2, 1, 2>>,
                                           tuple<sequence<0>, sequence<1, 0, 2>>,
                                           sequence<1, 2>,
                                           sequence<1, 3>>{});
        }
        else
        {
            constexpr index_t K1   = (K2 * N0) / get_warp_size();
            constexpr index_t K2_m = K2 / K1;
            constexpr index_t K0   = kBlockSize / get_warp_size() / K1;
            static_assert(kKPerBlock == K0 * K1 * K2_m * K3);
            return make_static_tile_distribution(
                tile_distribution_encoding<sequence<1>,
                                           tuple<sequence<N0, N1>, sequence<K0, K1, K2_m, K3>>,
                                           tuple<sequence<2, 2>, sequence<1, 2>>,
                                           tuple<sequence<0, 1>, sequence<0, 2>>,
                                           sequence<1, 2>,
                                           sequence<1, 3>>{});
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetKVBlockGemm()
    {
        using GemmProblem =
            BlockGemmProblem<typename Problem::PDataType,
                             typename Problem::VDataType,
                             typename Problem::OaccDataType,
                             Problem::kNumGemm1Warps * get_warp_size(),
                             TileGemmShape<sequence<Problem::BlockFmhaShape::kM0,
                                                    Problem::BlockFmhaShape::kN1,
                                                    Problem::BlockFmhaShape::kK1>,
                                           typename Problem::BlockFmhaShape::Gemm1BlockWarps,
                                           typename Problem::BlockFmhaShape::Gemm1WarpTile>>;

        auto warp_gemm = [&]() {
            if constexpr(std::is_same_v<typename Problem::KDataType, fp8_t> &&
                         std::is_same_v<typename Problem::VDataType, fp8_t> &&
                         std::is_same_v<typename Problem::OaccDataType, float>)
            {
                return WarpGemmMfmaFp8Fp8F32M32N32K16SwizzleBTransposedCDistribution<>{};
                // return
                // WarpGemmImpl<WarpGemmAtrributeMfmaTransposedCDistribution_SwizzleB<
                //         WarpGemmAttributeMfmaImpl_f32_32x32x16_f8_base<typename
                //         Problem::PDataType, typename Problem::VDataType>>>{};
            }
            else
            {
                return WarpGemmMfmaDispatcher<
                    typename Problem::PDataType,
                    typename Problem::VDataType,
                    typename Problem::OaccDataType,
                    Problem::BlockFmhaShape::Gemm1WarpTile::at(number<0>{}),
                    Problem::BlockFmhaShape::Gemm1WarpTile::at(number<1>{}),
                    Problem::BlockFmhaShape::Gemm1WarpTile::at(number<2>{}),
                    true>{};
            }
        }();

        using WarpGemm = remove_cvref_t<decltype(warp_gemm)>;

        using BlockGemmPolicy =
            BlockGemmARegBSmemCRegV2CustomPolicy<typename Problem::PDataType,
                                                 typename Problem::VDataType,
                                                 typename Problem::OaccDataType,
                                                 typename Problem::BlockFmhaShape::Gemm1BlockWarps,
                                                 WarpGemm>;
        return BlockGemmARegBSmemCRegV2<GemmProblem, BlockGemmPolicy>{};
    }
};

} // namespace ck_tile
