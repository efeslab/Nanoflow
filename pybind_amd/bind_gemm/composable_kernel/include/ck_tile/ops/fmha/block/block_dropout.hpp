// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm.hpp"

namespace ck_tile {

struct NullBlockDropout
{
    template <typename BlockGemm, bool IsFwd = true, typename RandValDramBlockWindowTmp>
    __host__ __device__ static constexpr auto
    MakeRandvalDramWindow(RandValDramBlockWindowTmp& randval_dram_block_window_tmp,
                          index_t seqlen_qk_start)
    {
        (void)randval_dram_block_window_tmp;
        (void)seqlen_qk_start;

        return make_null_tile_window(make_tuple(number<0>{}, number<0>{}));
    }
};

struct BlockDropout
{
    CK_TILE_HOST_DEVICE BlockDropout(index_t i_batch,
                                     index_t i_head,
                                     index_t nheads,
                                     unsigned long long seed,
                                     unsigned long long offset,
                                     float rp_undrop_,
                                     uint8_t p_undrop_in_uint8_t_,
                                     bool is_store_randval_)
        : ph(seed, offset + (i_batch * nheads + i_head) * get_warp_size() + get_lane_id()),
          rp_undrop(rp_undrop_),
          p_undrop_in_uint8_t(p_undrop_in_uint8_t_),
          is_store_randval(is_store_randval_)
    {
    }

    template <typename BlockGemm, bool IsFwd = true, typename RandValDramBlockWindowTmp>
    CK_TILE_HOST_DEVICE static constexpr auto
    MakeRandvalDramWindow(RandValDramBlockWindowTmp& randval_dram_block_window_tmp,
                          index_t seqlen_qk_start)
    {
        constexpr auto config =
            BlockGemm::Policy::template GetWarpGemmMWarpNWarp<typename BlockGemm::Problem>();
        using WG                    = remove_cvref_t<decltype(config.template at<0>())>;
        constexpr index_t MWarp     = config.template at<1>();
        constexpr index_t NWarp     = config.template at<2>();
        constexpr index_t kMPerStep = MWarp * WG::kM;
        constexpr index_t kNPerStep = NWarp * WG::kN;

        const auto block_origin  = randval_dram_block_window_tmp.get_window_origin();
        auto randval_dram_window = [&]() {
            if constexpr(IsFwd)
            {
                return make_tile_window(
                    randval_dram_block_window_tmp.get_bottom_tensor_view(),
                    ck_tile::make_tuple(number<kMPerStep>{}, number<kNPerStep>{}),
                    {block_origin.at(number<0>{}), seqlen_qk_start}); // M/N
            }
            else
            {
                return make_tile_window(
                    randval_dram_block_window_tmp.get_bottom_tensor_view(),
                    ck_tile::make_tuple(number<kMPerStep>{}, number<kNPerStep>{}),
                    {seqlen_qk_start, block_origin.at(number<1>{})}); // M/N
            }
        }();

        return randval_dram_window;
    }

    template <typename BlockGemm>
    CK_TILE_HOST_DEVICE static constexpr auto MakeRandValLdsBlockDescriptor()
    {
        constexpr auto config =
            BlockGemm::Policy::template GetWarpGemmMWarpNWarp<typename BlockGemm::Problem>();
        using WG                    = remove_cvref_t<decltype(config.template at<0>())>;
        constexpr index_t MWarp     = config.template at<1>();
        constexpr index_t kMPerStep = MWarp * WG::kM;
        constexpr index_t kNPerStep = WG::kN;
        constexpr index_t kN1       = 8;
        constexpr index_t kN0       = kNPerStep / kN1;

        constexpr auto randval_lds_block_desc_0 = make_naive_tensor_descriptor(
            ck_tile::make_tuple(number<kN0>{}, number<kMPerStep>{}, number<kN1>{}),
            ck_tile::make_tuple(number<(kMPerStep + 1) * kN1>{}, number<kN1>{}, number<1>{}),
            number<kN1>{},
            number<1>{});

        constexpr auto randval_lds_block_desc = transform_tensor_descriptor(
            randval_lds_block_desc_0,
            ck_tile::make_tuple(
                make_pass_through_transform(number<kMPerStep>{}),
                make_merge_transform(ck_tile::make_tuple(number<kN0>{}, number<kN1>{}))),
            ck_tile::make_tuple(sequence<1>{}, sequence<0, 2>{}),
            ck_tile::make_tuple(sequence<0>{}, sequence<1>{}));

        return randval_lds_block_desc;
    }

    template <typename BlockGemm>
    CK_TILE_HOST_DEVICE static constexpr auto MakeRandValTileDistribution()
    {
        constexpr auto config =
            BlockGemm::Policy::template GetWarpGemmMWarpNWarp<typename BlockGemm::Problem>();
        constexpr index_t MWarp = config.template at<1>();
        constexpr index_t NWarp = config.template at<2>();

        constexpr index_t MIterPerWarp = 1;
        constexpr index_t NIterPerWarp = 1;

        constexpr auto randval_block_outer_part_dstr_encoding = tile_distribution_encoding<
            sequence<>,
            tuple<sequence<MIterPerWarp, MWarp>, sequence<NIterPerWarp, NWarp>>,
            tuple<sequence<1, 2>>,
            tuple<sequence<1, 1>>,
            sequence<1, 2>,
            sequence<0, 0>>{};

        // Use Bwd WarpGemm to ensure that Fwd's random values ​​are consistent with Bwd.
        constexpr auto randval_block_inner_part_dstr_encoding = []() {
            if constexpr(std::is_same_v<typename BlockGemm::ADataType, half_t> &&
                         std::is_same_v<typename BlockGemm::BDataType, half_t> &&
                         std::is_same_v<typename BlockGemm::CDataType, float>)
            {
                return typename WarpGemmMfmaF16F16F32M32N32K16SwizzleA::CWarpDstrEncoding{};
            }
            else
            {
                return typename WarpGemmMfmaBf16Bf16F32M32N32K16SwizzleA::CWarpDstrEncoding{};
            }
        }();

        constexpr auto randval_block_part_dstr_encode =
            detail::make_embed_tile_distribution_encoding(randval_block_outer_part_dstr_encoding,
                                                          randval_block_inner_part_dstr_encoding);

        return make_static_tile_distribution(randval_block_part_dstr_encode);
    }

    template <typename BlockGemm>
    CK_TILE_HOST_DEVICE static constexpr auto MakeRandValLdsShuffleTileDistribution()
    {
        constexpr auto config =
            BlockGemm::Policy::template GetWarpGemmMWarpNWarp<typename BlockGemm::Problem>();
        using WG                = remove_cvref_t<decltype(config.template at<0>())>;
        constexpr index_t MWarp = config.template at<1>();
        constexpr index_t NWarp = config.template at<2>();

        constexpr index_t MIterPerWarp = 1;
        constexpr index_t NIterPerWarp = 1;

        constexpr auto randval_block_outer_part_dstr_encoding = tile_distribution_encoding<
            sequence<>,
            tuple<sequence<MIterPerWarp, MWarp>, sequence<NIterPerWarp, NWarp>>,
            tuple<sequence<1, 2>>,
            tuple<sequence<1, 1>>,
            sequence<1, 2>,
            sequence<0, 0>>{};

        constexpr auto randval_block_part_dstr_encode =
            detail::make_embed_tile_distribution_encoding(randval_block_outer_part_dstr_encoding,
                                                          typename WG::CWarpDstrEncoding{});

        return make_static_tile_distribution(randval_block_part_dstr_encode);
    }

    template <typename BlockGemm,
              typename PComputeDataType,
              typename RandValOutputDataType,
              typename PComputeWindow,
              typename RandValDramWindow>
    CK_TILE_HOST_DEVICE void Run(void* randval_ptr,
                                 const index_t start_n0_idx,
                                 PComputeWindow& p_compute,
                                 RandValDramWindow& randval_dram_window) const
    {
        constexpr auto config =
            BlockGemm::Policy::template GetWarpGemmMWarpNWarp<typename BlockGemm::Problem>();
        using WG                     = remove_cvref_t<decltype(config.template at<0>())>;
        constexpr index_t MWarp      = config.template at<1>();
        constexpr index_t NWarp      = config.template at<2>();
        using BlockGemmShape         = remove_cvref_t<typename BlockGemm::BlockGemmShape>;
        constexpr index_t kMPerBlock = BlockGemmShape::kM;
        constexpr index_t kNPerBlock = BlockGemmShape::kN;
        constexpr index_t kMPerStep  = MWarp * WG::kM;
        constexpr index_t kNPerStep  = NWarp * WG::kN;

        // randval tile in LDS
        auto randval_lds = make_tensor_view<address_space_enum::lds>(
            reinterpret_cast<uint8_t*>(randval_ptr), MakeRandValLdsBlockDescriptor<BlockGemm>());

        auto randval_lds_window = make_tile_window(
            randval_lds, MakeRandValLdsBlockDescriptor<BlockGemm>().get_lengths(), {0, 0});

        // register distribute
        auto randval_dist_generated =
            make_static_distributed_tensor<uint8_t>(MakeRandValTileDistribution<BlockGemm>());
        static_assert(randval_dist_generated.kThreadElementSpaceSize == 16);

        auto randval_lds_read_window =
            make_tile_window(randval_lds_window.get_bottom_tensor_view(),
                             randval_lds_window.get_window_lengths(),
                             randval_lds_window.get_window_origin(),
                             MakeRandValLdsShuffleTileDistribution<BlockGemm>());

        const int start_m0_idx = randval_dram_window.get_window_origin().at(number<0>{});
        if(is_store_randval)
        {
            static_for<0, kMPerBlock / kMPerStep, 1>{}([&](auto i_m0) {
                static_for<0, kNPerBlock / kNPerStep, 1>{}([&](auto i_n0) {
                    int block_row_start = (start_m0_idx / WG::kM) + (i_m0 * MWarp) + get_warp_id();
                    int block_col_start = (start_n0_idx / WG::kN) + i_n0;
                    uint2 rowcol        = make_uint2(block_row_start, block_col_start);

                    // generate random number
                    uint8_t random_uint8_t[16];
                    ph.get_random_16x8(random_uint8_t,
                                       reinterpret_cast<unsigned long long&>(rowcol));

                    constexpr auto randval_dist_generated_spans =
                        decltype(randval_dist_generated)::get_distributed_spans();
                    int i_random_idx = 0;
                    sweep_tile_span(randval_dist_generated_spans[number<0>{}], [&](auto idx0) {
                        sweep_tile_span(randval_dist_generated_spans[number<1>{}], [&](auto idx1) {
                            constexpr auto i_j_idx          = ck_tile::make_tuple(idx0, idx1);
                            randval_dist_generated(i_j_idx) = random_uint8_t[i_random_idx++];
                        });
                    });
                    // save to LDS
                    store_tile(randval_lds_window, randval_dist_generated);
                    block_sync_lds();
                    // read from LDS to register
                    auto randval = load_tile(randval_lds_read_window);
                    // save to Global
                    const auto randval_store = cast_tile<RandValOutputDataType>(randval);
                    store_tile(randval_dram_window, randval_store);
                    move_tile_window(randval_dram_window, {0, kNPerStep});
                });
                move_tile_window(randval_dram_window, {kMPerStep, -kNPerBlock});
            });
            move_tile_window(randval_dram_window, {-kMPerBlock, kNPerBlock});
        };
        static_for<0, kMPerBlock / kMPerStep, 1>{}([&](auto i_m0) {
            static_for<0, kNPerBlock / kNPerStep, 1>{}([&](auto i_n0) {
                int block_row_start = (start_m0_idx / WG::kM) + (i_m0 * MWarp) + get_warp_id();
                int block_col_start = (start_n0_idx / WG::kN) + i_n0;
                uint2 rowcol        = make_uint2(block_row_start, block_col_start);

                // generate random number
                uint8_t random_uint8_t[16];
                ph.get_random_16x8(random_uint8_t, reinterpret_cast<unsigned long long&>(rowcol));

                constexpr auto randval_dist_generated_spans =
                    decltype(randval_dist_generated)::get_distributed_spans();
                int i_random_idx = 0;
                sweep_tile_span(randval_dist_generated_spans[number<0>{}], [&](auto idx0) {
                    sweep_tile_span(randval_dist_generated_spans[number<1>{}], [&](auto idx1) {
                        constexpr auto i_j_idx          = ck_tile::make_tuple(idx0, idx1);
                        randval_dist_generated(i_j_idx) = random_uint8_t[i_random_idx++];
                    });
                });
                // save to LDS
                store_tile(randval_lds_window, randval_dist_generated);
                block_sync_lds();
                // read from LDS to register
                auto randval                 = load_tile(randval_lds_read_window);
                constexpr auto randval_spans = decltype(randval)::get_distributed_spans();
                sweep_tile_span(randval_spans[number<0>{}], [&](auto idx0) {
                    sweep_tile_span(randval_spans[number<1>{}], [&](auto idx1) {
                        constexpr auto p_idx0 = tile_distributed_index<i_m0>{};
                        constexpr auto p_idx1 =
                            tile_distributed_index<i_n0, idx1.impl_.at(1), idx1.impl_.at(2)>{};
                        constexpr auto p_idx = ck_tile::make_tuple(p_idx0, p_idx1);
                        constexpr auto r_idx = ck_tile::make_tuple(idx0, idx1);
                        p_compute(p_idx)     = randval[r_idx] <= p_undrop_in_uint8_t
                                                   ? p_compute[p_idx] * rp_undrop
                                                   : PComputeDataType(0);
                    });
                });
            });
        });
    }

    ck_tile::philox ph;
    const float rp_undrop;
    const uint8_t p_undrop_in_uint8_t;
    const bool is_store_randval;
};

template <bool IsDropout_, bool IsWG32_, bool IsStoreRandval_>
struct BlockDropoutBwd;

template <bool IsWG32_, bool IsStoreRandval_>
struct BlockDropoutBwd<false, IsWG32_, IsStoreRandval_>
{
    static constexpr bool IsDropout      = false;
    static constexpr bool IsStoreRandval = IsStoreRandval_;

    template <typename BlockGemm, bool IsFwd = true, typename RandValDramBlockWindowTmp>
    __host__ __device__ static constexpr auto
    MakeRandvalDramWindow(RandValDramBlockWindowTmp& randval_dram_block_window_tmp,
                          index_t seqlen_qk_start)
    {
        (void)randval_dram_block_window_tmp;
        (void)seqlen_qk_start;

        return make_null_tile_window(make_tuple(number<0>{}, number<0>{}));
    }
};

template <bool IsWG32_, bool IsStoreRandval_>
struct BlockDropoutBwd<true, IsWG32_, IsStoreRandval_>
{
    static constexpr bool IsDropout = true;
    // true:  32*32 warp gemm
    // false: 16*16 warp gemm
    static constexpr bool IsWG32         = IsWG32_;
    static constexpr bool IsStoreRandval = IsStoreRandval_;

    CK_TILE_HOST_DEVICE BlockDropoutBwd(index_t i_batch,
                                        index_t i_head,
                                        index_t nheads,
                                        unsigned long long seed,
                                        unsigned long long offset,
                                        float rp_undrop_,
                                        uint8_t p_undrop_in_uint8_t_)
        : ph(seed,
             offset + (i_batch * nheads + i_head) * get_warp_size() +
                 (IsWG32 ? get_lane_id() : ((get_lane_id() & 47) + ((get_warp_id() & 1) << 4)))),
          rp_undrop(rp_undrop_),
          p_undrop_in_uint8_t(p_undrop_in_uint8_t_)
    {
    }

    template <typename BlockGemm, bool IsFwd = true, typename RandValDramBlockWindowTmp>
    CK_TILE_HOST_DEVICE static constexpr auto
    MakeRandvalDramWindow(RandValDramBlockWindowTmp& randval_dram_block_window_tmp,
                          index_t seqlen_qk_start)
    {
        constexpr auto config =
            BlockGemm::Policy::template GetWarpGemmMWarpNWarp<typename BlockGemm::Problem>();
        using BlockGemmShape                  = remove_cvref_t<typename BlockGemm::BlockGemmShape>;
        using WG                              = remove_cvref_t<decltype(config.template at<0>())>;
        constexpr index_t kMPerBlock          = BlockGemmShape::kM;
        constexpr index_t MWarp               = config.template at<1>();
        constexpr index_t NWarp               = config.template at<2>();
        constexpr bool MBwdWG16MultiIterCheck = (!IsFwd) && (!IsWG32) && (kMPerBlock > 16);
        constexpr index_t kMPerStep           = [&]() {
            if constexpr(MBwdWG16MultiIterCheck)
            {
                return MWarp * WG::kM * 2;
            }
            else
            {
                return MWarp * WG::kM;
            }
        }();
        constexpr index_t kNPerStep = NWarp * WG::kN;

        const auto block_origin  = randval_dram_block_window_tmp.get_window_origin();
        auto randval_dram_window = [&]() {
            if constexpr(IsFwd)
            {
                return make_tile_window(
                    randval_dram_block_window_tmp.get_bottom_tensor_view(),
                    ck_tile::make_tuple(number<kMPerStep>{}, number<kNPerStep>{}),
                    {block_origin.at(number<0>{}), seqlen_qk_start}); // M/N
            }
            else
            {
                return make_tile_window(
                    randval_dram_block_window_tmp.get_bottom_tensor_view(),
                    ck_tile::make_tuple(number<kMPerStep>{}, number<kNPerStep>{}),
                    {seqlen_qk_start, block_origin.at(number<1>{})}); // M/N
            }
        }();

        return randval_dram_window;
    }

    template <typename BlockGemm>
    CK_TILE_HOST_DEVICE static constexpr auto MakeRandValLdsBlockDescriptor()
    {
        constexpr auto config =
            BlockGemm::Policy::template GetWarpGemmMWarpNWarp<typename BlockGemm::Problem>();
        using WG                    = remove_cvref_t<decltype(config.template at<0>())>;
        constexpr index_t MWarp     = config.template at<1>();
        constexpr index_t kMPerStep = MWarp * WG::kM;
        constexpr index_t kNPerStep = WG::kN;
        constexpr index_t kN1       = 8;
        constexpr index_t kN0       = kNPerStep / kN1;

        constexpr auto randval_lds_block_desc_0 = make_naive_tensor_descriptor(
            ck_tile::make_tuple(number<kN0>{}, number<kMPerStep>{}, number<kN1>{}),
            ck_tile::make_tuple(number<(kMPerStep + 1) * kN1>{}, number<kN1>{}, number<1>{}),
            number<kN1>{},
            number<1>{});

        constexpr auto randval_lds_block_desc = transform_tensor_descriptor(
            randval_lds_block_desc_0,
            ck_tile::make_tuple(
                make_pass_through_transform(number<kMPerStep>{}),
                make_merge_transform(ck_tile::make_tuple(number<kN0>{}, number<kN1>{}))),
            ck_tile::make_tuple(sequence<1>{}, sequence<0, 2>{}),
            ck_tile::make_tuple(sequence<0>{}, sequence<1>{}));

        return randval_lds_block_desc;
    }

    template <typename BlockGemm, bool IsFwd = true>
    CK_TILE_HOST_DEVICE static constexpr auto MakeRandValTileDistribution()
    {
        constexpr auto config =
            BlockGemm::Policy::template GetWarpGemmMWarpNWarp<typename BlockGemm::Problem>();
        using BlockGemmShape                  = remove_cvref_t<typename BlockGemm::BlockGemmShape>;
        constexpr index_t kMPerBlock          = BlockGemmShape::kM;
        constexpr index_t MWarp               = config.template at<1>();
        constexpr index_t NWarp               = config.template at<2>();
        constexpr bool MBwdWG16MultiIterCheck = (!IsFwd) && (!IsWG32) && (kMPerBlock > 16);

        constexpr index_t MIterPerWarp = [&]() {
            if constexpr(MBwdWG16MultiIterCheck)
            {
                return 2;
            }
            else
            {
                return 1;
            }
        }();
        constexpr index_t NIterPerWarp = 1;

        constexpr auto randval_block_outer_part_dstr_encoding = tile_distribution_encoding<
            sequence<>,
            tuple<sequence<MIterPerWarp, MWarp>, sequence<NIterPerWarp, NWarp>>,
            tuple<sequence<1, 2>>,
            tuple<sequence<1, 1>>,
            sequence<1, 2>,
            sequence<0, 0>>{};

        // Use Bwd WarpGemm to ensure that Fwd's random values ​​are consistent with Bwd.
        // except headdim256.
        constexpr auto randval_block_inner_part_dstr_encoding = []() {
            if constexpr(std::is_same_v<typename BlockGemm::ADataType, half_t> &&
                         std::is_same_v<typename BlockGemm::BDataType, half_t> &&
                         std::is_same_v<typename BlockGemm::CDataType, float>)
            {
                if constexpr(IsWG32)
                    return typename WarpGemmMfmaF16F16F32M32N32K16SwizzleA::CWarpDstrEncoding{};
                else
                    return typename WarpGemmMfmaF16F16F32M16N16K16::CWarpDstrEncoding{};
            }
            else
            {
                if constexpr(IsWG32)
                    return typename WarpGemmMfmaBf16Bf16F32M32N32K16SwizzleA::CWarpDstrEncoding{};
                else
                    return typename WarpGemmMfmaBf16Bf16F32M16N16K16::CWarpDstrEncoding{};
            }
        }();

        constexpr auto randval_block_part_dstr_encode =
            detail::make_embed_tile_distribution_encoding(randval_block_outer_part_dstr_encoding,
                                                          randval_block_inner_part_dstr_encoding);

        return make_static_tile_distribution(randval_block_part_dstr_encode);
    }

    template <typename BlockGemm>
    CK_TILE_HOST_DEVICE static constexpr auto MakeRandValLdsShuffleTileDistribution()
    {
        constexpr auto config =
            BlockGemm::Policy::template GetWarpGemmMWarpNWarp<typename BlockGemm::Problem>();
        using WG                = remove_cvref_t<decltype(config.template at<0>())>;
        constexpr index_t MWarp = config.template at<1>();
        constexpr index_t NWarp = config.template at<2>();

        constexpr index_t MIterPerWarp = 1;
        constexpr index_t NIterPerWarp = 1;

        constexpr auto randval_block_outer_part_dstr_encoding = tile_distribution_encoding<
            sequence<>,
            tuple<sequence<MIterPerWarp, MWarp>, sequence<NIterPerWarp, NWarp>>,
            tuple<sequence<1, 2>>,
            tuple<sequence<1, 1>>,
            sequence<1, 2>,
            sequence<0, 0>>{};

        constexpr auto randval_block_part_dstr_encode =
            detail::make_embed_tile_distribution_encoding(randval_block_outer_part_dstr_encoding,
                                                          typename WG::CWarpDstrEncoding{});

        return make_static_tile_distribution(randval_block_part_dstr_encode);
    }

    template <typename BlockGemm,
              typename PComputeDataType,
              typename RandValOutputDataType,
              typename PComputeWindow,
              typename RandValDramWindow>
    CK_TILE_HOST_DEVICE void Run(void* randval_ptr,
                                 const index_t start_m0_idx,
                                 const index_t start_n0_idx,
                                 PComputeWindow& p_compute,
                                 RandValDramWindow& randval_dram_window) const
    {
        constexpr auto config =
            BlockGemm::Policy::template GetWarpGemmMWarpNWarp<typename BlockGemm::Problem>();
        using WG                     = remove_cvref_t<decltype(config.template at<0>())>;
        constexpr index_t MWarp      = config.template at<1>();
        constexpr index_t NWarp      = config.template at<2>();
        using BlockGemmShape         = remove_cvref_t<typename BlockGemm::BlockGemmShape>;
        constexpr index_t kMPerBlock = BlockGemmShape::kM;
        constexpr index_t kNPerBlock = BlockGemmShape::kN;
        constexpr index_t kMPerStep  = MWarp * WG::kM;
        constexpr index_t kNPerStep  = NWarp * WG::kN;

        // randval tile in LDS
        auto randval_lds = make_tensor_view<address_space_enum::lds>(
            reinterpret_cast<uint8_t*>(randval_ptr), MakeRandValLdsBlockDescriptor<BlockGemm>());

        auto randval_lds_window = make_tile_window(
            randval_lds, MakeRandValLdsBlockDescriptor<BlockGemm>().get_lengths(), {0, 0});

        // register distribute
        auto randval_dist_generated =
            make_static_distributed_tensor<uint8_t>(MakeRandValTileDistribution<BlockGemm>());
        static_assert(randval_dist_generated.kThreadElementSpaceSize == 16);

        auto randval_lds_read_window =
            make_tile_window(randval_lds_window.get_bottom_tensor_view(),
                             randval_lds_window.get_window_lengths(),
                             randval_lds_window.get_window_origin(),
                             MakeRandValLdsShuffleTileDistribution<BlockGemm>());

        static_for<0, kMPerBlock / kMPerStep, 1>{}([&](auto i_m0) {
            static_for<0, kNPerBlock / kNPerStep, 1>{}([&](auto i_n0) {
                int block_row_start = (start_m0_idx / WG::kM) + (i_m0 * MWarp) + get_warp_id();
                int block_col_start = (start_n0_idx / WG::kN) + i_n0;
                uint2 rowcol        = make_uint2(block_row_start, block_col_start);

                // generate random number
                uint8_t random_uint8_t[16];
                ph.get_random_16x8(random_uint8_t, reinterpret_cast<unsigned long long&>(rowcol));

                constexpr auto randval_dist_generated_spans =
                    decltype(randval_dist_generated)::get_distributed_spans();
                int i_random_idx = 0;
                sweep_tile_span(randval_dist_generated_spans[number<0>{}], [&](auto idx0) {
                    sweep_tile_span(randval_dist_generated_spans[number<1>{}], [&](auto idx1) {
                        constexpr auto i_j_idx          = ck_tile::make_tuple(idx0, idx1);
                        randval_dist_generated(i_j_idx) = random_uint8_t[i_random_idx++];
                    });
                });
                // save to LDS
                store_tile(randval_lds_window, randval_dist_generated);
                block_sync_lds();
                // read from LDS to register
                auto randval                 = load_tile(randval_lds_read_window);
                constexpr auto randval_spans = decltype(randval)::get_distributed_spans();
                sweep_tile_span(randval_spans[number<0>{}], [&](auto idx0) {
                    sweep_tile_span(randval_spans[number<1>{}], [&](auto idx1) {
                        constexpr auto p_idx0 = tile_distributed_index<i_m0>{};
                        constexpr auto p_idx1 =
                            tile_distributed_index<i_n0, idx1.impl_.at(1), idx1.impl_.at(2)>{};
                        constexpr auto p_idx = ck_tile::make_tuple(p_idx0, p_idx1);
                        constexpr auto r_idx = ck_tile::make_tuple(idx0, idx1);
                        p_compute(p_idx)     = randval[r_idx] <= p_undrop_in_uint8_t
                                                   ? p_compute[p_idx] * rp_undrop
                                                   : PComputeDataType(0);
                    });
                });
                // save to Global
                if constexpr(IsStoreRandval)
                {
                    const auto randval_store = cast_tile<RandValOutputDataType>(randval);
                    store_tile(randval_dram_window, randval_store);
                    move_tile_window(randval_dram_window, {0, kNPerStep});
                }
            });
            if constexpr(IsStoreRandval)
            {
                move_tile_window(randval_dram_window, {kMPerStep, -kNPerBlock});
            }
        });
        if constexpr(IsStoreRandval)
        {
            move_tile_window(randval_dram_window, {-kMPerBlock, kNPerBlock});
        }
    }

    template <typename BlockGemm,
              typename RandValOutputDataType,
              typename PComputeWindow,
              typename RandValDramWindow>
    CK_TILE_HOST_DEVICE void Run(const index_t start_m0_idx,
                                 const index_t start_n0_idx,
                                 PComputeWindow& p_compute,
                                 RandValDramWindow& randval_dram_window) const
    {
        constexpr auto config =
            BlockGemm::Policy::template GetWarpGemmMWarpNWarp<typename BlockGemm::Problem>();
        using WG                               = remove_cvref_t<decltype(config.template at<0>())>;
        constexpr index_t MWarp                = config.template at<1>();
        constexpr index_t NWarp                = config.template at<2>();
        using BlockGemmShape                   = remove_cvref_t<typename BlockGemm::BlockGemmShape>;
        constexpr index_t kMPerBlock           = BlockGemmShape::kM;
        constexpr index_t kNPerBlock           = BlockGemmShape::kN;
        constexpr bool MBwdWG16MultiIterCheck  = (!IsWG32) && (kMPerBlock > 16);
        constexpr bool MBwdWG16SingleIterCheck = (!IsWG32) && (kMPerBlock == 16);
        constexpr index_t kMPerStep            = [&]() {
            if constexpr(MBwdWG16MultiIterCheck)
            {
                return MWarp * WG::kM * 2;
            }
            else
            {
                return MWarp * WG::kM;
            }
        }();
        constexpr index_t kNPerStep = NWarp * WG::kN;

        // register distribute
        auto randval = make_static_distributed_tensor<uint8_t>(
            MakeRandValTileDistribution<BlockGemm, false>());
        if constexpr(IsWG32)
            static_assert(randval.kThreadElementSpaceSize == 16);
        else
            static_assert(randval.kThreadElementSpaceSize == 4 ||
                          randval.kThreadElementSpaceSize == 8);

        static_for<0, kNPerBlock / kNPerStep, 1>{}([&](auto i_n0) {
            static_for<0, kMPerBlock / kMPerStep, 1>{}([&](auto i_m0) {
                int block_row_start, block_col_start;
                if constexpr(IsWG32)
                {
                    block_row_start = (start_m0_idx / WG::kM) + i_m0;
                    block_col_start = (start_n0_idx / WG::kN) + (i_n0 * NWarp) + get_warp_id();
                }
                else
                {
                    block_row_start = start_m0_idx / 32 + i_m0;
                    block_col_start = (start_n0_idx / 32) + get_warp_id() / 2 + i_n0 * 2;
                }
                uint2 rowcol = make_uint2(block_row_start, block_col_start);

                // generate random number
                uint8_t* random_uint8_t_;
                if constexpr(MBwdWG16SingleIterCheck)
                {
                    uint8_t random_uint8_t[4];
                    // m0t0 ~m0t15/m0t32~m0t47: 0
                    // m0t16~m0t31/m0t48~m0t63: 1
                    // m1t0 ~m1t15/m1t32~m1t47: 2
                    // m1t16~m1t31/m1t48~m1t63: 3
                    const index_t start_idx =
                        ((get_lane_id() >> 4) & 1) + (((start_m0_idx >> 4) & 1) << 1);
                    ph.get_random_4x8(
                        random_uint8_t, reinterpret_cast<unsigned long long&>(rowcol), start_idx);
                    random_uint8_t_ = random_uint8_t;
                }
                else if constexpr(MBwdWG16MultiIterCheck)
                {
                    uint8_t random_uint8_t[8];
                    // t0 ~t15/t32~t47: 0
                    // t16~t31/t48~t63: 1
                    const index_t start_idx = (get_lane_id() >> 4) & 1;
                    ph.get_random_8x8(
                        random_uint8_t, reinterpret_cast<unsigned long long&>(rowcol), start_idx);
                    random_uint8_t_ = random_uint8_t;
                }
                else
                {
                    uint8_t random_uint8_t[16];
                    ph.get_random_16x8(random_uint8_t,
                                       reinterpret_cast<unsigned long long&>(rowcol));
                    random_uint8_t_ = random_uint8_t;
                }

                constexpr auto randval_spans = decltype(randval)::get_distributed_spans();
                int i_random_idx             = 0;
                sweep_tile_span(randval_spans[number<0>{}], [&](auto idx0) {
                    sweep_tile_span(randval_spans[number<1>{}], [&](auto idx1) {
                        constexpr auto r_idx  = ck_tile::make_tuple(idx0, idx1);
                        randval(r_idx)        = random_uint8_t_[i_random_idx++];
                        constexpr auto p_idx0 = tile_distributed_index<i_m0 + idx0.impl_.at(0),
                                                                       idx0.impl_.at(1),
                                                                       idx0.impl_.at(2)>{};
                        constexpr auto p_idx1 = tile_distributed_index<i_n0>{};
                        constexpr auto p_idx  = ck_tile::make_tuple(p_idx0, p_idx1);
                        p_compute(p_idx)      = randval[r_idx] <= p_undrop_in_uint8_t
                                                    ? p_compute[p_idx]
                                                    : -p_compute[p_idx];
                    });
                });
                // save to Global
                if constexpr(IsStoreRandval)
                {
                    const auto randval_store = cast_tile<RandValOutputDataType>(randval);
                    store_tile(randval_dram_window, randval_store);
                    move_tile_window(randval_dram_window, {kMPerStep, 0});
                }
            });
            if constexpr(IsStoreRandval)
            {
                move_tile_window(randval_dram_window, {-kMPerBlock, kNPerStep});
            }
        });
        if constexpr(IsStoreRandval)
        {
            move_tile_window(randval_dram_window, {kMPerBlock, -kNPerBlock});
        }
    }

    ck_tile::philox ph;
    const float rp_undrop;
    const uint8_t p_undrop_in_uint8_t;
};

} // namespace ck_tile
