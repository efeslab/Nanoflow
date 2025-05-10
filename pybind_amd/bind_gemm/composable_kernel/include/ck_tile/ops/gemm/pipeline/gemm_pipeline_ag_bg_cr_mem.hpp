// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_agmem_bgmem_creg_v1_default_policy.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_scheduler.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_base.hpp"

namespace ck_tile {

//  A Tile Window: global memory
//  B Tile Window: global memory
//  C Distributed tensor: register
template <typename Problem>
struct BaseGemmPipelineAgBgCrMem
{
    using ADataType      = remove_cvref_t<typename Problem::ADataType>;
    using BDataType      = remove_cvref_t<typename Problem::BDataType>;
    using BlockGemmShape = remove_cvref_t<typename Problem::BlockGemmShape>;

    CK_TILE_HOST_DEVICE static constexpr auto TransposeC() { return Problem::TransposeC; }

    static constexpr index_t BlockSize = Problem::kBlockSize;
    static constexpr index_t MPerBlock = BlockGemmShape::kM;
    static constexpr index_t NPerBlock = BlockGemmShape::kN;
    static constexpr index_t KPerBlock = BlockGemmShape::kK;

    // TODO: Is this 32K value gfx9 arch specific?
    static constexpr index_t MinMemInFlyBytes = 32768;

    static constexpr index_t WgpPerCU =
        (4 * get_warp_size() / BlockSize) >= 1 ? 4 * get_warp_size() / BlockSize : 1;
    static constexpr index_t FullMemBandPrefetchStages = integer_divide_ceil(
        MinMemInFlyBytes / WgpPerCU,
        (MPerBlock * sizeof(ADataType) + NPerBlock * sizeof(BDataType)) * KPerBlock);
    static constexpr index_t PrefetchStages =
        FullMemBandPrefetchStages >= 2
            ? FullMemBandPrefetchStages <= 8 ? FullMemBandPrefetchStages : 8
            : 2;

    static constexpr index_t LocalPrefillStages = 1;
    static constexpr index_t GlobalBufferNum    = PrefetchStages;

    CK_TILE_HOST static constexpr bool BlockHasHotloop(index_t num_loop)
    {
        return num_loop > PrefetchStages;
    }

    CK_TILE_HOST static constexpr TailNumber GetBlockLoopTailNum(index_t num_loop)
    {
        if(num_loop % PrefetchStages == 1)
        {
            return TailNumber::One;
        }
        else if(num_loop % PrefetchStages == 2)
        {
            return TailNumber::Two;
        }
        else if(num_loop % PrefetchStages == 3)
        {
            return TailNumber::Three;
        }
        else if(num_loop % PrefetchStages == 4)
        {
            return TailNumber::Four;
        }
        else if(num_loop % PrefetchStages == 5)
        {
            return TailNumber::Five;
        }
        else if(num_loop % PrefetchStages == 6)
        {
            return TailNumber::Six;
        }
        else if(num_loop % PrefetchStages == 7)
        {
            return TailNumber::Seven;
        }
        else
        {
            return TailNumber::Full;
        }
    }
};

// Maximum Global Memory throughput pipeline with >=32KB data in fly
// GlobalPrefetchStages: >=2
// LocalPreFillStages: 1
// LocalPreFetchStages: 0
// LocalSharedMemoryBuffer: 1
template <typename Problem, typename Policy = GemmPipelineAGmemBGmemCRegV1DefaultPolicy>
struct GemmPipelineAgBgCrMem : public BaseGemmPipelineAgBgCrMem<Problem>
{
    using Base             = BaseGemmPipelineAgBgCrMem<Problem>;
    using PipelineImplBase = GemmPipelineAgBgCrImplBase<Problem, Policy>;

    using ADataType      = remove_cvref_t<typename Problem::ADataType>;
    using BDataType      = remove_cvref_t<typename Problem::BDataType>;
    using CDataType      = remove_cvref_t<typename Problem::CDataType>;
    using BlockGemmShape = remove_cvref_t<typename Problem::BlockGemmShape>;

    using ALayout = remove_cvref_t<typename Problem::ALayout>;
    using BLayout = remove_cvref_t<typename Problem::BLayout>;
    using CLayout = remove_cvref_t<typename Problem::CLayout>;

    using BlockGemm = remove_cvref_t<decltype(Policy::template GetBlockGemm<Problem>())>;

    using I0 = number<0>;
    using I1 = number<1>;
    using I2 = number<2>;

    static constexpr index_t MPerBlock = BlockGemmShape::kM;
    static constexpr index_t NPerBlock = BlockGemmShape::kN;
    static constexpr index_t KPerBlock = BlockGemmShape::kK;

    static constexpr index_t GetVectorSizeA() { return Policy::template GetVectorSizeA<Problem>(); }
    static constexpr index_t GetVectorSizeB() { return Policy::template GetVectorSizeB<Problem>(); }
    static constexpr index_t GetVectorSizeC() { return Policy::template GetVectorSizeC<Problem>(); }

    static constexpr bool kPadM = Problem::kPadM;
    static constexpr bool kPadN = Problem::kPadN;
    static constexpr bool kPadK = Problem::kPadK;

    // Where is the right place for HasHotLoop and TailNum ???
    static constexpr bool HasHotLoop = Problem::HasHotLoop;
    static constexpr auto TailNum    = Problem::TailNum;
    static constexpr auto Scheduler  = Problem::Scheduler;

    using Base::PrefetchStages;

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return Policy::template GetSmemSize<Problem>();
    }

    template <GemmPipelineScheduler Scheduler>
    struct PipelineImpl : public PipelineImplBase
    {
    };

    template <>
    struct PipelineImpl<GemmPipelineScheduler::Intrawave> : public PipelineImplBase
    {
        using Base = PipelineImplBase;

        template <bool HasHotLoop,
                  TailNumber TailNum,
                  typename ADramBlockWindowTmp,
                  typename BDramBlockWindowTmp,
                  typename AElementFunction,
                  typename BElementFunction>
        CK_TILE_DEVICE auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                       const AElementFunction& a_element_func,
                                       const BDramBlockWindowTmp& b_dram_block_window_tmp,
                                       const BElementFunction& b_element_func,
                                       index_t num_loop,
                                       void* p_smem) const
        {
            static_assert(
                std::is_same_v<ADataType, remove_cvref_t<typename ADramBlockWindowTmp::DataType>> &&
                    std::is_same_v<BDataType,
                                   remove_cvref_t<typename BDramBlockWindowTmp::DataType>>,
                "A/B Dram block window should have the same data type as appropriate "
                "([A|B]DataType) defined in Problem definition!");

            static_assert(MPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[I0{}] &&
                              NPerBlock == BDramBlockWindowTmp{}.get_window_lengths()[I0{}] &&
                              KPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[I1{}],
                          "A/B block window appropriate sizes must be equal to MPerBlock/NPerblock"
                          " or KPerBlock!");

            // ------------------------------------------------------------------------------------
            // Definitions of all needed tiles

            // A/B tiles in LDS
            // With c++20 could simplify to below line.
            // Currently get error: captured structured bindings are a C++20 extension
            // auto&& [a_lds_block, b_lds_block] = Base::GetABLdsTensorViews(p_smem);
            auto ab_lds_blocks = Base::GetABLdsTensorViews(p_smem);
            auto& a_lds_block  = ab_lds_blocks.at(I0{});
            auto& b_lds_block  = ab_lds_blocks.at(I1{});

            // A DRAM tile window for load
            // A LDS tile window for store
            // A LDS tile for block GEMM
            auto a_windows           = Base::GetAWindows(a_dram_block_window_tmp, a_lds_block);
            auto& a_copy_dram_window = a_windows.at(I0{});
            auto& a_copy_lds_window  = a_windows.at(I1{});
            auto& a_lds_gemm_window  = a_windows.at(I2{});

            // B DRAM tile window for load
            // B LDS tile window for store
            // B LDS tile for block GEMM
            auto b_windows           = Base::GetBWindows(b_dram_block_window_tmp, b_lds_block);
            auto& b_copy_dram_window = b_windows.at(I0{});
            auto& b_copy_lds_window  = b_windows.at(I1{});
            auto& b_lds_gemm_window  = b_windows.at(I2{});

            // Block GEMM
            auto block_gemm   = BlockGemm();
            auto c_block_tile = block_gemm.MakeCBlockTile();

            using ABlockTileDistr = decltype(a_copy_dram_window.get_tile_distribution());
            using BBlockTileDistr = decltype(b_copy_dram_window.get_tile_distribution());

            using ABlockTile =
                decltype(make_static_distributed_tensor<ADataType>(ABlockTileDistr{}));
            using BBlockTile =
                decltype(make_static_distributed_tensor<BDataType>(BBlockTileDistr{}));

            tuple_array<ABlockTile, PrefetchStages> a_block_tiles;
            tuple_array<BBlockTile, PrefetchStages> b_block_tiles;

            // -----------------------------------------------------------------------------------------
            // Gemm pipeline start

            // prefetch
            // global read 0
            Base::GlobalPrefetch(a_block_tiles.get(I0{}), a_copy_dram_window);
            Base::GlobalPrefetch(b_block_tiles.get(I0{}), b_copy_dram_window);

            // initialize C
            tile_elementwise_inout([](auto& c) { c = 0; }, c_block_tile);

            // LDS write 0
            Base::LocalPrefill(a_copy_lds_window, a_block_tiles.get(I0{}), a_element_func);
            Base::LocalPrefill(b_copy_lds_window, b_block_tiles.get(I0{}), b_element_func);

            // Global prefetch [1, PrefetchStages]
            static_for<1, PrefetchStages, 1>{}([&](auto prefetch_idx) {
                Base::GlobalPrefetch(a_block_tiles.get(number<prefetch_idx>{}), a_copy_dram_window);
                Base::GlobalPrefetch(b_block_tiles.get(number<prefetch_idx>{}), b_copy_dram_window);
            });

            // main body
            if constexpr(HasHotLoop)
            {
                index_t i = 0;
                do
                {
                    static_for<0, PrefetchStages, 1>{}([&](auto prefetch_idx) {
                        block_sync_lds();
                        block_gemm.LocalPrefetch(a_lds_gemm_window, b_lds_gemm_window);
                        block_gemm(c_block_tile, a_lds_gemm_window, b_lds_gemm_window);

                        block_sync_lds();

                        Base::LocalPrefill(
                            a_copy_lds_window,
                            a_block_tiles.get(number<(prefetch_idx + 1) % PrefetchStages>{}),
                            a_element_func);
                        Base::LocalPrefill(
                            b_copy_lds_window,
                            b_block_tiles.get(number<(prefetch_idx + 1) % PrefetchStages>{}),
                            b_element_func);

                        Base::GlobalPrefetch(a_block_tiles.get(number<prefetch_idx>{}),
                                             a_copy_dram_window);
                        Base::GlobalPrefetch(b_block_tiles.get(number<prefetch_idx>{}),
                                             b_copy_dram_window);
                    });

                    i += PrefetchStages;
                } while(i < (num_loop - PrefetchStages));
            }

            auto HotLoopTail = [&](auto tail_num) {
                static_for<1, tail_num, 1>{}([&](auto prefetch_idx) {
                    block_sync_lds();

                    block_gemm.LocalPrefetch(a_lds_gemm_window, b_lds_gemm_window);
                    block_gemm(c_block_tile, a_lds_gemm_window, b_lds_gemm_window);

                    block_sync_lds();

                    Base::LocalPrefill(a_copy_lds_window,
                                       a_block_tiles.get(number<prefetch_idx>{}),
                                       a_element_func);
                    Base::LocalPrefill(b_copy_lds_window,
                                       b_block_tiles.get(number<prefetch_idx>{}),
                                       b_element_func);
                });

                block_sync_lds();
                block_gemm.LocalPrefetch(a_lds_gemm_window, b_lds_gemm_window);
                block_gemm(c_block_tile, a_lds_gemm_window, b_lds_gemm_window);
            };

            if constexpr(TailNum == TailNumber::One)
            {
                block_sync_lds();
                block_gemm.LocalPrefetch(a_lds_gemm_window, b_lds_gemm_window);
                block_gemm(c_block_tile, a_lds_gemm_window, b_lds_gemm_window);
            }
            else if constexpr(TailNum == TailNumber::Two)
            {
                HotLoopTail(number<2>{});
            }
            else if constexpr(TailNum == TailNumber::Three)
            {
                HotLoopTail(number<3>{});
            }
            else if constexpr(TailNum == TailNumber::Four)
            {
                HotLoopTail(number<4>{});
            }
            else if constexpr(TailNum == TailNumber::Five)
            {
                HotLoopTail(number<5>{});
            }
            else if constexpr(TailNum == TailNumber::Six)
            {
                HotLoopTail(number<6>{});
            }
            else if constexpr(TailNum == TailNumber::Seven)
            {
                HotLoopTail(number<7>{});
            }
            else if constexpr(TailNum == TailNumber::Full)
            {
                HotLoopTail(number<PrefetchStages>{});
            }

            return c_block_tile;
        }
    };

    template <>
    struct PipelineImpl<GemmPipelineScheduler::Interwave> : public PipelineImplBase
    {
        using Base = PipelineImplBase;

        template <bool HasHotLoop,
                  TailNumber TailNum,
                  typename ADramBlockWindowTmp,
                  typename BDramBlockWindowTmp,
                  typename AElementFunction,
                  typename BElementFunction>
        CK_TILE_DEVICE auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                       const AElementFunction& a_element_func,
                                       const BDramBlockWindowTmp& b_dram_block_window_tmp,
                                       const BElementFunction& b_element_func,
                                       index_t num_loop,
                                       void* p_smem) const
        {
            static_assert(
                std::is_same_v<ADataType, remove_cvref_t<typename ADramBlockWindowTmp::DataType>> &&
                    std::is_same_v<BDataType,
                                   remove_cvref_t<typename BDramBlockWindowTmp::DataType>>,
                "A/B Dram block window should have the same data type as appropriate "
                "([A|B]DataType) defined in Problem definition!");

            static_assert(MPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[I0{}] &&
                              NPerBlock == BDramBlockWindowTmp{}.get_window_lengths()[I0{}] &&
                              KPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[I1{}],
                          "A/B block window appropriate sizes must be equal to MPerBlock/NPerblock"
                          " or KPerBlock!");

            // ------------------------------------------------------------------------------------
            // Definitions of all needed tiles

            // A/B tiles in LDS
            // With c++20 could simplify to below line.
            // Currently get error: captured structured bindings are a C++20 extension
            // auto&& [a_lds_block, b_lds_block] = Base::GetABLdsTensorViews(p_smem);
            auto ab_lds_blocks = Base::GetABLdsTensorViews(p_smem);
            auto& a_lds_block  = ab_lds_blocks.at(I0{});
            auto& b_lds_block  = ab_lds_blocks.at(I1{});

            // A DRAM tile window for load
            // A LDS tile window for store
            // A LDS tile for block GEMM
            auto a_windows           = Base::GetAWindows(a_dram_block_window_tmp, a_lds_block);
            auto& a_copy_dram_window = a_windows.at(I0{});
            auto& a_copy_lds_window  = a_windows.at(I1{});
            auto& a_lds_gemm_window  = a_windows.at(I2{});

            // B DRAM tile window for load
            // B LDS tile window for store
            // B LDS tile for block GEMM
            auto b_windows           = Base::GetBWindows(b_dram_block_window_tmp, b_lds_block);
            auto& b_copy_dram_window = b_windows.at(I0{});
            auto& b_copy_lds_window  = b_windows.at(I1{});
            auto& b_lds_gemm_window  = b_windows.at(I2{});

            // Block GEMM
            auto block_gemm   = BlockGemm();
            auto c_block_tile = block_gemm.MakeCBlockTile();

            using ABlockTileDistr = decltype(a_copy_dram_window.get_tile_distribution());
            using BBlockTileDistr = decltype(b_copy_dram_window.get_tile_distribution());

            using ABlockTile =
                decltype(make_static_distributed_tensor<ADataType>(ABlockTileDistr{}));
            using BBlockTile =
                decltype(make_static_distributed_tensor<BDataType>(BBlockTileDistr{}));

            tuple_array<ABlockTile, PrefetchStages> a_block_tiles;
            tuple_array<BBlockTile, PrefetchStages> b_block_tiles;

            // -----------------------------------------------------------------------------------------
            // Gemm pipeline start

            // prefetch
            // global read 0
            Base::GlobalPrefetch(a_block_tiles.get(I0{}), a_copy_dram_window);
            Base::GlobalPrefetch(b_block_tiles.get(I0{}), b_copy_dram_window);

            // initialize C
            tile_elementwise_inout([](auto& c) { c = 0; }, c_block_tile);

            // LDS write 0
            Base::LocalPrefill(a_copy_lds_window, a_block_tiles.get(I0{}), a_element_func);
            Base::LocalPrefill(b_copy_lds_window, b_block_tiles.get(I0{}), b_element_func);

            // Global prefetch [1, PrefetchStages]
            static_for<1, PrefetchStages, 1>{}([&](auto prefetch_idx) {
                Base::GlobalPrefetch(a_block_tiles.get(number<prefetch_idx>{}), a_copy_dram_window);
                Base::GlobalPrefetch(b_block_tiles.get(number<prefetch_idx>{}), b_copy_dram_window);
            });

            // main body
            if constexpr(HasHotLoop)
            {
                index_t i = 0;
                do
                {
                    static_for<0, PrefetchStages, 1>{}([&](auto prefetch_idx) {
                        block_sync_lds();
                        block_gemm(c_block_tile, a_lds_gemm_window, b_lds_gemm_window);
                        // no second block_sync_lds because it's interwave

                        Base::LocalPrefill(
                            a_copy_lds_window,
                            a_block_tiles.get(number<(prefetch_idx + 1) % PrefetchStages>{}),
                            a_element_func);
                        Base::LocalPrefill(
                            b_copy_lds_window,
                            b_block_tiles.get(number<(prefetch_idx + 1) % PrefetchStages>{}),
                            b_element_func);

                        Base::GlobalPrefetch(a_block_tiles.get(number<prefetch_idx>{}),
                                             a_copy_dram_window);
                        Base::GlobalPrefetch(b_block_tiles.get(number<prefetch_idx>{}),
                                             b_copy_dram_window);
                    });

                    i += PrefetchStages;
                } while(i < (num_loop - PrefetchStages));
            }

            auto HotLoopTail = [&](auto tail_num) {
                static_for<1, tail_num, 1>{}([&](auto prefetch_idx) {
                    block_sync_lds();
                    block_gemm(c_block_tile, a_lds_gemm_window, b_lds_gemm_window);
                    // no second block_sync_lds because it's interwave

                    Base::LocalPrefill(a_copy_lds_window,
                                       a_block_tiles.get(number<prefetch_idx>{}),
                                       a_element_func);
                    Base::LocalPrefill(b_copy_lds_window,
                                       b_block_tiles.get(number<prefetch_idx>{}),
                                       b_element_func);
                });

                block_sync_lds();
                block_gemm(c_block_tile, a_lds_gemm_window, b_lds_gemm_window);
            };

            if constexpr(TailNum == TailNumber::One)
            {
                block_sync_lds();
                block_gemm(c_block_tile, a_lds_gemm_window, b_lds_gemm_window);
            }
            else if constexpr(TailNum == TailNumber::Two)
            {
                HotLoopTail(number<2>{});
            }
            else if constexpr(TailNum == TailNumber::Three)
            {
                HotLoopTail(number<3>{});
            }
            else if constexpr(TailNum == TailNumber::Four)
            {
                HotLoopTail(number<4>{});
            }
            else if constexpr(TailNum == TailNumber::Five)
            {
                HotLoopTail(number<5>{});
            }
            else if constexpr(TailNum == TailNumber::Six)
            {
                HotLoopTail(number<6>{});
            }
            else if constexpr(TailNum == TailNumber::Seven)
            {
                HotLoopTail(number<7>{});
            }
            else if constexpr(TailNum == TailNumber::Full)
            {
                HotLoopTail(number<PrefetchStages>{});
            }

            return c_block_tile;
        }
    };

    template <typename ADramBlockWindowTmp,
              typename BDramBlockWindowTmp,
              typename AElementFunction,
              typename BElementFunction>
    CK_TILE_DEVICE auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                   const AElementFunction& a_element_func,
                                   const BDramBlockWindowTmp& b_dram_block_window_tmp,
                                   const BElementFunction& b_element_func,
                                   index_t num_loop,
                                   void* p_smem) const
    {
        return PipelineImpl<Scheduler>{}.template operator()<HasHotLoop, TailNum>(
            a_dram_block_window_tmp,
            a_element_func,
            b_dram_block_window_tmp,
            b_element_func,
            num_loop,
            p_smem);
    }

    template <typename ADramBlockWindowTmp, typename BDramBlockWindowTmp>
    CK_TILE_DEVICE auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                   const BDramBlockWindowTmp& b_dram_block_window_tmp,
                                   index_t num_loop,
                                   void* p_smem) const
    {
        return PipelineImpl<Scheduler>{}.template operator()<HasHotLoop, TailNum>(
            a_dram_block_window_tmp,
            [](const ADataType& a) { return a; },
            b_dram_block_window_tmp,
            [](const BDataType& b) { return b; },
            num_loop,
            p_smem);
    }
};

} // namespace ck_tile
