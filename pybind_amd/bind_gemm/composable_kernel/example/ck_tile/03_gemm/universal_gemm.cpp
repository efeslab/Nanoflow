// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#include <hip/hip_runtime.h>

#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>

#include "ck_tile/host.hpp"
#include "gemm_basic.hpp"

template <typename ALayout, typename BLayout, typename CLayout>
float gemm_calc(const ck_tile::GemmHostArgs& args, const ck_tile::stream_config& s)
{
#if(CK_TILE_PIPELINE_DEFAULT == CK_TILE_PIPELINE_MEMORY)
    // Memory friendly for Interwave scheduler
    constexpr ck_tile::index_t M_Tile = 128;
    constexpr ck_tile::index_t N_Tile = 32;
    constexpr ck_tile::index_t K_Tile = 64;

    constexpr ck_tile::index_t M_Warp = 4;
    constexpr ck_tile::index_t N_Warp = 1;
    constexpr ck_tile::index_t K_Warp = 1;

    constexpr ck_tile::index_t M_Warp_Tile = 32;
    constexpr ck_tile::index_t N_Warp_Tile = 32;
    constexpr ck_tile::index_t K_Warp_Tile = 8;
#endif
#if(CK_TILE_PIPELINE_DEFAULT == CK_TILE_PIPELINE_COMPUTE)
    // Compute friendly for Intrawave scheduler
    constexpr ck_tile::index_t M_Tile = 256;
    constexpr ck_tile::index_t N_Tile = 256;
    constexpr ck_tile::index_t K_Tile = 32;

    constexpr ck_tile::index_t M_Warp = 2;
    constexpr ck_tile::index_t N_Warp = 2;
    constexpr ck_tile::index_t K_Warp = 1;

    constexpr ck_tile::index_t M_Warp_Tile = 32;
    constexpr ck_tile::index_t N_Warp_Tile = 32;
    constexpr ck_tile::index_t K_Warp_Tile = 16;
#endif

    constexpr bool kPadM = false;
    constexpr bool kPadN = false;
    constexpr bool kPadK = false;

    constexpr bool TransposeC = false;

    constexpr int kBlockPerCu                         = 1;
    constexpr ck_tile::index_t TileParitionerGroupNum = 8;
    constexpr ck_tile::index_t TileParitionerM01      = 4;

    // ===============================================

    using GemmShape =
        ck_tile::TileGemmShape<ck_tile::sequence<M_Tile, N_Tile, K_Tile>,
                               ck_tile::sequence<M_Warp, N_Warp, K_Warp>,
                               ck_tile::sequence<M_Warp_Tile, N_Warp_Tile, K_Warp_Tile>>;
    using TilePartitioner = ck_tile::
        GemmSpatiallyLocalTilePartitioner<GemmShape, TileParitionerGroupNum, TileParitionerM01>;

    using Traits = ck_tile::TileGemmTraits<kPadM, kPadN, kPadK, ALayout, BLayout, CLayout>;
    using GemmUniversalTraits = ck_tile::
        TileGemmUniversalTraits<kPadM, kPadN, kPadK, ALayout, BLayout, CLayout, TransposeC>;
    using GemmPipelineProblem =
        ck_tile::GemmPipelineProblem<ADataType, BDataType, AccDataType, GemmShape, Traits>;

    using BaseGemmPipeline = UNIVERSAL_GEMM_PIPELINE<GemmPipelineProblem>;

    const ck_tile::index_t k_grain     = args.k_batch * K_Tile;
    const ck_tile::index_t K_split     = (args.K + k_grain - 1) / k_grain * K_Tile;
    const ck_tile::index_t num_loop    = TilePartitioner::GetLoopNum(K_split);
    const bool has_hot_loop            = BaseGemmPipeline::BlockHasHotloop(num_loop);
    const ck_tile::TailNumber tail_num = BaseGemmPipeline::GetBlockLoopTailNum(num_loop);

    float ave_time{0};

    const auto Run = [&](const auto has_hot_loop_, const auto tail_number_) {
        constexpr bool has_hot_loop_v = has_hot_loop_.value;
        constexpr auto tail_number_v  = tail_number_.value;
        constexpr auto scheduler      = GEMM_PIPELINE_SCHEDULER;

        using UniversalGemmProblem = ck_tile::UniversalGemmPipelineProblem<ADataType,
                                                                           BDataType,
                                                                           AccDataType,
                                                                           GemmShape,
                                                                           GemmUniversalTraits,
                                                                           scheduler,
                                                                           has_hot_loop_v,
                                                                           tail_number_v>;

        using GemmPipeline =
            GEMM_PIPELINE<UniversalGemmProblem, ck_tile::UniversalGemmPipelineAgBgCrPolicy>;
        using GemmEpilogue = ck_tile::CShuffleEpilogue<
            ck_tile::CShuffleEpilogueProblem<AccDataType,
                                             CDataType,
                                             CLayout,
                                             GemmPipelineProblem::kBlockSize,
                                             TilePartitioner::MPerBlock,
                                             TilePartitioner::NPerBlock,
                                             M_Warp,
                                             N_Warp,
                                             M_Warp_Tile,
                                             N_Warp_Tile,
                                             K_Warp_Tile,
                                             UniversalGemmProblem::TransposeC>>;
        using Kernel = ck_tile::GemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue>;
        auto kargs   = Kernel::MakeKernelArgs(args);

        const dim3 grids      = Kernel::GridSize(args.M, args.N, args.k_batch);
        constexpr dim3 blocks = Kernel::BlockSize();

        if(!Kernel::IsSupportedArgument(kargs))
        {
            throw std::runtime_error("Wrong! Arguments not supported! Skipping gemm!\n");
        }

        if(s.log_level_ > 0)
        {
            std::cout << "Launching kernel with args:"
                      << " grid: {" << grids.x << ", " << grids.y << ", " << grids.z << "}"
                      << ", blocks: {" << blocks.x << ", " << blocks.y << ", " << blocks.z << "}"
                      << std::endl;
        }

        ave_time = ck_tile::launch_kernel(
            s, ck_tile::make_kernel<blocks.x, kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));
        return ave_time;
    };

    if(has_hot_loop)
    {
#if(CK_TILE_PIPELINE_DEFAULT == CK_TILE_PIPELINE_COMPUTE)
        if(tail_num == ck_tile::TailNumber::Full)
        {
            Run(ck_tile::bool_constant<true>{},
                ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Full>{});
        }
        else
        {
            std::ostringstream err;
            err << "For compute pipeline tail number should always be Full, but have \"" << tail_num
                << "\" which is not supported! PrefetchStages: " << BaseGemmPipeline::PrefetchStages
                << "\n File: " << __FILE__ << ":" << __LINE__ << ", in function: " << __func__;
            throw std::runtime_error(err.str());
        }
#elif(CK_TILE_PIPELINE_DEFAULT == CK_TILE_PIPELINE_MEMORY)
        // Tail pipeline One to Seven
        if(tail_num == ck_tile::TailNumber::One)
        {
            Run(ck_tile::bool_constant<true>{},
                ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::One>{});
        }
        else if(tail_num == ck_tile::TailNumber::Full)
        {
            Run(ck_tile::bool_constant<true>{},
                ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Full>{});
        }

        if constexpr(BaseGemmPipeline::PrefetchStages > 2)
        {
            if(tail_num == ck_tile::TailNumber::Two)
            {
                Run(ck_tile::bool_constant<true>{},
                    ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Two>{});
            }
        }
        if constexpr(BaseGemmPipeline::PrefetchStages > 3)
        {
            if(tail_num == ck_tile::TailNumber::Three)
            {
                Run(ck_tile::bool_constant<true>{},
                    ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Three>{});
            }
        }
        if constexpr(BaseGemmPipeline::PrefetchStages > 4)
        {
            if(tail_num == ck_tile::TailNumber::Four)
            {
                Run(ck_tile::bool_constant<true>{},
                    ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Four>{});
            }
        }
        if constexpr(BaseGemmPipeline::PrefetchStages > 5)
        {
            if(tail_num == ck_tile::TailNumber::Five)
            {
                Run(ck_tile::bool_constant<true>{},
                    ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Five>{});
            }
        }
        if constexpr(BaseGemmPipeline::PrefetchStages > 6)
        {
            if(tail_num == ck_tile::TailNumber::Six)
            {
                Run(ck_tile::bool_constant<true>{},
                    ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Six>{});
            }
        }
        if constexpr(BaseGemmPipeline::PrefetchStages > 7)
        {
            if(tail_num == ck_tile::TailNumber::Seven)
            {
                Run(ck_tile::bool_constant<true>{},
                    ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Seven>{});
            }
        }
#endif
    }
    else
    {
        // Tail number always Full - #PrefetchStages
        if(tail_num == ck_tile::TailNumber::Full)
        {
            Run(ck_tile::bool_constant<false>{},
                ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Full>{});
        }
        else
        {
            std::ostringstream err;
            err << "When there's no hot loop, this tail number \"" << tail_num
                << "\" is not supported! PrefetchStages: " << BaseGemmPipeline::PrefetchStages
                << "\n File: " << __FILE__ << ":" << __LINE__ << ", in function: " << __func__;
            throw std::runtime_error(err.str());
        }
    }

    return ave_time;
}

#include "run_gemm_example.inc"

int run_gemm_example(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return -1;

    using Row = ck_tile::tensor_layout::gemm::RowMajor;
    using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

    std::string a_layout = arg_parser.get_str("a_layout");
    std::string b_layout = arg_parser.get_str("b_layout");

    if(a_layout == "R" && b_layout == "R")
    {
        return run_gemm_example_with_layouts(argc, argv, Row{}, Row{}, Row{});
    }
    else if(a_layout == "R" && b_layout == "C")
    {
        return run_gemm_example_with_layouts(argc, argv, Row{}, Col{}, Row{});
    }
    else if(a_layout == "C" && b_layout == "C")
    {
        return run_gemm_example_with_layouts(argc, argv, Col{}, Col{}, Row{});
    }
    else if(a_layout == "C" && b_layout == "R")
    {
        return run_gemm_example_with_layouts(argc, argv, Col{}, Row{}, Row{});
    }
    else
    {
        throw std::runtime_error("Unsupported data layout configuration for A,B and C tensors!");
    }
}

int main(int argc, char* argv[]) { return !run_gemm_example(argc, argv); }
