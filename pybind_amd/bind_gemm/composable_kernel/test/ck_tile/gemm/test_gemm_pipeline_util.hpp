// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include <sstream>
#include <gtest/gtest.h>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"

enum struct GemmPipelineType
{
    Mem,
    Comp
};

template <typename Tuple>
class TestCkTileGemmPipeline : public ::testing::Test
{
    protected:
    using ALayout                      = std::tuple_element_t<0, Tuple>;
    using BLayout                      = std::tuple_element_t<1, Tuple>;
    using CLayout                      = std::tuple_element_t<2, Tuple>;
    using ADataType                    = std::tuple_element_t<3, Tuple>;
    using BDataType                    = std::tuple_element_t<4, Tuple>;
    using AccDataType                  = std::tuple_element_t<5, Tuple>;
    using CDataType                    = std::tuple_element_t<6, Tuple>;
    static constexpr auto Scheduler    = std::tuple_element_t<7, Tuple>::value;
    static constexpr auto PipelineType = std::tuple_element_t<8, Tuple>::value;
    // TODO: expose tile size through test t-param ?

    template <bool PadM, bool PadN, bool PadK>
    void invoke_gemm(const ck_tile::GemmHostArgs& args, const ck_tile::stream_config& s)
    {
        // TODO: This should be parameterized in tests
        constexpr ck_tile::index_t M_Tile = 128;
        constexpr ck_tile::index_t N_Tile = 128;
        constexpr ck_tile::index_t K_Tile = 32;

        constexpr ck_tile::index_t M_Warp = 2;
        constexpr ck_tile::index_t N_Warp = 2;
        constexpr ck_tile::index_t K_Warp = 1;

        constexpr ck_tile::index_t M_Warp_Tile = 32;
        constexpr ck_tile::index_t N_Warp_Tile = 32;
        constexpr ck_tile::index_t K_Warp_Tile = 8;

        constexpr bool kPadM = PadM;
        constexpr bool kPadN = PadN;
        constexpr bool kPadK = PadK;

        // TODO: For now - but this should also be a test parameter
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

        using BaseGemmPipeline =
            std::conditional_t<PipelineType == GemmPipelineType::Mem,
                               ck_tile::BaseGemmPipelineAgBgCrMem<GemmPipelineProblem>,
                               ck_tile::BaseGemmPipelineAgBgCrCompV3<GemmPipelineProblem>>;

        const ck_tile::index_t k_grain     = args.k_batch * K_Tile;
        const ck_tile::index_t K_split     = (args.K + k_grain - 1) / k_grain * K_Tile;
        const ck_tile::index_t num_loop    = TilePartitioner::GetLoopNum(K_split);
        const bool has_hot_loop            = BaseGemmPipeline::BlockHasHotloop(num_loop);
        const ck_tile::TailNumber tail_num = BaseGemmPipeline::GetBlockLoopTailNum(num_loop);

        const auto Run = [&](const auto has_hot_loop_, const auto tail_number_) {
            constexpr bool has_hot_loop_v = has_hot_loop_.value;
            constexpr auto tail_number_v  = tail_number_.value;

            using UniversalGemmProblem = ck_tile::UniversalGemmPipelineProblem<ADataType,
                                                                               BDataType,
                                                                               AccDataType,
                                                                               GemmShape,
                                                                               GemmUniversalTraits,
                                                                               Scheduler,
                                                                               has_hot_loop_v,
                                                                               tail_number_v>;

            using GemmPipeline = std::conditional_t<
                PipelineType == GemmPipelineType::Mem,
                ck_tile::GemmPipelineAgBgCrMem<UniversalGemmProblem,
                                               ck_tile::UniversalGemmPipelineAgBgCrPolicy>,
                ck_tile::GemmPipelineAgBgCrCompV3<UniversalGemmProblem,
                                                  ck_tile::UniversalGemmPipelineAgBgCrPolicy>>;

            using GemmEpilogue = ck_tile::CShuffleEpilogue<
                ck_tile::CShuffleEpilogueProblem<AccDataType,
                                                 CDataType,
                                                 CLayout,
                                                 GemmPipeline::BlockSize,
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
                          << ", blocks: {" << blocks.x << ", " << blocks.y << ", " << blocks.z
                          << "}" << std::endl;
            }

            ck_tile::launch_kernel(
                s, ck_tile::make_kernel<blocks.x, kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));
        };

        if(has_hot_loop)
        {
            if constexpr(PipelineType == GemmPipelineType::Comp)
            {
                if(tail_num == ck_tile::TailNumber::Full)
                {
                    Run(ck_tile::bool_constant<true>{},
                        ck_tile::integral_constant<ck_tile::TailNumber,
                                                   ck_tile::TailNumber::Full>{});
                }
                else
                {
                    std::ostringstream err;
                    err << "For compute pipeline tail number should always be Full, but have \""
                        << tail_num << "\" which is not supported! PrefetchStages: "
                        << BaseGemmPipeline::PrefetchStages << "\n File: " << __FILE__ << ":"
                        << __LINE__ << ", in function: " << __func__;
                    throw std::runtime_error(err.str());
                }
            }

            if constexpr(PipelineType == GemmPipelineType::Mem)
            {
                // Tail pipeline One to Seven
                if(tail_num == ck_tile::TailNumber::One)
                {
                    Run(ck_tile::bool_constant<true>{},
                        ck_tile::integral_constant<ck_tile::TailNumber,
                                                   ck_tile::TailNumber::One>{});
                }
                else if(tail_num == ck_tile::TailNumber::Full)
                {
                    Run(ck_tile::bool_constant<true>{},
                        ck_tile::integral_constant<ck_tile::TailNumber,
                                                   ck_tile::TailNumber::Full>{});
                }

                if constexpr(BaseGemmPipeline::PrefetchStages > 2)
                {
                    if(tail_num == ck_tile::TailNumber::Two)
                    {
                        Run(ck_tile::bool_constant<true>{},
                            ck_tile::integral_constant<ck_tile::TailNumber,
                                                       ck_tile::TailNumber::Two>{});
                    }
                }
                if constexpr(BaseGemmPipeline::PrefetchStages > 3)
                {
                    if(tail_num == ck_tile::TailNumber::Three)
                    {
                        Run(ck_tile::bool_constant<true>{},
                            ck_tile::integral_constant<ck_tile::TailNumber,
                                                       ck_tile::TailNumber::Three>{});
                    }
                }
                if constexpr(BaseGemmPipeline::PrefetchStages > 4)
                {
                    if(tail_num == ck_tile::TailNumber::Four)
                    {
                        Run(ck_tile::bool_constant<true>{},
                            ck_tile::integral_constant<ck_tile::TailNumber,
                                                       ck_tile::TailNumber::Four>{});
                    }
                }
                if constexpr(BaseGemmPipeline::PrefetchStages > 5)
                {
                    if(tail_num == ck_tile::TailNumber::Five)
                    {
                        Run(ck_tile::bool_constant<true>{},
                            ck_tile::integral_constant<ck_tile::TailNumber,
                                                       ck_tile::TailNumber::Five>{});
                    }
                }
                if constexpr(BaseGemmPipeline::PrefetchStages > 6)
                {
                    if(tail_num == ck_tile::TailNumber::Six)
                    {
                        Run(ck_tile::bool_constant<true>{},
                            ck_tile::integral_constant<ck_tile::TailNumber,
                                                       ck_tile::TailNumber::Six>{});
                    }
                }
                if constexpr(BaseGemmPipeline::PrefetchStages > 7)
                {
                    if(tail_num == ck_tile::TailNumber::Seven)
                    {
                        Run(ck_tile::bool_constant<true>{},
                            ck_tile::integral_constant<ck_tile::TailNumber,
                                                       ck_tile::TailNumber::Seven>{});
                    }
                }
            }
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
                    << "\" is not supported! " << __FILE__ << ":" << __LINE__
                    << ", in function: " << __func__;
                throw std::runtime_error(err.str());
            }
        }
    }

    public:
    std::vector<int> k_batches_;

    void SetUp() override { k_batches_ = {1, 2}; }

    template <bool PadM = true, bool PadN = true, bool PadK = true>
    void Run(const int M,
             const int N,
             const int K,
             const int StrideA = 0,
             const int StrideB = 0,
             const int StrideC = 0)
    {
        for(auto kb : k_batches_)
        {
            RunSingle<PadM, PadN, PadK>(M, N, K, StrideA, StrideB, StrideC, kb);
        }
    }

    template <bool PadM, bool PadN, bool PadK>
    void RunSingle(const int M,
                   const int N,
                   const int K,
                   const int StrideA,
                   const int StrideB,
                   const int StrideC,
                   int kbatch = 1)
    {
        using namespace ck_tile::literals;

        auto f_host_tensor_descriptor = [](std::size_t row,
                                           std::size_t col,
                                           std::size_t stride,
                                           auto layout) {
            if constexpr(std::is_same_v<decltype(layout), ck_tile::tensor_layout::gemm::RowMajor>)
            {
                return ck_tile::HostTensorDescriptor({row, col}, {stride, 1_uz});
            }
            else
            {
                return ck_tile::HostTensorDescriptor({row, col}, {1_uz, stride});
            }
        };

        auto f_get_default_stride =
            [](std::size_t row, std::size_t col, std::size_t stride, auto layout) {
                if(stride == 0)
                {
                    // give a chance if stride is zero, return a default packed stride
                    if constexpr(std::is_same_v<decltype(layout),
                                                ck_tile::tensor_layout::gemm::RowMajor>)
                    {
                        return col;
                    }
                    else
                    {
                        return row;
                    }
                }
                else
                    return stride;
            };

        std::size_t stride_A = f_get_default_stride(M, K, StrideA, ALayout{});
        std::size_t stride_B = f_get_default_stride(K, N, StrideB, BLayout{});
        std::size_t stride_C = f_get_default_stride(M, N, StrideC, CLayout{});

        ck_tile::HostTensor<ADataType> a_m_k(f_host_tensor_descriptor(M, K, stride_A, ALayout{}));
        ck_tile::HostTensor<BDataType> b_k_n(f_host_tensor_descriptor(K, N, stride_B, BLayout{}));
        ck_tile::HostTensor<CDataType> c_m_n_dev_result(
            f_host_tensor_descriptor(M, N, stride_C, CLayout{}));

        ck_tile::FillUniformDistributionIntegerValue<ADataType>{-5, 5}(a_m_k);
        ck_tile::FillUniformDistributionIntegerValue<BDataType>{-5, 5}(b_k_n);

        ck_tile::DeviceMem a_m_k_dev_buf(a_m_k.get_element_space_size_in_bytes());
        ck_tile::DeviceMem b_k_n_dev_buf(b_k_n.get_element_space_size_in_bytes());
        ck_tile::DeviceMem c_m_n_dev_buf(c_m_n_dev_result.get_element_space_size_in_bytes());

        a_m_k_dev_buf.ToDevice(a_m_k.data());
        b_k_n_dev_buf.ToDevice(b_k_n.data());
        c_m_n_dev_buf.SetZero();
        c_m_n_dev_result.SetZero();

        ck_tile::GemmHostArgs args;
        args.a_ptr    = a_m_k_dev_buf.GetDeviceBuffer();
        args.b_ptr    = b_k_n_dev_buf.GetDeviceBuffer();
        args.c_ptr    = c_m_n_dev_buf.GetDeviceBuffer();
        args.k_batch  = kbatch;
        args.M        = M;
        args.N        = N;
        args.K        = K;
        args.stride_A = stride_A;
        args.stride_B = stride_B;
        args.stride_C = stride_C;

        invoke_gemm<PadM, PadN, PadK>(args, ck_tile::stream_config{nullptr, false});

        c_m_n_dev_buf.FromDevice(c_m_n_dev_result.data());
        bool pass = true;

        ck_tile::HostTensor<CDataType> c_m_n_host_ref(
            f_host_tensor_descriptor(M, N, stride_C, CLayout{}));
        c_m_n_host_ref.SetZero();

        ck_tile::reference_gemm<ADataType, BDataType, AccDataType, CDataType>(
            a_m_k, b_k_n, c_m_n_host_ref);

        pass = ck_tile::check_err(c_m_n_dev_result, c_m_n_host_ref);
        EXPECT_TRUE(pass);
    }
};
