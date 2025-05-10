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
#include "ck_tile/ops/gemm/kernel/batched_gemm_kernel.hpp"

template <typename Tuple>
class TestCkTileBatchedGemm : public ::testing::Test
{
    protected:
    using ALayout     = std::tuple_element_t<0, Tuple>;
    using BLayout     = std::tuple_element_t<1, Tuple>;
    using CLayout     = std::tuple_element_t<2, Tuple>;
    using ADataType   = std::tuple_element_t<3, Tuple>;
    using BDataType   = std::tuple_element_t<4, Tuple>;
    using AccDataType = std::tuple_element_t<5, Tuple>;
    using CDataType   = std::tuple_element_t<6, Tuple>;

    template <typename ALayout, typename BLayout, typename CLayout>
    void invoke_batched_gemm(const ck_tile::BatchedGemmHostArgs& args,
                             const ck_tile::stream_config& s)
    {
        // The kPadM, kPadN, kPadK & kBlockPerCu should also come from the Codegen part.
        constexpr bool kPadM = false;
        constexpr bool kPadN = false;
        constexpr bool kPadK = false;

        constexpr int kBlockPerCu = 1;

        // This part comes from the Codegen
        constexpr ck_tile::index_t M_Tile = 128;
        constexpr ck_tile::index_t N_Tile = 128;
        constexpr ck_tile::index_t K_Tile = 32;

        constexpr ck_tile::index_t M_Warp = 2;
        constexpr ck_tile::index_t N_Warp = 2;
        constexpr ck_tile::index_t K_Warp = 1;

        constexpr ck_tile::index_t M_Warp_Tile = 32;
        constexpr ck_tile::index_t N_Warp_Tile = 32;
        constexpr ck_tile::index_t K_Warp_Tile = 8;

        using CodegenGemmShape =
            ck_tile::TileGemmShape<ck_tile::sequence<M_Tile, N_Tile, K_Tile>,
                                   ck_tile::sequence<M_Warp, N_Warp, K_Warp>,
                                   ck_tile::sequence<M_Warp_Tile, N_Warp_Tile, K_Warp_Tile>>;

        using TilePartitioner = ck_tile::GemmTile1DPartitioner<CodegenGemmShape>;

        using CodegenGemmTraits =
            ck_tile::TileGemmTraits<kPadM, kPadN, kPadK, ALayout, BLayout, CLayout>;

        using CodegenPipelineProblem = ck_tile::GemmPipelineProblem<ADataType,
                                                                    BDataType,
                                                                    AccDataType,
                                                                    CodegenGemmShape,
                                                                    CodegenGemmTraits>;

        using CodegenGemmPipeline = ck_tile::GemmPipelineAGmemBGmemCRegV1<CodegenPipelineProblem>;

        using GemmEpilogue = ck_tile::CShuffleEpilogue<
            ck_tile::CShuffleEpilogueProblem<AccDataType,
                                             CDataType,
                                             CLayout,
                                             CodegenGemmPipeline::BlockSize,
                                             TilePartitioner::MPerBlock,
                                             TilePartitioner::NPerBlock,
                                             M_Warp,
                                             N_Warp,
                                             M_Warp_Tile,
                                             N_Warp_Tile,
                                             K_Warp_Tile,
                                             CodegenPipelineProblem::TransposeC>>;
        using Kernel =
            ck_tile::BatchedGemmKernel<TilePartitioner, CodegenGemmPipeline, GemmEpilogue>;

        auto kargs = Kernel::MakeKernelArgs(args);

        const dim3 grids      = Kernel::GridSize(args.M, args.N, args.k_batch, args.batch_count);
        constexpr dim3 blocks = Kernel::BlockSize();

        if(s.log_level_ > 0)
        {
            std::cout << "Launching kernel with args:"
                      << " grid: {" << grids.x << ", " << grids.y << ", " << grids.z << "}"
                      << ", blocks: {" << blocks.x << ", " << blocks.y << ", " << blocks.z << "}"
                      << std::endl;
        }

        ck_tile::launch_kernel(
            s, ck_tile::make_kernel<blocks.x, kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));
    }

    public:
    void Run(const int M,
             const int N,
             const int K,
             int StrideA            = 128,
             int StrideB            = 128,
             int StrideC            = 128,
             const int BatchStrideA = 32768,
             const int BatchStrideB = 16384,
             const int BatchStrideC = 32768,
             const int BatchCount   = 16)
    {
        using namespace ck_tile::literals;

        auto f_host_tensor_descriptor = [](std::size_t batch_count_,
                                           std::size_t row,
                                           std::size_t col,
                                           std::size_t stride,
                                           std::size_t batch_stride,
                                           auto layout) {
            if constexpr(std::is_same_v<decltype(layout), ck_tile::tensor_layout::gemm::RowMajor>)
            {
                return ck_tile::HostTensorDescriptor({batch_count_, row, col},
                                                     {batch_stride, stride, 1_uz});
            }
            else
            {
                return ck_tile::HostTensorDescriptor({batch_count_, row, col},
                                                     {batch_stride, 1_uz, stride});
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

        StrideA = f_get_default_stride(M, K, StrideA, ALayout{});
        StrideB = f_get_default_stride(K, N, StrideB, BLayout{});
        StrideC = f_get_default_stride(M, N, StrideC, CLayout{});

        ck_tile::HostTensor<ADataType> a_m_k(
            f_host_tensor_descriptor(BatchCount, M, K, StrideA, BatchStrideA, ALayout{}));
        ck_tile::HostTensor<BDataType> b_k_n(
            f_host_tensor_descriptor(BatchCount, K, N, StrideB, BatchStrideB, BLayout{}));
        ck_tile::HostTensor<CDataType> c_m_n_dev_result(
            f_host_tensor_descriptor(BatchCount, M, N, StrideC, BatchStrideC, CLayout{}));

        ck_tile::FillUniformDistribution<ADataType>{-5.f, 5.f}(a_m_k);
        ck_tile::FillUniformDistribution<BDataType>{-5.f, 5.f}(b_k_n);

        ck_tile::DeviceMem a_m_k_dev_buf(a_m_k.get_element_space_size_in_bytes());
        ck_tile::DeviceMem b_k_n_dev_buf(b_k_n.get_element_space_size_in_bytes());
        ck_tile::DeviceMem c_m_n_dev_buf(c_m_n_dev_result.get_element_space_size_in_bytes());

        a_m_k_dev_buf.ToDevice(a_m_k.data());
        b_k_n_dev_buf.ToDevice(b_k_n.data());
        c_m_n_dev_buf.SetZero();
        c_m_n_dev_result.SetZero();

        ck_tile::BatchedGemmHostArgs args;
        args.a_ptr          = a_m_k_dev_buf.GetDeviceBuffer();
        args.b_ptr          = b_k_n_dev_buf.GetDeviceBuffer();
        args.c_ptr          = c_m_n_dev_buf.GetDeviceBuffer();
        args.k_batch        = 1;
        args.M              = M;
        args.N              = N;
        args.K              = K;
        args.stride_A       = StrideA;
        args.stride_B       = StrideB;
        args.stride_C       = StrideC;
        args.batch_stride_A = BatchStrideA;
        args.batch_stride_B = BatchStrideB;
        args.batch_stride_C = BatchStrideC;
        args.batch_count    = BatchCount;

        invoke_batched_gemm<ALayout, BLayout, CLayout>(args,
                                                       ck_tile::stream_config{nullptr, false});

        std::cout << "Run kernel with M =" << M << " N =" << N << " K =" << K
                  << " StrideA =" << StrideA << " StrideB =" << StrideB << " StrideC =" << StrideC
                  << " BatchStrideA =" << BatchStrideA << " BatchStrideB =" << BatchStrideB
                  << " BatchStrideC =" << BatchStrideC << " BatchCount =" << BatchCount
                  << std::endl;

        c_m_n_dev_buf.FromDevice(c_m_n_dev_result.data());
        bool pass = true;

        ck_tile::HostTensor<CDataType> c_m_n_host_ref(
            f_host_tensor_descriptor(BatchCount, M, N, StrideC, BatchStrideC, CLayout{}));
        c_m_n_host_ref.SetZero();

        const auto b_n_k = b_k_n.transpose({0, 2, 1});
        ck_tile::reference_batched_gemm<ADataType, BDataType, AccDataType, CDataType>(
            a_m_k, b_n_k, c_m_n_host_ref);

        pass = ck_tile::check_err(c_m_n_dev_result, c_m_n_host_ref);
        EXPECT_TRUE(pass);
    }
};
