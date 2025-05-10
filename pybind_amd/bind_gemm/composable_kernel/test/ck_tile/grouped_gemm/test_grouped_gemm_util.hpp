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
#include "ck_tile/ops/gemm/kernel/grouped_gemm_kernel.hpp"

template <typename Tuple>
class TestCkTileGroupedGemm : public ::testing::Test
{
    protected:
    using ALayout     = std::tuple_element_t<0, Tuple>;
    using BLayout     = std::tuple_element_t<1, Tuple>;
    using CLayout     = std::tuple_element_t<2, Tuple>;
    using ADataType   = std::tuple_element_t<3, Tuple>;
    using BDataType   = std::tuple_element_t<4, Tuple>;
    using AccDataType = std::tuple_element_t<5, Tuple>;
    using CDataType   = std::tuple_element_t<6, Tuple>;

    struct GroupedGemKernelParam
    {
        static const bool kPadM = false;
        static const bool kPadN = false;
        static const bool kPadK = false;

        static const int kBlockPerCu         = 1;
        static const ck_tile::index_t M_Tile = 128;
        static const ck_tile::index_t N_Tile = 128;
        static const ck_tile::index_t K_Tile = 32;

        static const ck_tile::index_t M_Warp = 2;
        static const ck_tile::index_t N_Warp = 2;
        static const ck_tile::index_t K_Warp = 1;

        static const ck_tile::index_t M_Warp_Tile = 32;
        static const ck_tile::index_t N_Warp_Tile = 32;
        static const ck_tile::index_t K_Warp_Tile = 8;
    };

    using CodegenGemmShape =
        ck_tile::TileGemmShape<ck_tile::sequence<GroupedGemKernelParam::M_Tile,
                                                 GroupedGemKernelParam::N_Tile,
                                                 GroupedGemKernelParam::K_Tile>,
                               ck_tile::sequence<GroupedGemKernelParam::M_Warp,
                                                 GroupedGemKernelParam::N_Warp,
                                                 GroupedGemKernelParam::K_Warp>,
                               ck_tile::sequence<GroupedGemKernelParam::M_Warp_Tile,
                                                 GroupedGemKernelParam::N_Warp_Tile,
                                                 GroupedGemKernelParam::K_Warp_Tile>>;

    using TilePartitioner = ck_tile::GemmTile1DPartitioner<CodegenGemmShape>;

    template <typename ALayout, typename BLayout, typename CLayout>
    using CodegenGemmTraits = ck_tile::TileGemmTraits<GroupedGemKernelParam::kPadM,
                                                      GroupedGemKernelParam::kPadN,
                                                      GroupedGemKernelParam::kPadK,
                                                      ALayout,
                                                      BLayout,
                                                      CLayout>;

    template <typename ALayout, typename BLayout, typename CLayout>
    using CodegenPipelineProblem =
        ck_tile::GemmPipelineProblem<ADataType,
                                     BDataType,
                                     AccDataType,
                                     CodegenGemmShape,
                                     CodegenGemmTraits<ALayout, BLayout, CLayout>>;

    template <typename ALayout, typename BLayout, typename CLayout>
    using CodegenGemmPipeline =
        ck_tile::GemmPipelineAGmemBGmemCRegV1<CodegenPipelineProblem<ALayout, BLayout, CLayout>>;

    template <typename ALayout, typename BLayout, typename CLayout>
    using GemmEpilogue = ck_tile::CShuffleEpilogue<ck_tile::CShuffleEpilogueProblem<
        AccDataType,
        CDataType,
        CLayout,
        CodegenGemmPipeline<ALayout, BLayout, CLayout>::BlockSize,
        TilePartitioner::MPerBlock,
        TilePartitioner::NPerBlock,
        GroupedGemKernelParam::M_Warp,
        GroupedGemKernelParam::N_Warp,
        GroupedGemKernelParam::M_Warp_Tile,
        GroupedGemKernelParam::N_Warp_Tile,
        GroupedGemKernelParam::K_Warp_Tile,
        CodegenPipelineProblem<ALayout, BLayout, CLayout>::TransposeC>>;

    template <typename ALayout, typename BLayout, typename CLayout>
    using Kernel = ck_tile::GroupedGemmKernel<TilePartitioner,
                                              CodegenGemmPipeline<ALayout, BLayout, CLayout>,
                                              GemmEpilogue<ALayout, BLayout, CLayout>>;

    using grouped_gemm_kargs = ck_tile::GroupedGemmHostArgs;
    std::size_t GetWorkspaceSize(const std::vector<grouped_gemm_kargs>& gemm_descs)
    {
        return Kernel<std::nullptr_t, std::nullptr_t, std::nullptr_t>::GetWorkSpaceSize(gemm_descs);
    }

    template <typename ALayout, typename BLayout, typename CLayout>
    void invoke_grouped_gemm(const std::vector<grouped_gemm_kargs>& gemm_descs,
                             const ck_tile::stream_config& s,
                             void* p_workspace_)
    {
        using GroupedGemmKernel = Kernel<ALayout, BLayout, CLayout>;

        auto arguments = GroupedGemmKernel::MakeKargs(gemm_descs);

        const dim3 grids      = GroupedGemmKernel::GridSize(gemm_descs);
        constexpr dim3 blocks = GroupedGemmKernel::BlockSize();

        ck_tile::hip_check_error(hipMemcpyWithStream(
            p_workspace_,
            arguments.data(),
            arguments.size() * sizeof(typename GroupedGemmKernel::GemmTransKernelArg),
            hipMemcpyHostToDevice,
            s.stream_id_));

        if(s.log_level_ > 0)
        {
            std::cout << "Launching kernel with args:"
                      << " grid: {" << grids.x << ", " << grids.y << ", " << grids.z << "}"
                      << ", blocks: {" << blocks.x << ", " << blocks.y << ", " << blocks.z << "}"
                      << std::endl;
        }
        ck_tile::launch_kernel(s,
                               ck_tile::make_kernel<blocks.x, GroupedGemKernelParam::kBlockPerCu>(
                                   GroupedGemmKernel{},
                                   grids,
                                   blocks,
                                   0,
                                   ck_tile::cast_pointer_to_constant_address_space(p_workspace_),
                                   gemm_descs.size()));
    }

    public:
    void Run(const std::vector<int>& Ms,
             const std::vector<int>& Ns,
             const std::vector<int>& Ks,
             std::vector<int>& stride_As,
             std::vector<int>& stride_Bs,
             std::vector<int>& stride_Cs,
             const int group_count = 16)
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

        std::vector<ck_tile::HostTensor<ADataType>> a_m_k_tensors;
        std::vector<ck_tile::HostTensor<BDataType>> b_k_n_tensors;
        std::vector<ck_tile::HostTensor<CDataType>> c_m_n_tensors;

        a_m_k_tensors.reserve(group_count);
        b_k_n_tensors.reserve(group_count);
        c_m_n_tensors.reserve(group_count);

        std::vector<std::unique_ptr<ck_tile::DeviceMem>> a_m_k_dev_buf;
        std::vector<std::unique_ptr<ck_tile::DeviceMem>> b_k_n_dev_buf;
        std::vector<std::unique_ptr<ck_tile::DeviceMem>> c_m_n_dev_buf;

        a_m_k_dev_buf.reserve(group_count);
        b_k_n_dev_buf.reserve(group_count);
        c_m_n_dev_buf.reserve(group_count);

        std::vector<grouped_gemm_kargs> gemm_descs;
        gemm_descs.reserve(group_count);

        for(int i = 0; i < group_count; ++i)
        {
            const ck_tile::index_t M = Ms[i];
            const ck_tile::index_t N = Ns[i];
            const ck_tile::index_t K = Ks[i];

            stride_As[i] = f_get_default_stride(M, N, stride_As[i], ALayout{});
            stride_Bs[i] = f_get_default_stride(K, N, stride_Bs[i], BLayout{});
            stride_Cs[i] = f_get_default_stride(M, N, stride_Cs[i], CLayout{});

            a_m_k_tensors.push_back(ck_tile::HostTensor<ADataType>(
                f_host_tensor_descriptor(M, K, stride_As[i], ALayout{})));
            b_k_n_tensors.push_back(ck_tile::HostTensor<BDataType>(
                f_host_tensor_descriptor(K, N, stride_Bs[i], BLayout{})));
            c_m_n_tensors.push_back(ck_tile::HostTensor<CDataType>(
                f_host_tensor_descriptor(M, N, stride_Cs[i], CLayout{})));

            std::cout << "gemm[" << i << "]"
                      << " a_m_k: " << a_m_k_tensors[i].mDesc
                      << " b_k_n: " << b_k_n_tensors[i].mDesc
                      << " c_m_n: " << c_m_n_tensors[i].mDesc << std::endl;

            ck_tile::FillUniformDistribution<ADataType>{-5.f, 5.f}(a_m_k_tensors[i]);
            ck_tile::FillUniformDistribution<BDataType>{-5.f, 5.f}(b_k_n_tensors[i]);

            a_m_k_dev_buf.push_back(std::make_unique<ck_tile::DeviceMem>(
                a_m_k_tensors[i].get_element_space_size_in_bytes()));
            b_k_n_dev_buf.push_back(std::make_unique<ck_tile::DeviceMem>(
                b_k_n_tensors[i].get_element_space_size_in_bytes()));
            c_m_n_dev_buf.push_back(std::make_unique<ck_tile::DeviceMem>(
                c_m_n_tensors[i].get_element_space_size_in_bytes()));

            a_m_k_dev_buf[i]->ToDevice(a_m_k_tensors[i].data());
            b_k_n_dev_buf[i]->ToDevice(b_k_n_tensors[i].data());
            c_m_n_dev_buf[i]->SetZero();
            c_m_n_tensors[i].SetZero();

            const void* p_a = a_m_k_dev_buf[i]->GetDeviceBuffer();
            const void* p_b = b_k_n_dev_buf[i]->GetDeviceBuffer();
            void* p_c       = c_m_n_dev_buf[i]->GetDeviceBuffer();

            gemm_descs.push_back(
                {p_a, p_b, p_c, M, N, K, stride_As[i], stride_Bs[i], stride_Cs[i]});
        }

        ck_tile::DeviceMem gemm_workspace;
        gemm_workspace.Realloc(GetWorkspaceSize(gemm_descs));

        invoke_grouped_gemm<ALayout, BLayout, CLayout>(
            gemm_descs, ck_tile::stream_config{nullptr, false}, gemm_workspace.GetDeviceBuffer());

        for(int i = 0; i < group_count; i++)
        {
            c_m_n_dev_buf[i]->FromDevice(c_m_n_tensors[i].data());
        }

        bool pass{true};
        for(int i = 0; i < group_count; ++i)
        {
            ck_tile::HostTensor<CDataType> c_m_n_host_ref(
                f_host_tensor_descriptor(Ms[i], Ns[i], stride_Cs[i], CLayout{}));
            c_m_n_host_ref.SetZero();
            ck_tile::reference_gemm<ADataType, BDataType, AccDataType, CDataType>(
                a_m_k_tensors[i], b_k_n_tensors[i], c_m_n_host_ref);
            pass &= ck_tile::check_err(c_m_n_tensors[i], c_m_n_host_ref);
        }
        EXPECT_TRUE(pass);
    }
};
