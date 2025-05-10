// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#include <hip/hip_runtime.h>

#include <cstring>
#include <iostream>
#include <ostream>
#include <string>
#include <tuple>
#include <memory>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/host.hpp"
#include "grouped_gemm.hpp"

namespace {

struct GroupedGemmKernelParam
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
    ck_tile::TileGemmShape<ck_tile::sequence<GroupedGemmKernelParam::M_Tile,
                                             GroupedGemmKernelParam::N_Tile,
                                             GroupedGemmKernelParam::K_Tile>,
                           ck_tile::sequence<GroupedGemmKernelParam::M_Warp,
                                             GroupedGemmKernelParam::N_Warp,
                                             GroupedGemmKernelParam::K_Warp>,
                           ck_tile::sequence<GroupedGemmKernelParam::M_Warp_Tile,
                                             GroupedGemmKernelParam::N_Warp_Tile,
                                             GroupedGemmKernelParam::K_Warp_Tile>>;

using TilePartitioner = ck_tile::GemmTile1DPartitioner<CodegenGemmShape>;

template <typename ALayout, typename BLayout, typename CLayout>
using CodegenGemmTraits = ck_tile::TileGemmTraits<GroupedGemmKernelParam::kPadM,
                                                  GroupedGemmKernelParam::kPadN,
                                                  GroupedGemmKernelParam::kPadK,
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
    CodegenPipelineProblem<ALayout, BLayout, CLayout>::kBlockSize,
    TilePartitioner::MPerBlock,
    TilePartitioner::NPerBlock,
    GroupedGemmKernelParam::M_Warp,
    GroupedGemmKernelParam::N_Warp,
    GroupedGemmKernelParam::M_Warp_Tile,
    GroupedGemmKernelParam::N_Warp_Tile,
    GroupedGemmKernelParam::K_Warp_Tile,
    CodegenPipelineProblem<ALayout, BLayout, CLayout>::TransposeC>>;

template <typename ALayout, typename BLayout, typename CLayout>
using Kernel = ck_tile::GroupedGemmKernel<TilePartitioner,
                                          CodegenGemmPipeline<ALayout, BLayout, CLayout>,
                                          GemmEpilogue<ALayout, BLayout, CLayout>>;
}; // namespace

std::size_t get_workspace_size(const std::vector<grouped_gemm_kargs>& gemm_descs)
{
    return ::Kernel<std::nullptr_t, std::nullptr_t, std::nullptr_t>::GetWorkSpaceSize(gemm_descs);
}

template <typename ALayout, typename BLayout, typename CLayout>
float grouped_gemm(const std::vector<grouped_gemm_kargs>& gemm_descs,
                   const ck_tile::stream_config& s,
                   void* p_workspace_)
{
    using GroupedGemmKernel = ::Kernel<ALayout, BLayout, CLayout>;

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

    float ave_time =
        ck_tile::launch_kernel(s,
                               ck_tile::make_kernel<blocks.x, GroupedGemmKernelParam::kBlockPerCu>(
                                   GroupedGemmKernel{},
                                   grids,
                                   blocks,
                                   0,
                                   ck_tile::cast_pointer_to_constant_address_space(p_workspace_),
                                   gemm_descs.size()));
    return ave_time;
}

#include "run_grouped_gemm_example.inc"

int main(int argc, char* argv[]) { return !run_grouped_gemm_example(argc, argv); }
