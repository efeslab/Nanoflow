// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core/numeric/math.hpp"
#include "ck_tile/core/utility/literals.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_scheduler.hpp"
#include "ck_tile/ops/gemm/kernel/gemm_kernel.hpp"
#include "ck_tile/host.hpp"

namespace ck_tile {

struct GroupedGemmHostArgs : public ck_tile::GemmHostArgs
{
    CK_TILE_HOST GroupedGemmHostArgs() noexcept = default;
    CK_TILE_HOST GroupedGemmHostArgs(const void* a_ptr_,
                                     const void* b_ptr_,
                                     void* c_ptr_,
                                     ck_tile::index_t M_,
                                     ck_tile::index_t N_,
                                     ck_tile::index_t K_,
                                     ck_tile::index_t stride_A_,
                                     ck_tile::index_t stride_B_,
                                     ck_tile::index_t stride_C_)
        : GemmHostArgs(a_ptr_, b_ptr_, c_ptr_, KBatch, M_, N_, K_, stride_A_, stride_B_, stride_C_)
    {
    }

    private:
    static constexpr index_t KBatch = 1;
};

template <typename TilePartitioner_, typename GemmPipeline_, typename EpiloguePipeline_>
struct GroupedGemmKernel : public GemmKernel<TilePartitioner_, GemmPipeline_, EpiloguePipeline_>
{
    using TilePartitioner  = remove_cvref_t<TilePartitioner_>;
    using GemmPipeline     = remove_cvref_t<GemmPipeline_>;
    using EpiloguePipeline = remove_cvref_t<EpiloguePipeline_>;
    using ALayout          = remove_cvref_t<typename GemmPipeline::ALayout>;
    using BLayout          = remove_cvref_t<typename GemmPipeline::BLayout>;
    using CLayout          = remove_cvref_t<typename GemmPipeline::CLayout>;

    using ADataType = remove_cvref_t<typename GemmPipeline::ADataType>;
    using BDataType = remove_cvref_t<typename GemmPipeline::BDataType>;
    using CDataType = remove_cvref_t<typename EpiloguePipeline::ODataType>;

    using OffsetTile1DPartitioner = OffsettedTile1DPartitioner<TilePartitioner>;
    using Base                    = GemmKernel<TilePartitioner_, GemmPipeline_, EpiloguePipeline_>;
    using GemmKernelArgs          = typename Base::GemmKernelArgs;

    static constexpr index_t KernelBlockSize = GemmPipeline::BlockSize;

    struct GemmTransKernelArg
    {
        GemmKernelArgs group_karg;
        ck_tile::index_t block_start;
        ck_tile::index_t block_end;

        GemmTransKernelArg() = default;
        GemmTransKernelArg(GemmKernelArgs&& karg, index_t bl_start, index_t bl_end)
            : group_karg{karg}, block_start{bl_start}, block_end{bl_end}
        {
        }
    };

    __host__ static auto GetWorkSpaceSize(const std::vector<GroupedGemmHostArgs>& gemm_descs)
        -> std::size_t
    {
        return gemm_descs.size() * sizeof(GemmTransKernelArg);
    }

    __host__ static constexpr auto BlockSize() -> dim3 { return dim3(KernelBlockSize); }

    __host__ static constexpr auto GridSize(const std::vector<GroupedGemmHostArgs>& gemm_descs)
    {
        index_t grid_size = 0;
        for(const auto& it_desc : gemm_descs)
        {
            const auto local_grid_size = TilePartitioner::GridSize(it_desc.M, it_desc.N);
            grid_size += local_grid_size * it_desc.k_batch;
        }
        return dim3(grid_size, 1, 1);
    }

    CK_TILE_HOST static auto MakeKargs(const std::vector<GroupedGemmHostArgs>& gemm_descs)
        -> std::vector<GemmTransKernelArg>
    {
        std::vector<GemmTransKernelArg> gemm_kernel_args_;
        index_t group_count = ck_tile::type_convert<ck_tile::index_t>(gemm_descs.size());
        index_t grid_size   = 0;
        gemm_kernel_args_.reserve(group_count);

        for(std::size_t i = 0; i < gemm_descs.size(); ++i)
        {
            const index_t M = gemm_descs[i].M;
            const index_t N = gemm_descs[i].N;
            const index_t K = gemm_descs[i].K;

            if(M == 0 || N == 0 || K == 0)
            {
                continue;
            }

            const index_t stride_a = gemm_descs[i].stride_A;
            const index_t stride_b = gemm_descs[i].stride_B;
            const index_t stride_c = gemm_descs[i].stride_C;

            const index_t grid_size_grp = TilePartitioner::GridSize(M, N) * gemm_descs[i].k_batch;

            const index_t block_start = grid_size;
            const index_t block_end   = grid_size + grid_size_grp;

            grid_size += grid_size_grp;

            auto karg = GemmKernelArgs{type_convert<const ADataType*>(gemm_descs[i].a_ptr),
                                       type_convert<const BDataType*>(gemm_descs[i].b_ptr),
                                       type_convert<CDataType*>(gemm_descs[i].c_ptr),
                                       M,
                                       N,
                                       K,
                                       stride_a,
                                       stride_b,
                                       stride_c,
                                       gemm_descs[i].k_batch};

            gemm_kernel_args_.emplace_back(std::move(karg), block_start, block_end);
        }

        return gemm_kernel_args_;
    }

    CK_TILE_HOST_DEVICE static constexpr auto GetSmemSize() -> index_t
    {
        return max(GemmPipeline::GetSmemSize(), EpiloguePipeline::GetSmemSize());
    }

    CK_TILE_DEVICE void Run(const GemmTransKernelArg& kargs) const
    {
        const auto [iM, iN] = OffsetTile1DPartitioner::GetOffsetedTileIndex(
            kargs.block_start, kargs.group_karg.M, kargs.group_karg.N);

        const index_t i_m = __builtin_amdgcn_readfirstlane(iM * TilePartitioner::MPerBlock);
        const index_t i_n = __builtin_amdgcn_readfirstlane(iN * TilePartitioner::NPerBlock);

        const typename Base::SplitKBatchOffset splitk_batch_offset(kargs.group_karg, blockIdx.z);

        const ADataType* a_ptr = static_cast<const ADataType*>(kargs.group_karg.a_ptr);
        const BDataType* b_ptr = static_cast<const BDataType*>(kargs.group_karg.b_ptr);
        CDataType* c_ptr       = static_cast<CDataType*>(kargs.group_karg.c_ptr);

        // allocate LDS
        __shared__ char smem_ptr[GetSmemSize()];

        this->RunGemm(
            a_ptr, b_ptr, c_ptr, smem_ptr, kargs.group_karg, splitk_batch_offset, i_m, i_n);
    }

    CK_TILE_DEVICE void operator()(const void CK_CONSTANT_ADDRESS_SPACE* gemm_descs_const,
                                   index_t group_count) const
    {
        const index_t block_id   = ck_tile::get_block_1d_id();
        const auto gemm_desc_ptr = reinterpret_cast<const GemmTransKernelArg*>(
            cast_pointer_to_generic_address_space(gemm_descs_const));

        index_t left     = 0;
        index_t right    = group_count;
        index_t group_id = index_t((left + right) >> 1);

        while((!(block_id >= gemm_desc_ptr[group_id].block_start &&
                 block_id < gemm_desc_ptr[group_id].block_end)) &&
              left <= right)
        {
            if(block_id < gemm_desc_ptr[group_id].block_start)
            {
                right = group_id;
            }
            else
            {
                left = group_id;
            }
            group_id = index_t((left + right) >> 1);
        }

        Run(gemm_desc_ptr[group_id]);
    }
};

} // namespace ck_tile
