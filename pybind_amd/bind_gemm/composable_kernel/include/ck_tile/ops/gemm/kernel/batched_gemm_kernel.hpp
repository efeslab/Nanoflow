// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/ops/gemm/kernel/gemm_kernel.hpp"

namespace ck_tile {

struct BatchedGemmHostArgs : public ck_tile::GemmHostArgs
{
    CK_TILE_HOST BatchedGemmHostArgs() = default;
    CK_TILE_HOST BatchedGemmHostArgs(const void* a_ptr_,
                                     const void* b_ptr_,
                                     void* c_ptr_,
                                     ck_tile::index_t k_batch_,
                                     ck_tile::index_t M_,
                                     ck_tile::index_t N_,
                                     ck_tile::index_t K_,
                                     ck_tile::index_t stride_A_,
                                     ck_tile::index_t stride_B_,
                                     ck_tile::index_t stride_C_,
                                     ck_tile::index_t batch_stride_A_,
                                     ck_tile::index_t batch_stride_B_,
                                     ck_tile::index_t batch_stride_C_,
                                     ck_tile::index_t batch_count_)
        : GemmHostArgs(
              a_ptr_, b_ptr_, c_ptr_, k_batch_, M_, N_, K_, stride_A_, stride_B_, stride_C_),
          batch_stride_A(batch_stride_A_),
          batch_stride_B(batch_stride_B_),
          batch_stride_C(batch_stride_C_),
          batch_count(batch_count_)
    {
    }

    ck_tile::index_t batch_stride_A;
    ck_tile::index_t batch_stride_B;
    ck_tile::index_t batch_stride_C;
    ck_tile::index_t batch_count;
};

template <typename TilePartitioner_, typename GemmPipeline_, typename EpiloguePipeline_>
struct BatchedGemmKernel : public GemmKernel<TilePartitioner_, GemmPipeline_, EpiloguePipeline_>
{
    using Base = GemmKernel<TilePartitioner_, GemmPipeline_, EpiloguePipeline_>;

    using GemmKernelArgs = typename Base::GemmKernelArgs;

    using ADataType = typename Base::ADataType;
    using BDataType = typename Base::BDataType;
    using CDataType = typename Base::CDataType;

    using TilePartitioner  = typename Base::TilePartitioner;
    using GemmPipeline     = typename Base::GemmPipeline;
    using EpiloguePipeline = typename Base::EpiloguePipeline;
    using ALayout          = typename Base::ALayout;
    using BLayout          = typename Base::BLayout;
    using CLayout          = typename Base::CLayout;

    struct BatchedGemmKernelArgs : GemmKernelArgs
    {
        index_t batch_stride_A;
        index_t batch_stride_B;
        index_t batch_stride_C;
        index_t batch_count;
    };

    using KernelArgs = BatchedGemmKernelArgs;

    __host__ static constexpr auto
    GridSize(index_t M, index_t N, index_t KBatch, index_t batch_count)
    {
        return dim3(TilePartitioner::GridSize(M, N), batch_count, KBatch);
    }

    __host__ static constexpr auto BlockSize() { return dim3(Base::KernelBlockSize); }

    CK_TILE_HOST static constexpr BatchedGemmKernelArgs
    MakeKernelArgs(const BatchedGemmHostArgs& hostArgs)
    {
        return BatchedGemmKernelArgs{{hostArgs.a_ptr,
                                      hostArgs.b_ptr,
                                      hostArgs.c_ptr,
                                      hostArgs.M,
                                      hostArgs.N,
                                      hostArgs.K,
                                      hostArgs.stride_A,
                                      hostArgs.stride_B,
                                      hostArgs.stride_C,
                                      hostArgs.k_batch},
                                     hostArgs.batch_stride_A,
                                     hostArgs.batch_stride_B,
                                     hostArgs.batch_stride_C,
                                     hostArgs.batch_count};
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return max(GemmPipeline::GetSmemSize(), EpiloguePipeline::GetSmemSize());
    }

    CK_TILE_DEVICE void operator()(BatchedGemmKernelArgs kargs) const
    {
        const auto [iM, iN] = TilePartitioner{kargs.M, kargs.N}.GetOutputTileIndex(blockIdx.x);
        const index_t i_m   = __builtin_amdgcn_readfirstlane(iM * TilePartitioner::MPerBlock);
        const index_t i_n   = __builtin_amdgcn_readfirstlane(iN * TilePartitioner::NPerBlock);

        const auto i_batch  = __builtin_amdgcn_readfirstlane(blockIdx.y);
        const auto i_splitk = __builtin_amdgcn_readfirstlane(blockIdx.z);

        const typename Base::SplitKBatchOffset splitk_batch_offset(kargs, i_splitk);

        //  options
        const auto batch_stride_A = __builtin_amdgcn_readfirstlane(kargs.batch_stride_A);
        const auto batch_offset_A = __builtin_amdgcn_readfirstlane(i_batch * batch_stride_A);
        const ADataType* a_ptr    = static_cast<const ADataType*>(kargs.a_ptr) + batch_offset_A +
                                 splitk_batch_offset.a_k_split_offset;

        const auto batch_stride_B = __builtin_amdgcn_readfirstlane(kargs.batch_stride_B);
        const auto batch_offset_B = __builtin_amdgcn_readfirstlane(i_batch * batch_stride_B);
        const BDataType* b_ptr    = static_cast<const BDataType*>(kargs.b_ptr) + batch_offset_B +
                                 splitk_batch_offset.b_k_split_offset;

        const auto batch_stride_C = __builtin_amdgcn_readfirstlane(kargs.batch_stride_C);
        const auto batch_offset_C = __builtin_amdgcn_readfirstlane(i_batch * batch_stride_C);
        CDataType* c_ptr          = static_cast<CDataType*>(kargs.c_ptr) + batch_offset_C;

        // allocate LDS
        __shared__ char smem_ptr[GetSmemSize()];

        if(kargs.k_batch == 1)
        {
            this->RunGemm(a_ptr, b_ptr, c_ptr, smem_ptr, kargs, splitk_batch_offset, i_m, i_n);
        }
        else
        {
            this->template RunGemm<memory_operation_enum::atomic_add>(
                a_ptr, b_ptr, c_ptr, smem_ptr, kargs, splitk_batch_offset, i_m, i_n);
        }
    }
};

} // namespace ck_tile
