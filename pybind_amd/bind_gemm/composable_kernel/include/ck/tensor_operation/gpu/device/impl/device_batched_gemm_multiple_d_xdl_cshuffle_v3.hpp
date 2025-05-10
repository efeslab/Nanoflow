// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <sstream>

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_batched_gemm_multi_d.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_v3_multi_d.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"
#include "ck/host_utility/flush_cache.hpp"

namespace ck {

// Currently we do not have a elegant way to put single lds buffer & double lds buffer pipe in same
// kernel function Blockers:
// 1. Two separted declaration of __shared__ pointer is the key to make sure data access operate on
// two lds chunks.
// 2. Occupied __shared__ won't release until whole shader end, a.k.a AB and C may not use same lds
// buffer when we declare __shared__ inside blkgemmpipe
template <typename GridwiseGemm,
          typename BatchedGemmArg,
          bool HasMainKBlockLoop,
          InMemoryDataOperationEnum CGlobalMemoryDataOperation,
          index_t MinimumOccupancy = 1,
          TailNumber TailNum       = TailNumber::Full>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, MinimumOccupancy)
#endif
        kernel_batched_gemm_xdl_cshuffle_v3_multi_d(BatchedGemmArg karg)
{
#if(!defined(__HIP_DEVICE_COMPILE__) || defined(__gfx9__))
    __shared__ char p_shared[GridwiseGemm::GetSharedMemoryNumberOfByte()];

    const index_t g_idx = blockIdx.z % karg.Batch;
    const index_t k_idx = blockIdx.z / karg.Batch;

    const auto a_batch_offset  = karg.compute_ptr_offset_of_batch.GetAPtrOffset(g_idx);
    const auto b_batch_offset  = karg.compute_ptr_offset_of_batch.GetBPtrOffset(g_idx);
    const auto ds_batch_offset = karg.compute_ptr_offset_of_batch.GetDsPtrOffset(g_idx);
    const auto c_batch_offset  = karg.compute_ptr_offset_of_batch.GetCPtrOffset(g_idx);

    auto splitk_batch_offset = typename GridwiseGemm::SplitKBatchOffset(karg, k_idx);

    // populate pointer, desc for Ds
    static_for<0, GridwiseGemm::NumDTensor, 1>{}([&](auto i) {
        // D pointer
        karg.p_ds_grid(i) = karg.p_ds_grid(i) + ds_batch_offset[i];
    });

    GridwiseGemm::template Run<HasMainKBlockLoop, CGlobalMemoryDataOperation, TailNum>(
        karg.p_a_grid + a_batch_offset + splitk_batch_offset.a_k_split_offset,
        karg.p_b_grid + b_batch_offset + splitk_batch_offset.b_k_split_offset,
        karg.p_ds_grid,
        karg.p_c_grid + c_batch_offset,
        p_shared,
        karg,
        karg.a_element_op,
        karg.b_element_op,
        karg.c_element_op);
#else
    ignore = karg;
#endif // end of if (defined(__gfx9__))
}

template <typename GridwiseGemm,
          typename BatchedGemmArg,
          bool HasMainKBlockLoop,
          InMemoryDataOperationEnum CGlobalMemoryDataOperation,
          index_t MinimumOccupancy = 1,
          TailNumber TailNum       = TailNumber::Full>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, MinimumOccupancy)
#endif
        kernel_batched_gemm_xdl_cshuffle_v3_multi_d_2lds(BatchedGemmArg karg)
{
#if(!defined(__HIP_DEVICE_COMPILE__) || defined(__gfx9__))
    // Pass two lds pointer is the key to tell compiler that ds_read/write
    // operate on different lds chunk at same time without order dependecy
    __shared__ char p_shared_0[GridwiseGemm::GetSharedMemoryNumberOfByte()];
    __shared__ char p_shared_1[GridwiseGemm::GetSharedMemoryNumberOfByte()];

    const index_t g_idx = blockIdx.z % karg.Batch;
    const index_t k_idx = blockIdx.z / karg.Batch;

    const auto a_batch_offset  = karg.compute_ptr_offset_of_batch.GetAPtrOffset(g_idx);
    const auto b_batch_offset  = karg.compute_ptr_offset_of_batch.GetBPtrOffset(g_idx);
    const auto ds_batch_offset = karg.compute_ptr_offset_of_batch.GetDsPtrOffset(g_idx);
    const auto c_batch_offset  = karg.compute_ptr_offset_of_batch.GetCPtrOffset(g_idx);

    auto splitk_batch_offset = typename GridwiseGemm::SplitKBatchOffset(karg, k_idx);

    // populate pointer, desc for Ds
    static_for<0, GridwiseGemm::NumDTensor, 1>{}([&](auto i) {
        // D pointer
        karg.p_ds_grid(i) = karg.p_ds_grid(i) + ds_batch_offset[i];
    });

    GridwiseGemm::template Run_2Lds<HasMainKBlockLoop, CGlobalMemoryDataOperation, TailNum>(
        karg.p_a_grid + a_batch_offset + splitk_batch_offset.a_k_split_offset,
        karg.p_b_grid + b_batch_offset + splitk_batch_offset.b_k_split_offset,
        karg.p_ds_grid,
        karg.p_c_grid + c_batch_offset,
        p_shared_0,
        p_shared_1,
        karg,
        karg.a_element_op,
        karg.b_element_op,
        karg.c_element_op);
#else
    ignore = karg;
#endif // end of if (defined(__gfx9__))
}

namespace tensor_operation {
namespace device {

template <typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename CLayout,
          typename ADataType,
          typename BDataType,
          typename DsDataType,
          typename CDataType,
          typename GemmAccDataType,
          typename CShuffleDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          GemmSpecialization GemmSpec,
          index_t BlockSize,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t AK1,
          index_t BK1,
          index_t MPerXDL,
          index_t NPerXDL,
          index_t MXdlPerWave,
          index_t NXdlPerWave,
          typename ABlockTransferThreadClusterLengths_AK0_M_AK1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          index_t ABlockTransferSrcVectorDim,
          index_t ABlockTransferSrcScalarPerVector,
          index_t ABlockTransferDstScalarPerVector_AK1,
          bool ABlockLdsExtraM,
          typename BBlockTransferThreadClusterLengths_BK0_N_BK1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          index_t BBlockTransferSrcVectorDim,
          index_t BBlockTransferSrcScalarPerVector,
          index_t BBlockTransferDstScalarPerVector_BK1,
          bool BBlockLdsExtraN,
          index_t CShuffleMXdlPerWavePerShuffle,
          index_t CShuffleNXdlPerWavePerShuffle,
          typename CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          typename CDEShuffleBlockTransferScalarPerVectors,
          BlockGemmPipelineScheduler BlkGemmPipeSched = BlockGemmPipelineScheduler::Intrawave,
          BlockGemmPipelineVersion BlkGemmPipelineVer = BlockGemmPipelineVersion::v1,
          typename ComputeTypeA                       = ADataType,
          typename ComputeTypeB                       = BDataType,
          typename LDSTypeA                           = ComputeTypeA,
          typename LDSTypeB                           = ComputeTypeB>
struct DeviceBatchedGemmMultiD_Xdl_CShuffle_V3
    : public DeviceBatchedGemmV2MultiD<ALayout,
                                       BLayout,
                                       DsLayout,
                                       CLayout,
                                       ADataType,
                                       BDataType,
                                       DsDataType,
                                       CDataType,
                                       AElementwiseOperation,
                                       BElementwiseOperation,
                                       CElementwiseOperation>
{
    static constexpr index_t NumDTensor = DsDataType::Size();

    // GridwiseGemm
    using GridwiseGemm = GridwiseGemmMultiD_xdl_cshuffle_v3<
        ALayout,
        BLayout,
        DsLayout,
        CLayout,
        ADataType,
        BDataType,
        GemmAccDataType,
        CShuffleDataType,
        DsDataType,
        CDataType,
        AElementwiseOperation,
        BElementwiseOperation,
        CElementwiseOperation,
        GemmSpec,
        BlockSize,
        MPerBlock,
        NPerBlock,
        KPerBlock,
        AK1,
        BK1,
        MPerXDL,
        NPerXDL,
        MXdlPerWave,
        NXdlPerWave,
        ABlockTransferThreadClusterLengths_AK0_M_AK1,
        ABlockTransferThreadClusterArrangeOrder,
        ABlockTransferSrcAccessOrder,
        ABlockTransferSrcVectorDim,
        ABlockTransferSrcScalarPerVector,
        ABlockTransferDstScalarPerVector_AK1,
        false,
        ABlockLdsExtraM,
        BBlockTransferThreadClusterLengths_BK0_N_BK1,
        BBlockTransferThreadClusterArrangeOrder,
        BBlockTransferSrcAccessOrder,
        BBlockTransferSrcVectorDim,
        BBlockTransferSrcScalarPerVector,
        BBlockTransferDstScalarPerVector_BK1,
        false,
        BBlockLdsExtraN,
        CShuffleMXdlPerWavePerShuffle,
        CShuffleNXdlPerWavePerShuffle,
        CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        CDEShuffleBlockTransferScalarPerVectors,
        BlkGemmPipeSched,
        BlkGemmPipelineVer,
        ComputeTypeA,
        ComputeTypeB,
        LDSTypeA,
        LDSTypeB>;

    struct ComputePtrOffsetOfStridedBatch
    {
        ComputePtrOffsetOfStridedBatch(index_t BatchStrideA,
                                       index_t BatchStrideB,
                                       std::array<ck::index_t, NumDTensor> BatchStrideDs,
                                       index_t BatchStrideC)
            : BatchStrideA_(BatchStrideA),
              BatchStrideB_(BatchStrideB),
              BatchStrideDs_(BatchStrideDs),
              BatchStrideC_(BatchStrideC)
        {
        }

        __host__ __device__ constexpr long_index_t GetAPtrOffset(index_t g_idx) const
        {
            return static_cast<long_index_t>(BatchStrideA_) * g_idx;
        }

        __host__ __device__ constexpr long_index_t GetBPtrOffset(index_t g_idx) const
        {
            return static_cast<long_index_t>(BatchStrideB_) * g_idx;
        }

        __host__ __device__ constexpr auto GetDsPtrOffset(index_t g_idx) const
        {
            std::array<long_index_t, NumDTensor> ds_offset_;

            static_for<0, GridwiseGemm::NumDTensor, 1>{}([&](auto i) {
                ds_offset_[i] = static_cast<long_index_t>(BatchStrideDs_[i]) * g_idx;
            });

            return ds_offset_;
        }

        __host__ __device__ constexpr long_index_t GetCPtrOffset(index_t g_idx) const
        {
            return static_cast<long_index_t>(BatchStrideC_) * g_idx;
        }

        private:
        index_t BatchStrideA_;
        index_t BatchStrideB_;
        const std::array<ck::index_t, NumDTensor> BatchStrideDs_;
        index_t BatchStrideC_;
    };

    struct Argument : public GridwiseGemm::Argument
    {
        index_t Batch;
        ComputePtrOffsetOfStridedBatch compute_ptr_offset_of_batch;

        Argument(const ADataType* p_a_grid_,
                 const BDataType* p_b_grid_,
                 std::array<const void*, NumDTensor> p_ds_grid_,
                 CDataType* p_e_grid_,
                 index_t M_,
                 index_t N_,
                 index_t K_,
                 index_t StrideA_,
                 index_t StrideB_,
                 std::array<index_t, NumDTensor> StrideDs_,
                 index_t StrideE_,
                 index_t BatchStrideA_,
                 index_t BatchStrideB_,
                 const std::array<ck::index_t, NumDTensor>& BatchStrideDs_,
                 index_t BatchStrideE_,
                 index_t Batch_,
                 AElementwiseOperation a_element_op_,
                 BElementwiseOperation b_element_op_,
                 CElementwiseOperation c_element_op_,
                 index_t KBatch_)
            : GridwiseGemm::Argument{p_a_grid_,
                                     p_b_grid_,
                                     p_ds_grid_,
                                     p_e_grid_,
                                     M_,
                                     N_,
                                     K_,
                                     StrideA_,
                                     StrideB_,
                                     StrideDs_,
                                     StrideE_,
                                     KBatch_,
                                     a_element_op_,
                                     b_element_op_,
                                     c_element_op_},
              Batch{Batch_},
              compute_ptr_offset_of_batch{
                  BatchStrideA_, BatchStrideB_, BatchStrideDs_, BatchStrideE_}
        {
        }
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            if(stream_config.log_level_ > 0)
            {
                arg.Print();
            }

            if(!GridwiseGemm::CheckValidity(arg))
            {
                throw std::runtime_error("wrong! GridwiseGemm has invalid setting");
            }

            index_t gdx, gdy, gdz;
            std::tie(gdx, gdy, gdz) =
                GridwiseGemm::CalculateGridSize(arg.M, arg.N, arg.Batch * arg.KBatch);

            float ave_time = 0;

            index_t k_grain = arg.KBatch * KPerBlock;
            index_t K_split = (arg.K + k_grain - 1) / k_grain * KPerBlock;

            const bool has_main_k_block_loop = GridwiseGemm::CalculateHasMainKBlockLoop(K_split);

            const auto Run = [&](const auto& kernel) {
                if(stream_config.flush_cache)
                {

                    std::array<std::size_t, NumDTensor> DsSize;

                    Argument arg_ = arg;

                    const auto a_grid_desc_ak0_m_ak1 = GridwiseGemm::MakeAGridDescriptor_AK0_M_AK1(
                        arg_.M, arg_.MPadded, arg_.K, arg_.KPadded, arg_.StrideA, arg_.AK0);
                    const auto b_grid_desc_bk0_n_bk1 = GridwiseGemm::MakeBGridDescriptor_BK0_N_BK1(
                        arg_.K, arg_.KPadded, arg_.N, arg_.NPadded, arg_.StrideB, arg_.BK0);

                    auto size_a_buffer =
                        a_grid_desc_ak0_m_ak1.GetElementSpaceSize() * sizeof(ADataType) * arg.Batch;
                    auto size_b_buffer =
                        b_grid_desc_bk0_n_bk1.GetElementSpaceSize() * sizeof(BDataType) * arg.Batch;

                    const auto ds_grid_desc_m_n = GridwiseGemm::MakeDsGridDescriptor_M_N(
                        arg_.M, arg_.MPadded, arg_.N, arg_.NPadded, arg_.StrideDs);

                    static_for<0, NumDTensor, 1>{}([&](auto i) {
                        using DDataType = remove_cvref_t<tuple_element_t<i.value, DsDataType>>;
                        DsSize[i] = ds_grid_desc_m_n[i].GetElementSpaceSize() * sizeof(DDataType);
                    });
                    ck::utility::RotatingMemWrapperMultiD<Argument, DsDataType> rotating_mem(
                        arg_, stream_config.rotating_count, size_a_buffer, size_b_buffer, DsSize);
                    rotating_mem.Print();

                    auto run_flush_cache = [&]() {
                        // flush icache
                        ck::utility::flush_icache();
                        // rotating mem
                        rotating_mem.Next();
                        // clear c mem
                        if(arg_.KBatch > 1)
                            hipGetErrorString(
                                hipMemsetAsync(arg_.p_c_grid,
                                               0,
                                               arg.Batch * arg_.M * arg_.N * sizeof(CDataType),
                                               stream_config.stream_id_));
                    };

                    ave_time = ck::utility::launch_and_time_kernel_with_preprocess<false>(
                        stream_config,
                        run_flush_cache,
                        kernel,
                        dim3(gdx, gdy, gdz),
                        dim3(BlockSize),
                        0,
                        arg_);
                }
                else
                {
                    if(arg.KBatch > 1)
                        hipGetErrorString(hipMemsetAsync(arg.p_c_grid,
                                                         0,
                                                         arg.M * arg.N * sizeof(CDataType),
                                                         stream_config.stream_id_));

                    ave_time = launch_and_time_kernel(
                        stream_config, kernel, dim3(gdx, gdy, gdz), dim3(BlockSize), 0, arg);
                }
            };

            constexpr index_t minimum_occupancy =
                BlkGemmPipeSched == BlockGemmPipelineScheduler::Intrawave ? 1 : 2;

            if(has_main_k_block_loop)
            {
                // Tail number always full
                if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v1 ||
                             BlkGemmPipelineVer == BlockGemmPipelineVersion::v3)
                {
                    if(arg.KBatch > 1)
                    {
                        const auto kernel = kernel_batched_gemm_xdl_cshuffle_v3_multi_d<
                            GridwiseGemm,
                            Argument,
                            true,
                            InMemoryDataOperationEnum::AtomicAdd,
                            minimum_occupancy>;
                        Run(kernel);
                    }
                    else
                    {
                        const auto kernel = kernel_batched_gemm_xdl_cshuffle_v3_multi_d<
                            GridwiseGemm,
                            Argument,
                            true,
                            InMemoryDataOperationEnum::Set,
                            minimum_occupancy>;
                        Run(kernel);
                    }
                }
                // Tail number could be One to Seven
                else if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v2)
                {
                    if(arg.KBatch > 1)
                    {
                        if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::One)
                        {
                            const auto kernel = kernel_batched_gemm_xdl_cshuffle_v3_multi_d<
                                GridwiseGemm,
                                Argument,
                                true,
                                InMemoryDataOperationEnum::AtomicAdd,
                                minimum_occupancy,
                                TailNumber::One>;
                            Run(kernel);
                        }
                        else if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) ==
                                TailNumber::Full)
                        {
                            const auto kernel = kernel_batched_gemm_xdl_cshuffle_v3_multi_d<
                                GridwiseGemm,
                                Argument,
                                true,
                                InMemoryDataOperationEnum::AtomicAdd,
                                minimum_occupancy,
                                TailNumber::Full>;
                            Run(kernel);
                        }

                        if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 2)
                        {
                            if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::Two)
                            {
                                const auto kernel = kernel_batched_gemm_xdl_cshuffle_v3_multi_d<
                                    GridwiseGemm,
                                    Argument,
                                    true,
                                    InMemoryDataOperationEnum::AtomicAdd,
                                    minimum_occupancy,
                                    TailNumber::Two>;
                                Run(kernel);
                            }
                        }

                        if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 3)
                        {
                            if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) ==
                               TailNumber::Three)
                            {
                                const auto kernel = kernel_batched_gemm_xdl_cshuffle_v3_multi_d<
                                    GridwiseGemm,
                                    Argument,
                                    true,
                                    InMemoryDataOperationEnum::AtomicAdd,
                                    minimum_occupancy,
                                    TailNumber::Three>;
                                Run(kernel);
                            }
                        }

                        if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 4)
                        {
                            if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) ==
                               TailNumber::Four)
                            {
                                const auto kernel = kernel_batched_gemm_xdl_cshuffle_v3_multi_d<
                                    GridwiseGemm,
                                    Argument,
                                    true,
                                    InMemoryDataOperationEnum::AtomicAdd,
                                    minimum_occupancy,
                                    TailNumber::Four>;
                                Run(kernel);
                            }
                        }

                        if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 5)
                        {
                            if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) ==
                               TailNumber::Five)
                            {
                                const auto kernel = kernel_batched_gemm_xdl_cshuffle_v3_multi_d<
                                    GridwiseGemm,
                                    Argument,
                                    true,
                                    InMemoryDataOperationEnum::AtomicAdd,
                                    minimum_occupancy,
                                    TailNumber::Five>;
                                Run(kernel);
                            }
                        }

                        if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 6)
                        {
                            if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::Six)
                            {
                                const auto kernel = kernel_batched_gemm_xdl_cshuffle_v3_multi_d<
                                    GridwiseGemm,
                                    Argument,
                                    true,
                                    InMemoryDataOperationEnum::AtomicAdd,
                                    minimum_occupancy,
                                    TailNumber::Six>;
                                Run(kernel);
                            }
                        }

                        if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 7)
                        {
                            if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) ==
                               TailNumber::Seven)
                            {
                                const auto kernel = kernel_batched_gemm_xdl_cshuffle_v3_multi_d<
                                    GridwiseGemm,
                                    Argument,
                                    true,
                                    InMemoryDataOperationEnum::AtomicAdd,
                                    minimum_occupancy,
                                    TailNumber::Seven>;
                                Run(kernel);
                            }
                        }
                    }
                    else
                    {
                        if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::One)
                        {
                            const auto kernel = kernel_batched_gemm_xdl_cshuffle_v3_multi_d<
                                GridwiseGemm,
                                Argument,
                                true,
                                InMemoryDataOperationEnum::Set,
                                minimum_occupancy,
                                TailNumber::One>;
                            Run(kernel);
                        }
                        else if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) ==
                                TailNumber::Full)
                        {
                            const auto kernel = kernel_batched_gemm_xdl_cshuffle_v3_multi_d<
                                GridwiseGemm,
                                Argument,
                                true,
                                InMemoryDataOperationEnum::Set,
                                minimum_occupancy,
                                TailNumber::Full>;
                            Run(kernel);
                        }

                        if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 2)
                        {
                            if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::Two)
                            {
                                const auto kernel = kernel_batched_gemm_xdl_cshuffle_v3_multi_d<
                                    GridwiseGemm,
                                    Argument,
                                    true,
                                    InMemoryDataOperationEnum::Set,
                                    minimum_occupancy,
                                    TailNumber::Two>;
                                Run(kernel);
                            }
                        }

                        if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 3)
                        {
                            if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) ==
                               TailNumber::Three)
                            {
                                const auto kernel = kernel_batched_gemm_xdl_cshuffle_v3_multi_d<
                                    GridwiseGemm,
                                    Argument,
                                    true,
                                    InMemoryDataOperationEnum::Set,
                                    minimum_occupancy,
                                    TailNumber::Three>;
                                Run(kernel);
                            }
                        }

                        if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 4)
                        {
                            if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) ==
                               TailNumber::Four)
                            {
                                const auto kernel = kernel_batched_gemm_xdl_cshuffle_v3_multi_d<
                                    GridwiseGemm,
                                    Argument,
                                    true,
                                    InMemoryDataOperationEnum::Set,
                                    minimum_occupancy,
                                    TailNumber::Four>;
                                Run(kernel);
                            }
                        }

                        if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 5)
                        {
                            if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) ==
                               TailNumber::Five)
                            {
                                const auto kernel = kernel_batched_gemm_xdl_cshuffle_v3_multi_d<
                                    GridwiseGemm,
                                    Argument,
                                    true,
                                    InMemoryDataOperationEnum::Set,
                                    minimum_occupancy,
                                    TailNumber::Five>;
                                Run(kernel);
                            }
                        }

                        if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 6)
                        {
                            if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::Six)
                            {
                                const auto kernel = kernel_batched_gemm_xdl_cshuffle_v3_multi_d<
                                    GridwiseGemm,
                                    Argument,
                                    true,
                                    InMemoryDataOperationEnum::Set,
                                    minimum_occupancy,
                                    TailNumber::Six>;
                                Run(kernel);
                            }
                        }

                        if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 7)
                        {
                            if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) ==
                               TailNumber::Seven)
                            {
                                const auto kernel = kernel_batched_gemm_xdl_cshuffle_v3_multi_d<
                                    GridwiseGemm,
                                    Argument,
                                    true,
                                    InMemoryDataOperationEnum::Set,
                                    minimum_occupancy,
                                    TailNumber::Seven>;
                                Run(kernel);
                            }
                        }
                    }
                }
                // Tail number could be Odd or Even
                else if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v4)
                {
                    if(arg.KBatch > 1)
                    {
                        if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::Odd)
                        {
                            const auto kernel = kernel_batched_gemm_xdl_cshuffle_v3_multi_d_2lds<
                                GridwiseGemm,
                                Argument,
                                true,
                                InMemoryDataOperationEnum::AtomicAdd,
                                minimum_occupancy,
                                TailNumber::Odd>;
                            Run(kernel);
                        }
                        else
                        {
                            const auto kernel = kernel_batched_gemm_xdl_cshuffle_v3_multi_d_2lds<
                                GridwiseGemm,
                                Argument,
                                true,
                                InMemoryDataOperationEnum::AtomicAdd,
                                minimum_occupancy,
                                TailNumber::Even>;
                            Run(kernel);
                        }
                    }
                    else
                    {
                        if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::Odd)
                        {
                            const auto kernel = kernel_batched_gemm_xdl_cshuffle_v3_multi_d_2lds<
                                GridwiseGemm,
                                Argument,
                                true,
                                InMemoryDataOperationEnum::Set,
                                minimum_occupancy,
                                TailNumber::Odd>;
                            Run(kernel);
                        }
                        else
                        {
                            const auto kernel = kernel_batched_gemm_xdl_cshuffle_v3_multi_d_2lds<
                                GridwiseGemm,
                                Argument,
                                true,
                                InMemoryDataOperationEnum::Set,
                                minimum_occupancy,
                                TailNumber::Even>;
                            Run(kernel);
                        }
                    }
                }
                else
                {
                    if(arg.KBatch > 1)
                    {
                        if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::Odd)
                        {
                            const auto kernel = kernel_batched_gemm_xdl_cshuffle_v3_multi_d<
                                GridwiseGemm,
                                Argument,
                                true,
                                InMemoryDataOperationEnum::AtomicAdd,
                                minimum_occupancy,
                                TailNumber::Odd>;
                            Run(kernel);
                        }
                        else
                        {
                            const auto kernel = kernel_batched_gemm_xdl_cshuffle_v3_multi_d<
                                GridwiseGemm,
                                Argument,
                                true,
                                InMemoryDataOperationEnum::AtomicAdd,
                                minimum_occupancy,
                                TailNumber::Even>;
                            Run(kernel);
                        }
                    }
                    else
                    {
                        if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::Odd)
                        {
                            const auto kernel = kernel_batched_gemm_xdl_cshuffle_v3_multi_d<
                                GridwiseGemm,
                                Argument,
                                true,
                                InMemoryDataOperationEnum::Set,
                                minimum_occupancy,
                                TailNumber::Odd>;
                            Run(kernel);
                        }
                        else
                        {
                            const auto kernel = kernel_batched_gemm_xdl_cshuffle_v3_multi_d<
                                GridwiseGemm,
                                Argument,
                                true,
                                InMemoryDataOperationEnum::Set,
                                minimum_occupancy,
                                TailNumber::Even>;
                            Run(kernel);
                        }
                    }
                }
            }
            else
            {
                // Tail number always 1
                if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v1)
                {
                    if(arg.KBatch > 1)
                    {
                        const auto kernel = kernel_batched_gemm_xdl_cshuffle_v3_multi_d<
                            GridwiseGemm,
                            Argument,
                            false,
                            InMemoryDataOperationEnum::AtomicAdd,
                            minimum_occupancy>;
                        Run(kernel);
                    }
                    else
                    {
                        const auto kernel = kernel_batched_gemm_xdl_cshuffle_v3_multi_d<
                            GridwiseGemm,
                            Argument,
                            false,
                            InMemoryDataOperationEnum::Set,
                            minimum_occupancy>;
                        Run(kernel);
                    }
                }
            }

            return ave_time;
        }

        // polymorphic
        float Run(const BaseArgument* p_arg,
                  const StreamConfig& stream_config = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), stream_config);
        }
    };

    static constexpr bool IsValidCompilationParameter()
    {
        // TODO: properly implement this check
        return true;
    }

    static bool IsSupportedArgument(const Argument& arg)
    {
        if(!ck::is_xdl_supported())
        {
            return false;
        }

        if(!is_bf16_atomic_supported() && std::is_same_v<CDataType, ck::bhalf_t> && arg.KBatch > 1)
        {
            return false;
        }

        if((arg.K % AK1 != 0 || arg.K % BK1 != 0) && !(GemmSpec == GemmSpecialization::MKPadding ||
                                                       GemmSpec == GemmSpecialization::NKPadding ||
                                                       GemmSpec == GemmSpecialization::MNKPadding ||
                                                       GemmSpec == GemmSpecialization::KPadding))
        {
            return false;
        }

        return GridwiseGemm::CheckValidity(arg);
    }

    // polymorphic
    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static auto MakeArgument(const void* p_a,
                             const void* p_b,
                             std::array<const void*, NumDTensor> p_ds,
                             void* p_e,
                             index_t M,
                             index_t N,
                             index_t K,
                             index_t Batch,
                             index_t StrideA,
                             index_t StrideB,
                             std::array<index_t, NumDTensor> StrideDs,
                             index_t StrideE,
                             index_t BatchStrideA,
                             index_t BatchStrideB,
                             const std::array<ck::index_t, NumDTensor>& BatchStrideDs,
                             index_t BatchStrideE,
                             AElementwiseOperation a_element_op,
                             BElementwiseOperation b_element_op,
                             CElementwiseOperation c_element_op,
                             index_t KBatch = 1)
    {
        return Argument{static_cast<const ADataType*>(p_a),
                        static_cast<const BDataType*>(p_b),
                        p_ds,
                        static_cast<CDataType*>(p_e),
                        M,
                        N,
                        K,
                        StrideA,
                        StrideB,
                        StrideDs,
                        StrideE,
                        BatchStrideA,
                        BatchStrideB,
                        BatchStrideDs,
                        BatchStrideE,
                        Batch,
                        a_element_op,
                        b_element_op,
                        c_element_op,
                        KBatch};
    }

    static auto MakeInvoker() { return Invoker{}; }

    // polymorphic
    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_a,
                        const void* p_b,
                        const std::array<const void*, NumDTensor>& p_ds,
                        void* p_e,
                        index_t M,
                        index_t N,
                        index_t K,
                        index_t Batch,
                        index_t StrideA,
                        index_t StrideB,
                        const std::array<ck::index_t, NumDTensor>& StrideDs,
                        index_t StrideE,
                        index_t BatchStrideA,
                        index_t BatchStrideB,
                        const std::array<ck::index_t, NumDTensor>& BatchStrideDs,
                        index_t BatchStrideE,
                        AElementwiseOperation a_element_op,
                        BElementwiseOperation b_element_op,
                        CElementwiseOperation c_element_op,
                        index_t KBatch = 1) override
    {
        return std::make_unique<Argument>(static_cast<const ADataType*>(p_a),
                                          static_cast<const BDataType*>(p_b),
                                          p_ds,
                                          static_cast<CDataType*>(p_e),
                                          M,
                                          N,
                                          K,
                                          StrideA,
                                          StrideB,
                                          StrideDs,
                                          StrideE,
                                          BatchStrideA,
                                          BatchStrideB,
                                          BatchStrideDs,
                                          BatchStrideE,
                                          Batch,
                                          a_element_op,
                                          b_element_op,
                                          c_element_op,
                                          KBatch);
    }

    // polymorphic
    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    // polymorphic
    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        std::map<BlockGemmPipelineScheduler, std::string> BlkGemmPipelineSchedulerToString{
            {BlockGemmPipelineScheduler::Intrawave, "Intrawave"},
            {BlockGemmPipelineScheduler::Interwave, "Interwave"}};

        std::map<BlockGemmPipelineVersion, std::string> BlkGemmPipelineVersionToString{
            {BlockGemmPipelineVersion::v1, "v1"},
            {BlockGemmPipelineVersion::v2, "v2"},
            {BlockGemmPipelineVersion::v3, "v3"},
            {BlockGemmPipelineVersion::v4, "v4"},
            {BlockGemmPipelineVersion::v5, "v5"}};

        // clang-format off
        str << "DeviceBatchedGemmXdlUniversal"
            << "<"
            << getGemmSpecializationString(GemmSpec) << ", "
            << std::string(ALayout::name)[0]
            << std::string(BLayout::name)[0]
            << std::string(CLayout::name)[0]
            << ">"
            << " BlkSize: "
            << BlockSize << ", "
            << "BlkTile: "
            << MPerBlock<<"x"<<NPerBlock<<"x"<<KPerBlock << ", "
            << "WaveTile: "
            << MPerXDL<<"x"<<NPerXDL << ", "
            << "WaveMap: "
            << MXdlPerWave<<"x" << NXdlPerWave<<", "
            << "VmemReadVec: "
            << ABlockTransferSrcScalarPerVector<<"x"<<BBlockTransferSrcScalarPerVector<<", "
            << "BlkGemmPipelineScheduler: "
            << BlkGemmPipelineSchedulerToString[BlkGemmPipeSched] << ", "
            << "BlkGemmPipelineVersion: "
            << BlkGemmPipelineVersionToString[BlkGemmPipelineVer] << ", "
            << "BlkGemmPipelinePrefetchStages: "
            << GridwiseGemm::BlockwiseGemmPipe::PrefetchStages;
        // clang-format on

        return str.str();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
