// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <sstream>

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm_v2.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_v3.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"
#include "ck/host_utility/flush_cache.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename ALayout,
          typename BLayout,
          typename CLayout,
          typename ADataType,
          typename BDataType,
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
          index_t CShuffleBlockTransferScalarPerVector_NPerBlock,
          BlockGemmPipelineScheduler BlkGemmPipeSched = BlockGemmPipelineScheduler::Intrawave,
          BlockGemmPipelineVersion BlkGemmPipelineVer = BlockGemmPipelineVersion::v1,
          typename ComputeTypeA                       = CDataType,
          typename ComputeTypeB                       = ComputeTypeA>
struct DeviceGemm_Xdl_CShuffleV3 : public DeviceGemmV2<ALayout,
                                                       BLayout,
                                                       CLayout,
                                                       ADataType,
                                                       BDataType,
                                                       CDataType,
                                                       AElementwiseOperation,
                                                       BElementwiseOperation,
                                                       CElementwiseOperation>
{
    // GridwiseGemm
    using GridwiseGemm = GridwiseGemm_xdl_cshuffle_v3<
        ALayout,
        BLayout,
        CLayout,
        ADataType,
        BDataType,
        GemmAccDataType,
        CShuffleDataType,
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
        CShuffleBlockTransferScalarPerVector_NPerBlock,
        BlkGemmPipeSched,
        BlkGemmPipelineVer,
        ComputeTypeA,
        ComputeTypeB>;

    using Argument = typename GridwiseGemm::Argument;

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
            std::tie(gdx, gdy, gdz) = GridwiseGemm::CalculateGridSize(arg.M, arg.N, arg.KBatch);

            float ave_time = 0;

            index_t k_grain = arg.KBatch * KPerBlock;
            index_t K_split = (arg.K + k_grain - 1) / k_grain * KPerBlock;

            const bool has_main_k_block_loop = GridwiseGemm::CalculateHasMainKBlockLoop(K_split);

            const auto Run = [&](const auto& kernel) {
                if(stream_config.flush_cache)
                {
                    Argument arg_ = arg;
                    ck::utility::RotatingMemWrapper<Argument> rotating_mem(
                        arg_,
                        stream_config.rotating_count,
                        arg_.M * arg_.K * sizeof(ADataType),
                        arg_.K * arg_.N * sizeof(BDataType));
                    rotating_mem.Print();

                    auto run_flush_cache = [&]() {
                        // flush icache
                        ck::utility::flush_icache();
                        // rotating mem
                        rotating_mem.Next();
                        // clear c mem
                        if constexpr(!is_same<remove_cvref_t<CDataType>, bhalf_t>::value)
                        {
                            if(arg_.KBatch > 1)
                                hipGetErrorString(
                                    hipMemsetAsync(arg_.p_c_grid,
                                                   0,
                                                   arg_.M * arg_.N * sizeof(CDataType),
                                                   stream_config.stream_id_));
                        }
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
                    if constexpr(!is_same<remove_cvref_t<CDataType>, bhalf_t>::value)
                    {
                        if(arg.KBatch > 1)
                            hipGetErrorString(hipMemsetAsync(arg.p_c_grid,
                                                             0,
                                                             arg.M * arg.N * sizeof(CDataType),
                                                             stream_config.stream_id_));
                    }

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
                        if constexpr(!is_same<remove_cvref_t<CDataType>, bhalf_t>::value)
                        {
                            const auto kernel =
                                kernel_gemm_xdl_cshuffle_v3<GridwiseGemm,
                                                            true,
                                                            InMemoryDataOperationEnum::AtomicAdd,
                                                            minimum_occupancy>;
                            Run(kernel);
                        }
                    }
                    else
                    {
                        const auto kernel =
                            kernel_gemm_xdl_cshuffle_v3<GridwiseGemm,
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
                        if constexpr(!is_same<remove_cvref_t<CDataType>, bhalf_t>::value)
                        {
                            if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::One)
                            {
                                const auto kernel = kernel_gemm_xdl_cshuffle_v3<
                                    GridwiseGemm,
                                    true,
                                    InMemoryDataOperationEnum::AtomicAdd,
                                    minimum_occupancy,
                                    TailNumber::One>;
                                Run(kernel);
                            }
                            else if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) ==
                                    TailNumber::Full)
                            {
                                const auto kernel = kernel_gemm_xdl_cshuffle_v3<
                                    GridwiseGemm,
                                    true,
                                    InMemoryDataOperationEnum::AtomicAdd,
                                    minimum_occupancy,
                                    TailNumber::Full>;
                                Run(kernel);
                            }

                            if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 2)
                            {
                                if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) ==
                                   TailNumber::Two)
                                {
                                    const auto kernel = kernel_gemm_xdl_cshuffle_v3<
                                        GridwiseGemm,
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
                                    const auto kernel = kernel_gemm_xdl_cshuffle_v3<
                                        GridwiseGemm,
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
                                    const auto kernel = kernel_gemm_xdl_cshuffle_v3<
                                        GridwiseGemm,
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
                                    const auto kernel = kernel_gemm_xdl_cshuffle_v3<
                                        GridwiseGemm,
                                        true,
                                        InMemoryDataOperationEnum::AtomicAdd,
                                        minimum_occupancy,
                                        TailNumber::Five>;
                                    Run(kernel);
                                }
                            }

                            if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 6)
                            {
                                if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) ==
                                   TailNumber::Six)
                                {
                                    const auto kernel = kernel_gemm_xdl_cshuffle_v3<
                                        GridwiseGemm,
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
                                    const auto kernel = kernel_gemm_xdl_cshuffle_v3<
                                        GridwiseGemm,
                                        true,
                                        InMemoryDataOperationEnum::AtomicAdd,
                                        minimum_occupancy,
                                        TailNumber::Seven>;
                                    Run(kernel);
                                }
                            }
                        }
                    }
                    else
                    {
                        if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::One)
                        {
                            const auto kernel =
                                kernel_gemm_xdl_cshuffle_v3<GridwiseGemm,
                                                            true,
                                                            InMemoryDataOperationEnum::Set,
                                                            minimum_occupancy,
                                                            TailNumber::One>;
                            Run(kernel);
                        }
                        else if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) ==
                                TailNumber::Full)
                        {
                            const auto kernel =
                                kernel_gemm_xdl_cshuffle_v3<GridwiseGemm,
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
                                const auto kernel =
                                    kernel_gemm_xdl_cshuffle_v3<GridwiseGemm,
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
                                const auto kernel =
                                    kernel_gemm_xdl_cshuffle_v3<GridwiseGemm,
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
                                const auto kernel =
                                    kernel_gemm_xdl_cshuffle_v3<GridwiseGemm,
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
                                const auto kernel =
                                    kernel_gemm_xdl_cshuffle_v3<GridwiseGemm,
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
                                const auto kernel =
                                    kernel_gemm_xdl_cshuffle_v3<GridwiseGemm,
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
                                const auto kernel =
                                    kernel_gemm_xdl_cshuffle_v3<GridwiseGemm,
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
                        if constexpr(!is_same<remove_cvref_t<CDataType>, bhalf_t>::value)
                        {
                            if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::Odd)
                            {
                                const auto kernel = kernel_gemm_xdl_cshuffle_v3_2lds<
                                    GridwiseGemm,
                                    true,
                                    InMemoryDataOperationEnum::AtomicAdd,
                                    minimum_occupancy,
                                    TailNumber::Odd>;
                                Run(kernel);
                            }
                            else
                            {
                                const auto kernel = kernel_gemm_xdl_cshuffle_v3_2lds<
                                    GridwiseGemm,
                                    true,
                                    InMemoryDataOperationEnum::AtomicAdd,
                                    minimum_occupancy,
                                    TailNumber::Even>;
                                Run(kernel);
                            }
                        }
                    }
                    else
                    {
                        if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::Odd)
                        {
                            const auto kernel =
                                kernel_gemm_xdl_cshuffle_v3_2lds<GridwiseGemm,
                                                                 true,
                                                                 InMemoryDataOperationEnum::Set,
                                                                 minimum_occupancy,
                                                                 TailNumber::Odd>;
                            Run(kernel);
                        }
                        else
                        {
                            const auto kernel =
                                kernel_gemm_xdl_cshuffle_v3_2lds<GridwiseGemm,
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
                        if constexpr(!is_same<remove_cvref_t<CDataType>, bhalf_t>::value)
                        {
                            if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::Odd)
                            {
                                const auto kernel = kernel_gemm_xdl_cshuffle_v3<
                                    GridwiseGemm,
                                    true,
                                    InMemoryDataOperationEnum::AtomicAdd,
                                    minimum_occupancy,
                                    TailNumber::Odd>;
                                Run(kernel);
                            }
                            else
                            {
                                const auto kernel = kernel_gemm_xdl_cshuffle_v3<
                                    GridwiseGemm,
                                    true,
                                    InMemoryDataOperationEnum::AtomicAdd,
                                    minimum_occupancy,
                                    TailNumber::Even>;
                                Run(kernel);
                            }
                        }
                    }
                    else
                    {
                        if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::Odd)
                        {
                            const auto kernel =
                                kernel_gemm_xdl_cshuffle_v3<GridwiseGemm,
                                                            true,
                                                            InMemoryDataOperationEnum::Set,
                                                            minimum_occupancy,
                                                            TailNumber::Odd>;
                            Run(kernel);
                        }
                        else
                        {
                            const auto kernel =
                                kernel_gemm_xdl_cshuffle_v3<GridwiseGemm,
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
                        if constexpr(!is_same<remove_cvref_t<CDataType>, bhalf_t>::value)
                        {
                            const auto kernel =
                                kernel_gemm_xdl_cshuffle_v3<GridwiseGemm,
                                                            false,
                                                            InMemoryDataOperationEnum::AtomicAdd,
                                                            minimum_occupancy>;
                            Run(kernel);
                        }
                    }
                    else
                    {
                        const auto kernel =
                            kernel_gemm_xdl_cshuffle_v3<GridwiseGemm,
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

    static auto MakeArgument(const ADataType* p_a,
                             const BDataType* p_b,
                             CDataType* p_c,
                             index_t M,
                             index_t N,
                             index_t K,
                             index_t StrideA,
                             index_t StrideB,
                             index_t StrideC,
                             index_t KBatch,
                             AElementwiseOperation,
                             BElementwiseOperation,
                             CElementwiseOperation)
    {
        return Argument{p_a, p_b, p_c, M, N, K, StrideA, StrideB, StrideC, KBatch};
    }

    static auto MakeInvoker() { return Invoker{}; }

    // polymorphic
    std::unique_ptr<BaseArgument> MakeArgumentPointer(const void* p_a,
                                                      const void* p_b,
                                                      void* p_c,
                                                      index_t M,
                                                      index_t N,
                                                      index_t K,
                                                      index_t StrideA,
                                                      index_t StrideB,
                                                      index_t StrideC,
                                                      index_t KBatch,
                                                      AElementwiseOperation,
                                                      BElementwiseOperation,
                                                      CElementwiseOperation) override
    {
        return std::make_unique<Argument>(static_cast<const ADataType*>(p_a),
                                          static_cast<const BDataType*>(p_b),
                                          static_cast<CDataType*>(p_c),
                                          M,
                                          N,
                                          K,
                                          StrideA,
                                          StrideB,
                                          StrideC,
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
        str << "DeviceGemmXdlUniversal"
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
