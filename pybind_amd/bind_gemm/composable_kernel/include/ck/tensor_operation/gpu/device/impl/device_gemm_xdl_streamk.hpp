// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <sstream>

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm_streamk.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_xdlops_streamk.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"
#include "ck/host_utility/hip_check_error.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename ADataType,
          typename BDataType,
          typename CDataType,
          typename AccDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          ck::index_t BlockSize,
          ck::index_t MPerBlock,
          ck::index_t NPerBlock,
          ck::index_t K0PerBlock,
          ck::index_t K1,
          ck::index_t MPerXDL,
          ck::index_t NPerXDL,
          ck::index_t MXdlPerWave,
          ck::index_t NXdlPerWave,
          typename ABlockTransferThreadClusterLengths_K0_M_K1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          ck::index_t ABlockTransferSrcVectorDim,
          ck::index_t ABlockTransferSrcScalarPerVector,
          ck::index_t ABlockTransferDstScalarPerVector_K1,
          ck::index_t ABlockLdsAddExtraM,
          typename BBlockTransferThreadClusterLengths_K0_N_K1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          ck::index_t BBlockTransferSrcVectorDim,
          ck::index_t BBlockTransferSrcScalarPerVector,
          ck::index_t BBlockTransferDstScalarPerVector_K1,
          ck::index_t BBlockLdsAddExtraN,
          index_t CShuffleMRepeatPerShuffle,
          index_t CShuffleNRepeatPerShuffle,
          typename CBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          index_t CBlockTransferScalarPerVector_NWaveNPerXDL>
struct DeviceGemmXdlStreamK : public DeviceGemmStreamK<ALayout,
                                                       BLayout,
                                                       CLayout,
                                                       ADataType,
                                                       BDataType,
                                                       CDataType,
                                                       AElementwiseOperation,
                                                       BElementwiseOperation,
                                                       CElementwiseOperation>
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};

    using GridwiseGemm = GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_streamk<
        BlockSize,
        BlockToCTileMap_GemmStreamK<MPerBlock,
                                    NPerBlock,
                                    K0PerBlock * K1,
                                    StreamKReductionStrategy::Atomic>,
        ADataType, // TODO: distinguish A/B datatype
        AccDataType,
        CDataType,
        ALayout,
        BLayout,
        CLayout,
        AElementwiseOperation,
        BElementwiseOperation,
        CElementwiseOperation,
        MPerBlock,
        NPerBlock,
        K0PerBlock,
        MPerXDL,
        NPerXDL,
        K1,
        MXdlPerWave,
        NXdlPerWave,
        ABlockTransferThreadClusterLengths_K0_M_K1,
        ABlockTransferThreadClusterArrangeOrder,
        ABlockTransferSrcAccessOrder,
        ABlockTransferSrcVectorDim,
        ABlockTransferSrcScalarPerVector,
        ABlockTransferDstScalarPerVector_K1,
        false, // AThreadTransferSrcResetCoordinateAfterRun,
        ABlockLdsAddExtraM,
        BBlockTransferThreadClusterLengths_K0_N_K1,
        BBlockTransferThreadClusterArrangeOrder,
        BBlockTransferSrcAccessOrder,
        BBlockTransferSrcVectorDim,
        BBlockTransferSrcScalarPerVector,
        BBlockTransferDstScalarPerVector_K1,
        false, // BThreadTransferSrcResetCoordinateAfterRun,
        BBlockLdsAddExtraN,
        CShuffleMRepeatPerShuffle,
        CShuffleNRepeatPerShuffle,
        CBlockTransferScalarPerVector_NWaveNPerXDL,
        CBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock>;

    using Argument = typename GridwiseGemm::Argument;

    // Invoker
    struct Invoker : public BaseInvoker
    {
        void Print(const Argument& karg) { karg.Print(); }

        float Run(const Argument& karg, const StreamConfig& stream_config = StreamConfig{})
        {
            if(stream_config.log_level_ > 0)
            {
                Print(karg);
            }
            if(!GridwiseGemm::CheckValidity(karg))
            {
                throw std::runtime_error(
                    "wrong! GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2 has invalid "
                    "setting");
            }

            dim3 grid_dims = karg.block_mapping.get_grid_dims();

            float ave_time = 0;

            const auto kernel = kernel_gemm_xdlops_streamk<GridwiseGemm>;

            // TODO: remove clear buffer for streamk kernels
            if constexpr(GridwiseGemm::Block2CTileMap::ReductionStrategy ==
                         StreamKReductionStrategy::Atomic)
            {
                hipGetErrorString(hipMemsetAsync(karg.p_c_grid,
                                                 0,
                                                 karg.M * karg.N * sizeof(CDataType),
                                                 stream_config.stream_id_));
                ave_time = launch_and_time_kernel(stream_config,
                                                  kernel,
                                                  grid_dims,
                                                  dim3(BlockSize),
                                                  0,
                                                  karg.p_a_grid,
                                                  karg.p_b_grid,
                                                  karg.p_c_grid,
                                                  karg.p_workspace_,
                                                  karg.M,
                                                  karg.N,
                                                  karg.K,
                                                  karg.StrideA,
                                                  karg.StrideB,
                                                  karg.StrideC,
                                                  karg.block_mapping);
            }
            else if constexpr(GridwiseGemm::Block2CTileMap::ReductionStrategy ==
                              StreamKReductionStrategy::Reduction)
            {
                char* workspace_semaphore = reinterpret_cast<char*>(karg.p_workspace_) +
                                            karg.block_mapping.get_workspace_size_for_acc(
                                                sizeof(typename GridwiseGemm::FloatAcc));
                auto preprocess = [&]() {
                    hipGetErrorString(
                        hipMemsetAsync(workspace_semaphore,
                                       0,
                                       karg.block_mapping.get_workspace_size_for_semaphore(),
                                       stream_config.stream_id_));
                };

                ave_time = launch_and_time_kernel_with_preprocess(stream_config,
                                                                  preprocess,
                                                                  kernel,
                                                                  grid_dims,
                                                                  dim3(BlockSize),
                                                                  0,
                                                                  karg.p_a_grid,
                                                                  karg.p_b_grid,
                                                                  karg.p_c_grid,
                                                                  karg.p_workspace_,
                                                                  karg.M,
                                                                  karg.N,
                                                                  karg.K,
                                                                  karg.StrideA,
                                                                  karg.StrideB,
                                                                  karg.StrideC,
                                                                  karg.block_mapping);
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

    size_t GetWorkSpaceSize(const BaseArgument* pArg) const override
    {
        const Argument* p_arg = dynamic_cast<const Argument*>(pArg);
        if constexpr(GridwiseGemm::Block2CTileMap::ReductionStrategy ==
                     StreamKReductionStrategy::Reduction)
        {
            return p_arg->block_mapping.get_workspace_size(sizeof(typename GridwiseGemm::FloatAcc));
        }
        else
        {
            return 0;
        }
    }

    void SetWorkSpacePointer(BaseArgument* pArg,
                             void* p_workspace,
                             const StreamConfig& = StreamConfig{}) const override
    {
        Argument* pArg_ = dynamic_cast<Argument*>(pArg);

        pArg_->p_workspace_ = p_workspace;
    }

    static constexpr bool IsValidCompilationParameter()
    {
        // TODO: properly implement this check
        return true;
    }

    static bool IsSupportedArgument(const Argument& karg)
    {
        if(!(ck::is_xdl_supported()))
        {
            return false;
        }
        return GridwiseGemm::CheckValidity(karg);
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
                             AElementwiseOperation,
                             BElementwiseOperation,
                             CElementwiseOperation,
                             uint32_t NumSKBlocks = 0xffffffff)
    {
        const auto kernel = kernel_gemm_xdlops_streamk<GridwiseGemm>;
        int occupancy, num_cu;
        hipError_t rtn;
        rtn = hipOccupancyMaxActiveBlocksPerMultiprocessor(
            &occupancy, kernel, BlockSize, GridwiseGemm::GetSharedMemoryNumberOfByte());
        hip_check_error(rtn);

        hipDeviceProp_t dev_prop;
        hipDevice_t dev;
        rtn = hipGetDevice(&dev);
        hip_check_error(rtn);
        rtn = hipGetDeviceProperties(&dev_prop, dev);
        hip_check_error(rtn);
        num_cu = dev_prop.multiProcessorCount;

        return Argument{p_a,
                        p_b,
                        p_c,
                        M,
                        N,
                        K,
                        StrideA,
                        StrideB,
                        StrideC,
                        static_cast<uint32_t>(num_cu),
                        static_cast<uint32_t>(occupancy),
                        NumSKBlocks};
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
                                                      AElementwiseOperation,
                                                      BElementwiseOperation,
                                                      CElementwiseOperation,
                                                      index_t NumSKBlocks = 0) override
    {
        const auto kernel = kernel_gemm_xdlops_streamk<GridwiseGemm>;
        int occupancy, num_cu;
        hipError_t rtn;
        rtn = hipOccupancyMaxActiveBlocksPerMultiprocessor(
            &occupancy, kernel, BlockSize, GridwiseGemm::GetSharedMemoryNumberOfByte());
        hip_check_error(rtn);

        hipDeviceProp_t dev_prop;
        hipDevice_t dev;
        rtn = hipGetDevice(&dev);
        hip_check_error(rtn);
        rtn = hipGetDeviceProperties(&dev_prop, dev);
        hip_check_error(rtn);
        num_cu = dev_prop.multiProcessorCount;

        return std::make_unique<Argument>(reinterpret_cast<const ADataType*>(p_a),
                                          reinterpret_cast<const BDataType*>(p_b),
                                          reinterpret_cast<CDataType*>(p_c),
                                          M,
                                          N,
                                          K,
                                          StrideA,
                                          StrideB,
                                          StrideC,
                                          static_cast<uint32_t>(num_cu),
                                          static_cast<uint32_t>(occupancy),
                                          static_cast<uint32_t>(NumSKBlocks));
    }

    // polymorphic
    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    // polymorphic
    std::string GetTypeString() const override { return GridwiseGemm::GetTypeString(); }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
