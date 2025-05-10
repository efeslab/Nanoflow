// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <sstream>
#include <tuple>

#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"
#include "ck/host_utility/hip_check_error.hpp"
#include "ck/host_utility/stream_utility.hpp"
#include "ck/utility/common_header.hpp"
#include "ck/utility/loop_scheduler.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_gemm_tile_loop.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include <ck/tensor_operation/gpu/grid/block_to_ctile_map.hpp>
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_multiple_d_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_pipeline_selector.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

///
/// @brief      Entry point kernel for device-wide Grouped GEMM operation.
///
/// @param[in]  gemm_descs_const  The pointer to the array of GEMM descriptor structures.
/// @param[in]  group_count       The number of together processed GEMMs.
///
/// @tparam     GridwiseGemm                The specific GridwiseGEMM algorithm implementation.
/// @tparam     GemmDesc                    The structure holding all necessary descriptors and
///                                         other data needed for grouped gemm calculation and work
///                                         distribution.
/// @tparam     LocalBlock2ETileMap         The structure providing mapping between workgroup ids,
///                                         the data tiles to process and the output tiles.
///
template <typename GridwiseGemm,
          typename GemmDesc,
          GemmSpecialization GemmSpec,
          typename DsDataType,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          typename OffsettedBlockToCTileMap,
          typename LocalBlock2ETileMap,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
        kernel_grouped_gemm_multiple_d_xdl(const void CK_CONSTANT_ADDRESS_SPACE* gemm_descs_const,
                                           const index_t group_count,
                                           const AElementwiseOperation a_element_op,
                                           const BElementwiseOperation b_element_op,
                                           const CDEElementwiseOperation cde_element_op)
{
#if(!defined(__HIP_DEVICE_COMPILE__) || defined(__gfx908__) || defined(__gfx90a__) || \
    defined(__gfx94__))

    constexpr index_t shared_size = GridwiseGemm::GetSharedMemoryNumberOfByte();
    __shared__ uint8_t p_shared[shared_size];

    const auto gemm_desc_ptr =
        reinterpret_cast<const GemmDesc*>(cast_pointer_to_generic_address_space(gemm_descs_const));

    constexpr auto NumDTensor = DsDataType::Size();
    index_t tile_id           = get_block_1d_id();
    index_t tile_offset       = 0;
    index_t group_id          = -1;
    index_t group_offset      = 0;
    index_t grid_size_grp     = 0;

    index_t gemm_tile_id_start = 0;
    index_t gemm_tile_id_end   = 0;

    using AGridDescMK =
        remove_cvref_t<decltype(GridwiseGemm::template MakeAGridDescriptor_M_K<ALayout, GemmSpec>(
            1, 1, 1))>;
    using BGridDescNK =
        remove_cvref_t<decltype(GridwiseGemm::template MakeBGridDescriptor_N_K<BLayout, GemmSpec>(
            1, 1, 1))>;
    using EGridDescMN =
        remove_cvref_t<decltype(GridwiseGemm::template MakeEGridDescriptor_M_N<ELayout, GemmSpec>(
            1, 1, 1))>;
    using DsGridDescMN =
        remove_cvref_t<decltype(GridwiseGemm::template MakeDsGridDescriptor_M_N<DsLayout, GemmSpec>(
            {}, {}, {}))>;

    index_t M = 0, N = 0, K = 0;
    index_t StrideA, StrideB, StrideE;
    std::array<index_t, NumDTensor> StrideDs;

    AGridDescMK a_grid_desc_mk;
    BGridDescNK b_grid_desc_nk;
    EGridDescMN e_grid_desc_mn;
    DsGridDescMN ds_grid_desc_mn;
    auto b2c_tile_map = OffsettedBlockToCTileMap(LocalBlock2ETileMap(1, 1), 1, 1);

    do
    {
        // Find corresponding GEMM group for our tile
        while(!(tile_id >= gemm_tile_id_start && tile_id < gemm_tile_id_end) &&
              group_id < group_count)
        {
            group_offset += grid_size_grp;
            group_id++;

            if(group_id >= group_count)
                return;

            M = gemm_desc_ptr[group_id].M;
            N = gemm_desc_ptr[group_id].N;
            K = gemm_desc_ptr[group_id].K;

            if(M * N * K == 0)
            {
                grid_size_grp = 0;
                continue;
            }

            b2c_tile_map =
                OffsettedBlockToCTileMap(LocalBlock2ETileMap(M, N), group_offset, tile_offset);
            grid_size_grp = b2c_tile_map.CalculateGridSize(M, N);

            gemm_tile_id_start = group_offset;
            gemm_tile_id_end   = group_offset + grid_size_grp;
        }

        StrideA  = gemm_desc_ptr[group_id].StrideA;
        StrideB  = gemm_desc_ptr[group_id].StrideB;
        StrideDs = gemm_desc_ptr[group_id].StrideDs;
        StrideE  = gemm_desc_ptr[group_id].StrideE;

        a_grid_desc_mk =
            GridwiseGemm::template MakeAGridDescriptor_M_K<ALayout, GemmSpec>(M, K, StrideA);
        b_grid_desc_nk =
            GridwiseGemm::template MakeBGridDescriptor_N_K<BLayout, GemmSpec>(K, N, StrideB);
        e_grid_desc_mn =
            GridwiseGemm::template MakeEGridDescriptor_M_N<ELayout, GemmSpec>(M, N, StrideE);

        static_for<0, NumDTensor, 1>{}([&](auto j) {
            using DLayout      = remove_cvref_t<tuple_element_t<j.value, DsLayout>>;
            ds_grid_desc_mn(j) = GridwiseGemm::template MakeEGridDescriptor_M_N<DLayout, GemmSpec>(
                M, N, StrideDs[j]);
        });

        using DsGridPointer = decltype(GridwiseGemm::MakeDsGridPointer());
        DsGridPointer p_ds_grid;

        static_for<0, NumDTensor, 1>{}([&](auto i) {
            using DDataType = remove_cvref_t<tuple_element_t<i.value, DsDataType>>;
            p_ds_grid(i)    = static_cast<const DDataType*>(gemm_desc_ptr[group_id].p_ds_grid[i]);
        });

        bool has_main_kblock_loop =
            GridwiseGemm::CalculateHasMainKBlockLoop(a_grid_desc_mk.GetLength(Number<1>{}));
        // Update tile offset if we have moved within group
        b2c_tile_map.UpdateTileOffset(tile_offset);

        if(has_main_kblock_loop)
        {
            GridwiseGemm::template Run<true>(gemm_desc_ptr[group_id].p_a_grid,
                                             gemm_desc_ptr[group_id].p_b_grid,
                                             p_ds_grid,
                                             gemm_desc_ptr[group_id].p_e_grid,
                                             static_cast<void*>(p_shared),
                                             a_element_op,
                                             b_element_op,
                                             cde_element_op,
                                             a_grid_desc_mk,
                                             b_grid_desc_nk,
                                             ds_grid_desc_mn,
                                             e_grid_desc_mn,
                                             b2c_tile_map);
        }
        else
        {
            GridwiseGemm::template Run<false>(gemm_desc_ptr[group_id].p_a_grid,
                                              gemm_desc_ptr[group_id].p_b_grid,
                                              p_ds_grid,
                                              gemm_desc_ptr[group_id].p_e_grid,
                                              static_cast<void*>(p_shared),
                                              a_element_op,
                                              b_element_op,
                                              cde_element_op,
                                              a_grid_desc_mk,
                                              b_grid_desc_nk,
                                              ds_grid_desc_mn,
                                              e_grid_desc_mn,
                                              b2c_tile_map);
        }

        tile_id += get_grid_size();
        tile_offset += get_grid_size();

    } while(group_id < group_count);
#else
    ignore = gemm_descs_const;
    ignore = group_count;
    ignore = a_element_op;
    ignore = b_element_op;
    ignore = cde_element_op;
#endif // end of if (defined(__gfx908__) || defined(__gfx90a__))
}

template <typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename CShuffleDataType,
          typename DsDataType,
          typename EDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation,
          GemmSpecialization GemmSpec,
          ck::index_t NumGemmKPrefetchStage,
          ck::index_t BlockSize,
          ck::index_t MPerBlock,
          ck::index_t NPerBlock,
          ck::index_t KPerBlock,
          ck::index_t AK1,
          ck::index_t BK1,
          ck::index_t MPerXDL,
          ck::index_t NPerXDL,
          ck::index_t MXdlPerWave,
          ck::index_t NXdlPerWave,
          typename ABlockTransferThreadClusterLengths_AK0_M_AK1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          index_t ABlockTransferSrcVectorDim,
          index_t ABlockTransferSrcScalarPerVector,
          index_t ABlockTransferDstScalarPerVector_AK1,
          index_t ABlockLdsExtraM,
          typename BBlockTransferThreadClusterLengths_BK0_N_BK1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          index_t BBlockTransferSrcVectorDim,
          index_t BBlockTransferSrcScalarPerVector,
          index_t BBlockTransferDstScalarPerVector_BK1,
          index_t BBlockLdsExtraN,
          index_t CShuffleMXdlPerWavePerShuffle,
          index_t CShuffleNXdlPerWavePerShuffle,
          typename CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          index_t CDEShuffleBlockTransferScalarPerVector_NPerBlock,
          PipelineVersion PipelineVer = PipelineVersion::v1,
          LoopScheduler LoopSched     = make_default_loop_scheduler(),
          typename ComputeDataType    = EDataType>
struct DeviceGroupedGemmMultipleDXdlCShuffleTileLoop
    : public DeviceGroupedGemmTileLoop<ALayout,
                                       BLayout,
                                       DsLayout,
                                       ELayout,
                                       ADataType,
                                       BDataType,
                                       DsDataType,
                                       EDataType,
                                       AElementwiseOperation,
                                       BElementwiseOperation,
                                       CDEElementwiseOperation>
{
    using DeviceOp                      = DeviceGroupedGemmMultipleDXdlCShuffleTileLoop;
    static constexpr index_t NumDTensor = DsDataType::Size();

    using GridwiseGemm = GridwiseGemmMultipleD_xdl_cshuffle<
        ADataType,
        BDataType,
        ComputeDataType,
        AccDataType,
        CShuffleDataType,
        DsDataType,
        EDataType,
        AElementwiseOperation,
        BElementwiseOperation,
        CDEElementwiseOperation,
        InMemoryDataOperationEnum::Set,
        NumGemmKPrefetchStage,
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
        false, // AThreadTransferSrcResetCoordinateAfterRun,
        ABlockLdsExtraM,
        BBlockTransferThreadClusterLengths_BK0_N_BK1,
        BBlockTransferThreadClusterArrangeOrder,
        BBlockTransferSrcAccessOrder,
        BBlockTransferSrcVectorDim,
        BBlockTransferSrcScalarPerVector,
        BBlockTransferDstScalarPerVector_BK1,
        false, // BThreadTransferSrcResetCoordinateAfterRun,
        BBlockLdsExtraN,
        CShuffleMXdlPerWavePerShuffle,
        CShuffleNXdlPerWavePerShuffle,
        CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        CDEShuffleBlockTransferScalarPerVector_NPerBlock,
        LoopSched,
        PipelineVer>;

    template <typename UnderlyingBlockToCTileMap>
    struct OffsettedBlockToCTileMap
    {
        using underlying_type = UnderlyingBlockToCTileMap;

        __host__ __device__ OffsettedBlockToCTileMap(UnderlyingBlockToCTileMap block_to_ctile_map,
                                                     index_t group_offset,
                                                     index_t tile_offset)
            : block_to_ctile_map_{block_to_ctile_map},
              group_offset_{group_offset},
              tile_offset_{tile_offset}
        {
        }

        template <typename TopIdx>
        __host__ __device__ constexpr auto CalculateBottomIndex(const TopIdx& idx_top) const
        {
            return block_to_ctile_map_.CalculateBottomIndex(
                make_multi_index(idx_top[Number<0>{}] + tile_offset_ - group_offset_));
        }

        template <typename CTileIdx, typename CTileDim>
        __host__ __device__ bool ValidCTileIndex(const CTileIdx& c_tile_idx,
                                                 const CTileDim& c_tile_dim) const
        {
            return block_to_ctile_map_.ValidCTileIndex(c_tile_idx, c_tile_dim);
        }

        template <typename CGridDesc_M_N>
        __host__ constexpr bool CheckValidity(const CGridDesc_M_N& c_grid_desc_m_n) const
        {
            return block_to_ctile_map_.CheckValidity(c_grid_desc_m_n);
        }

        __host__ __device__ constexpr index_t CalculateGridSize(index_t M, index_t N) const
        {
            return block_to_ctile_map_.CalculateGridSize(M, N);
        }

        __device__ void UpdateTileOffset(index_t offset) { tile_offset_ = offset; }
        UnderlyingBlockToCTileMap block_to_ctile_map_;
        index_t group_offset_;
        index_t tile_offset_;
    };

    using KernelArguments             = GroupedGemmTileLoopKernelArguments<NumDTensor>;
    using Block2ETileMap              = BlockToCTileMap_N00_M0_N01Adapt<MPerBlock, NPerBlock>;
    using OffsetedLocalBlock2ETileMap = OffsettedBlockToCTileMap<Block2ETileMap>;

    // Argument
    struct Argument : public BaseArgument
    {
        Argument(std::vector<const void*>& /* p_As */,
                 std::vector<const void*>& /* p_Bs */,
                 std::vector<std::array<const void*, NumDTensor>>& /* p_Ds */,
                 std::vector<void*>& /* p_Es */,
                 const std::vector<GemmDesc>& gemm_descs,
                 AElementwiseOperation a_element_op,
                 BElementwiseOperation b_element_op,
                 CDEElementwiseOperation cde_element_op,
                 int occupancy_num_blocks,
                 int gpu_cu_count)
            : group_count_{static_cast<index_t>(gemm_descs.size())},
              occupancy_num_blocks_{occupancy_num_blocks},
              gpu_cu_count_{gpu_cu_count},
              gemm_descs_{gemm_descs},
              a_element_op_{a_element_op},
              b_element_op_{b_element_op},
              cde_element_op_{cde_element_op},
              tile_count_{0}
        {
            for(const auto& desc : gemm_descs)
            {
                const auto M            = desc.M_;
                const auto N            = desc.N_;
                const auto b2c_tile_map = Block2ETileMap(M, N);
                tile_count_ += b2c_tile_map.CalculateGridSize(M, N);
            }
        }

        index_t group_count_;
        const void* p_dev_gemm_args_;
        int occupancy_num_blocks_;
        int gpu_cu_count_;

        const std::vector<GemmDesc>& gemm_descs_;
        AElementwiseOperation a_element_op_;
        BElementwiseOperation b_element_op_;
        CDEElementwiseOperation cde_element_op_;
        index_t tile_count_;
    };

    struct KernelConfig
    {
        // The oversubscription factor for the number of blocks that can simultaneously reside on
        // GPU.
        static constexpr int BLOCK_SUBSCRIPTION_FACTOR = 1;
        static constexpr int BLOCK_WAVES               = BlockSize / get_warp_size();
        static constexpr int CU_SIMDS                  = 4;
        // Assume we want to have at most 2 waves per SIMD
        static constexpr int CU_BLOCKS = math::integer_divide_floor(2 * CU_SIMDS, BLOCK_WAVES);
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        ///
        /// @brief      Launch Grouped Gemm kernel.
        ///
        /// @note       This function overload is using user provided device buffer for kernel
        ///             arguments.
        ///
        /// @param[in]  arg                 The structure containing kernel arguments (in host
        ///                                 memory).
        /// @param[in]  dev_gemm_args       The pointer to device memory with kernel arguments.
        /// @param[in]  stream_config       The device stream configuration.
        ///
        /// @return     The average kernel execution time (if time measurement is enabled.)
        ///
        float Run(const Argument& arg,
                  const void* dev_gemm_args,
                  const StreamConfig& stream_config = StreamConfig{})
        {
            if(dev_gemm_args == nullptr)
            {
                std::ostringstream err;
                err << "The gemm arguments device buffer is not allocated!"
                    << " In " << __FILE__ << ":" << __LINE__ << ", in function: " << __func__;
                throw std::runtime_error(err.str());
            }

            float ave_time = 0;
            ave_time       = DispatchKernel(arg, dev_gemm_args, stream_config);

            return ave_time;
        }

        ///
        /// @brief      Launch Grouped Gemm kernel.
        ///
        /// @note       This function overload is using device buffers (for kernel arguments and
        ///             for kernel auxiliary workspace) provided with an argument. The user should
        ///             call @see GetDeviceKernelArgSize, and @see SetDeviceKernelArgs, on arg
        ///             parameter to properly allocate those buffers.
        ///
        /// @param[in]  arg            The structure containing kernel arguments (in host memory).
        /// @param[in]  stream_config  The device stream configuration.
        ///
        /// @return     The average kernel execution time (if time measurement is enabled.)
        ///
        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            if(arg.p_dev_gemm_args_ == nullptr)
            {
                std::ostringstream err;
                err << "The gemm arguments device buffer is not allocated!"
                    << " In " << __FILE__ << ":" << __LINE__ << ", in function: " << __func__;
                throw std::runtime_error(err.str());
            }

            return Run(arg, arg.p_dev_gemm_args_, stream_config);
        }

        float Run(const BaseArgument* p_arg,
                  const StreamConfig& stream_config = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), stream_config);
        }

        private:
        float DispatchKernel(const Argument& arg,
                             const void* dev_gemm_args,
                             const StreamConfig& stream_config) const
        {
            const auto kernel = kernel_grouped_gemm_multiple_d_xdl<GridwiseGemm,
                                                                   KernelArguments,
                                                                   GemmSpec,
                                                                   DsDataType,
                                                                   ALayout,
                                                                   BLayout,
                                                                   DsLayout,
                                                                   ELayout,
                                                                   OffsetedLocalBlock2ETileMap,
                                                                   Block2ETileMap,
                                                                   AElementwiseOperation,
                                                                   BElementwiseOperation,
                                                                   CDEElementwiseOperation>;
            return LaunchKernel(kernel, arg, dev_gemm_args, stream_config);
        }

        template <typename KernelFunction>
        int CalculateMaxOccupancyGridSize(const KernelFunction& kernel,
                                          const StreamConfig& stream_config) const
        {
            // Calculate max number of workgroups that can simultaneously reside on the CU.
            int occ_num_blocks            = 0;
            size_t dyn_shared_mem_per_blk = 0;
            hip_check_error(hipOccupancyMaxActiveBlocksPerMultiprocessor(
                &occ_num_blocks, kernel, BlockSize, dyn_shared_mem_per_blk));

            int cu_count = getAvailableComputeUnitCount(stream_config);

            if(stream_config.log_level_ > 0)
            {
                std::cout << "MaxActiveBlocksPerCU: " << occ_num_blocks
                          << ", available CUs count: " << cu_count << ", occup. grid size: "
                          << ck::math::min(occ_num_blocks, KernelConfig::CU_BLOCKS) * cu_count
                          << std::endl;
            }

            return cu_count * ck::math::min(occ_num_blocks, KernelConfig::CU_BLOCKS);
        }

        template <typename KernelFunction>
        float LaunchKernel(const KernelFunction& kernel,
                           const Argument& arg,
                           const void* dev_gemm_args,
                           const StreamConfig& stream_config) const
        {
            int grid_size = CalculateMaxOccupancyGridSize(kernel, stream_config);

            if(stream_config.log_level_ > 0)
            {
                std::cout << "grid_size: " << grid_size << " tile_count: " << arg.tile_count_
                          << std::endl;
            }

            return launch_and_time_kernel(stream_config,
                                          kernel,
                                          dim3(grid_size),
                                          dim3(BlockSize),
                                          0,
                                          cast_pointer_to_constant_address_space(dev_gemm_args),
                                          arg.group_count_,
                                          arg.a_element_op_,
                                          arg.b_element_op_,
                                          arg.cde_element_op_);
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

        using DsGridDescMN = remove_cvref_t<
            decltype(GridwiseGemm::template MakeDsGridDescriptor_M_N<DsLayout, GemmSpec>(
                {}, {}, {}))>;

        bool supported = true;

        for(const auto& gdesc : arg.gemm_descs_)
        {
            const auto M = gdesc.M_;
            const auto N = gdesc.N_;
            const auto K = gdesc.K_;

            const auto StrideA   = gdesc.stride_A_;
            const auto StrideB   = gdesc.stride_B_;
            const auto StrideE   = gdesc.stride_C_;
            const auto& StrideDs = gdesc.stride_Ds_;

            // If M dimension is unknown at launch time then validate just NK.
            // If N or K dim is zero (or unknown) then the vector loads responsibility lies on
            // the user.
            if(N * K == 0)
                continue;

            const auto a_grid_desc_mk =
                GridwiseGemm::template MakeAGridDescriptor_M_K<ALayout, GemmSpec>(M, K, StrideA);
            const auto b_grid_desc_nk =
                GridwiseGemm::template MakeBGridDescriptor_N_K<BLayout, GemmSpec>(K, N, StrideB);
            const auto e_grid_desc_mn =
                GridwiseGemm::template MakeEGridDescriptor_M_N<ELayout, GemmSpec>(M, N, StrideE);

            DsGridDescMN ds_grid_desc_mn;
            static_for<0, NumDTensor, 1>{}([&](auto j) {
                using DLayout = remove_cvref_t<tuple_element_t<j.value, DsLayout>>;
                ds_grid_desc_mn(j) =
                    GridwiseGemm::template MakeEGridDescriptor_M_N<DLayout, GemmSpec>(
                        M, N, StrideDs[j]);
            });

            const auto b2c_tile_map = Block2ETileMap(M, N);

            if(!(GridwiseGemm::template CheckValidity(a_grid_desc_mk,
                                                      b_grid_desc_nk,
                                                      ds_grid_desc_mn,
                                                      e_grid_desc_mn,
                                                      b2c_tile_map) &&
                 GridwiseGemm::template CheckTensorTransfersValidity<ALayout, BLayout, ELayout>(
                     M, N, K)))
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "The provided GEMM problem size (M,N,K) [" << M << "," << N << ","
                              << K << "] are not supported by current template parameters!"
                              << " In " << __FILE__ << ":" << __LINE__
                              << ", in function: " << __func__;
                }
                supported = false;
            }
        }

        return supported;
    }

    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static auto MakeArgument(std::vector<const void*>& p_As,
                             std::vector<const void*>& p_Bs,
                             std::vector<std::array<const void*, NumDTensor>>& p_Ds,
                             std::vector<void*>& p_Es,
                             std::vector<GemmDesc>& gemm_descs,
                             AElementwiseOperation a_elementwise_op,
                             BElementwiseOperation b_elementwise_op,
                             CDEElementwiseOperation cde_elementwise_op)
    {
        const auto kernel = kernel_grouped_gemm_multiple_d_xdl<GridwiseGemm,
                                                               KernelArguments,
                                                               GemmSpec,
                                                               DsDataType,
                                                               ALayout,
                                                               BLayout,
                                                               DsLayout,
                                                               ELayout,
                                                               OffsetedLocalBlock2ETileMap,
                                                               Block2ETileMap,
                                                               AElementwiseOperation,
                                                               BElementwiseOperation,
                                                               CDEElementwiseOperation>;
        int occupancy, num_cu;
        hip_check_error(
            hipOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, kernel, BlockSize, 0));

        hipDeviceProp_t dev_prop;
        hipDevice_t dev;
        hip_check_error(hipGetDevice(&dev));
        hip_check_error(hipGetDeviceProperties(&dev_prop, dev));
        num_cu = dev_prop.multiProcessorCount;

        return Argument{p_As,
                        p_Bs,
                        p_Ds,
                        p_Es,
                        gemm_descs,
                        a_elementwise_op,
                        b_elementwise_op,
                        cde_elementwise_op,
                        occupancy,
                        num_cu};
    }

    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(std::vector<const void*>& p_As,
                        std::vector<const void*>& p_Bs,
                        std::vector<std::array<const void*, NumDTensor>>& p_Ds,
                        std::vector<void*>& p_Es,
                        std::vector<GemmDesc>& gemm_descs,
                        AElementwiseOperation a_elementwise_op,
                        BElementwiseOperation b_elementwise_op,
                        CDEElementwiseOperation cde_elementwise_op) override
    {
        const auto kernel = kernel_grouped_gemm_multiple_d_xdl<GridwiseGemm,
                                                               KernelArguments,
                                                               GemmSpec,
                                                               DsDataType,
                                                               ALayout,
                                                               BLayout,
                                                               DsLayout,
                                                               ELayout,
                                                               OffsetedLocalBlock2ETileMap,
                                                               Block2ETileMap,
                                                               AElementwiseOperation,
                                                               BElementwiseOperation,
                                                               CDEElementwiseOperation>;
        int occupancy, num_cu;
        hip_check_error(
            hipOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, kernel, BlockSize, 0));

        hipDeviceProp_t dev_prop;
        hipDevice_t dev;
        hip_check_error(hipGetDevice(&dev));
        hip_check_error(hipGetDeviceProperties(&dev_prop, dev));
        num_cu = dev_prop.multiProcessorCount;

        return std::make_unique<Argument>(p_As,
                                          p_Bs,
                                          p_Ds,
                                          p_Es,
                                          gemm_descs,
                                          a_elementwise_op,
                                          b_elementwise_op,
                                          cde_elementwise_op,
                                          occupancy,
                                          num_cu);
    }

    static auto MakeInvoker() { return Invoker{}; }

    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    std::string GetTypeString() const override
    {
        auto str = std::ostringstream();

        // clang-format off
        str << "DeviceGroupedGemmMultipleDXdlCShuffleTileLoop"
            << "<"
            << std::string(ALayout::name)[0] << ","
            << std::string(BLayout::name)[0] << ","
            << std::string(ELayout::name)[0] << ","
            << BlockSize << ", "
            << MPerBlock << ", "
            << NPerBlock << ", "
            << KPerBlock << ", "
            << AK1 << ", "
            << BK1 << ", "
            << MPerXDL << ", "
            << NPerXDL << ", "
            << MXdlPerWave << ", "
            << NXdlPerWave << ", "
            << ABlockTransferSrcScalarPerVector << ", "
            << BBlockTransferSrcScalarPerVector << ", "
            << CShuffleMXdlPerWavePerShuffle << ", "
            << CShuffleNXdlPerWavePerShuffle << ", "
            << getGemmSpecializationString(GemmSpec) << ", "
            << PipelineVer << ", "
            << LoopSched
            << ">";
        // clang-format on

        return str.str();
    }

    void SetDeviceKernelArgs(Argument& arg, void* p_dev_kernel_args) const
    {
        arg.p_dev_gemm_args_ = p_dev_kernel_args;
    }

    void SetDeviceKernelArgs(BaseArgument* p_arg, void* p_dev_kernel_args) const override
    {
        return SetDeviceKernelArgs(*dynamic_cast<Argument*>(p_arg), p_dev_kernel_args);
    }

    size_t GetDeviceKernelArgSize(const BaseArgument* p_arg) const override
    {
        return dynamic_cast<const Argument*>(p_arg)->group_count_ * sizeof(KernelArguments);
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
