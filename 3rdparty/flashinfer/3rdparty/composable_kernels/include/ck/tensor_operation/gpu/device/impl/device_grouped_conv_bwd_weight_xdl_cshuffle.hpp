// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <numeric>
#include <sstream>

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_conv_bwd_weight.hpp"
#include "ck/tensor_operation/operator_transform/transform_conv_bwd_weight_to_gemm.hpp"
#include "ck/tensor_operation/gpu/device/convolution_backward_weight_specialization.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_xdlops_bwd_weight.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_utils.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename GridwiseGemm,
          typename FloatA,
          typename FloatB,
          typename FloatC,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          typename AGridDesc_B_K0_M_K1,
          typename BGridDesc_B_K0_N_K1,
          typename CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
          typename Block2CTileMap,
          typename ComputePtrOffsetOfBatch,
          bool HasMainKBlockLoop>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
        kernel_batched_gemm_xdlops_bwd_weight(
            const FloatA* __restrict__ p_a_grid,
            const FloatB* __restrict__ p_b_grid,
            FloatC* __restrict__ p_c_grid,
            const AElementwiseOperation a_element_op,
            const BElementwiseOperation b_element_op,
            const CElementwiseOperation c_element_op,
            const index_t batch_count,
            const AGridDesc_B_K0_M_K1 a_b_k0_m_k1_grid_desc,
            const BGridDesc_B_K0_N_K1 b_b_k0_n_k1_grid_desc,
            const CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock
                c_grid_desc_mblock_mperblock_nblock_nperblock,
            const Block2CTileMap block_2_ctile_map,
            const ComputePtrOffsetOfBatch compute_ptr_offset_of_batch)
{
#if(!defined(__HIP_DEVICE_COMPILE__) || defined(__gfx908__) || defined(__gfx90a__) || \
    defined(__gfx94__))
    const index_t num_blocks_per_batch =
        __builtin_amdgcn_readfirstlane(get_grid_size() / batch_count);
    const index_t g_idx = __builtin_amdgcn_readfirstlane(get_block_1d_id() / num_blocks_per_batch);

    const long_index_t a_batch_offset = __builtin_amdgcn_readfirstlane(
        static_cast<long_index_t>(compute_ptr_offset_of_batch.GetAPtrOffset(g_idx)));
    const long_index_t b_batch_offset = __builtin_amdgcn_readfirstlane(
        static_cast<long_index_t>(compute_ptr_offset_of_batch.GetBPtrOffset(g_idx)));
    const long_index_t c_batch_offset = __builtin_amdgcn_readfirstlane(
        static_cast<long_index_t>(compute_ptr_offset_of_batch.GetCPtrOffset(g_idx)));

    __shared__ FloatA p_shared[GridwiseGemm::GetSharedMemoryNumberOfByte() / sizeof(FloatA)];

    GridwiseGemm::template Run<HasMainKBlockLoop>(p_a_grid + a_batch_offset,
                                                  p_b_grid + b_batch_offset,
                                                  p_c_grid + c_batch_offset,
                                                  p_shared,
                                                  a_b_k0_m_k1_grid_desc,
                                                  b_b_k0_n_k1_grid_desc,
                                                  c_grid_desc_mblock_mperblock_nblock_nperblock,
                                                  a_element_op,
                                                  b_element_op,
                                                  c_element_op,
                                                  block_2_ctile_map);
#else
    ignore = p_a_grid;
    ignore = p_b_grid;
    ignore = p_c_grid;
    ignore = a_b_k0_m_k1_grid_desc;
    ignore = b_b_k0_n_k1_grid_desc;
    ignore = c_grid_desc_mblock_mperblock_nblock_nperblock;
    ignore = a_element_op;
    ignore = b_element_op;
    ignore = c_element_op;
    ignore = batch_count;
    ignore = block_2_ctile_map;
    ignore = compute_ptr_offset_of_batch;

    compute_ptr_offset_of_batch.GetAPtrOffset(0);
    compute_ptr_offset_of_batch.GetBPtrOffset(0);
    compute_ptr_offset_of_batch.GetCPtrOffset(0);
#endif // end of if (defined(__gfx908__) || defined(__gfx90a__))
}

// out[N, Ho, Wo, K] = in[N, Hi, Wi, C] * wei[K, Y, X, C]
template <ck::index_t NDimSpatial,
          typename InLayout,
          typename WeiLayout,
          typename OutLayout,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename AccDataType,
          typename InElementwiseOperation,
          typename WeiElementwiseOperation,
          typename OutElementwiseOperation,
          ConvolutionBackwardWeightSpecialization ConvBackwardWeightSpecialization,
          ck::index_t BlockSize,
          ck::index_t MPerBlock,
          ck::index_t NPerBlock,
          ck::index_t K0PerBlock,
          ck::index_t K1,
          ck::index_t MPerXdl,
          ck::index_t NPerXdl,
          ck::index_t MXdlPerWave,
          ck::index_t NXdlPerWave,
          typename ABlockTransferThreadClusterLengths_K0_M_K1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          ck::index_t ABlockTransferSrcVectorDim,
          ck::index_t ABlockTransferSrcScalarPerVector,
          ck::index_t ABlockTransferDstScalarPerVector_K1,
          bool ABlockLdsAddExtraM,
          typename BBlockTransferThreadClusterLengths_K0_N_K1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          ck::index_t BBlockTransferSrcVectorDim,
          ck::index_t BBlockTransferSrcScalarPerVector,
          ck::index_t BBlockTransferDstScalarPerVector_K1,
          bool BBlockLdsAddExtraN,
          index_t CShuffleMXdlPerWavePerShuffle,
          index_t CShuffleNXdlPerWavePerShuffle,
          typename CBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          index_t CBlockTransferScalarPerVector_NWaveNPerXdl,
          typename ComputeTypeA = InDataType,
          typename ComputeTypeB = ComputeTypeA>
struct DeviceGroupedConvBwdWeight_Xdl_CShuffle
    : public DeviceGroupedConvBwdWeight<NDimSpatial,
                                        InLayout,
                                        WeiLayout,
                                        OutLayout,
                                        InDataType,
                                        WeiDataType,
                                        OutDataType,
                                        InElementwiseOperation,
                                        WeiElementwiseOperation,
                                        OutElementwiseOperation,
                                        ComputeTypeA,
                                        ComputeTypeB>
{
    using DeviceOp = DeviceGroupedConvBwdWeight_Xdl_CShuffle;

    using ADataType = OutDataType;
    using BDataType = InDataType;
    using CDataType = WeiDataType;

    using AElementwiseOperation = OutElementwiseOperation;
    using BElementwiseOperation = InElementwiseOperation;
    using CElementwiseOperation = WeiElementwiseOperation;

    // TODO make A/B datatype different
    using ABDataType = InDataType;

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};
    static constexpr auto I5 = Number<5>{};

    static constexpr auto K1Number = Number<K1>{};

    static constexpr auto conv_to_gemm_transformer =
        TransformConvBwdWeightToGemm<NDimSpatial,
                                     MPerBlock,
                                     NPerBlock,
                                     K1Number,
                                     K0PerBlock,
                                     ConvBackwardWeightSpecialization>{};

    // Bytes per 32 lds bank: 32 * 4 bytes
    static constexpr auto BankLength = 128;
    static constexpr auto ElePerBank = BankLength / sizeof(ADataType);

    // M1 & M0
    static constexpr auto ABlockLdsM1PerBlock = ElePerBank / K1;
    static constexpr auto ABlockLdsM0PerBlock = MPerBlock / ABlockLdsM1PerBlock;
    static constexpr auto ABlockLdsM1Padding  = 4;

    // N1 & N0
    static constexpr auto BBlockLdsN1PerBlock = ElePerBank / K1;
    static constexpr auto BBlockLdsN0PerBlock = NPerBlock / BBlockLdsN1PerBlock;
    static constexpr auto BBlockLdsN1Padding  = 4;

    template <ck::index_t NDim, typename ck::enable_if<NDim == 1, bool>::type = false>
    static auto GetABCGridDesc()
    {
        const ck::index_t dim   = 1;
        const ck::index_t batch = 1;
        const std::array<ck::index_t, NDimSpatial> lengths{1};
        const std::array<ck::index_t, NDimSpatial + 3> strides{1, 1, 1, 1};
        const std::array<ck::index_t, NDimSpatial> params{1};
        return conv_to_gemm_transformer.template MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N<1>(
            dim,
            dim,
            dim,
            lengths,
            lengths,
            lengths,
            strides,
            strides,
            strides,
            params,
            params,
            params,
            params,
            batch);
    }

    template <ck::index_t NDim, typename ck::enable_if<NDim == 2, bool>::type = false>
    static auto GetABCGridDesc()
    {
        const ck::index_t dim   = 1;
        const ck::index_t batch = 1;
        const std::array<ck::index_t, NDimSpatial> lengths{1, 1};
        const std::array<ck::index_t, NDimSpatial + 3> strides{1, 1, 1, 1, 1};
        const std::array<ck::index_t, NDimSpatial> params{1, 1};
        return conv_to_gemm_transformer.template MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N<2>(
            dim,
            dim,
            dim,
            lengths,
            lengths,
            lengths,
            strides,
            strides,
            strides,
            params,
            params,
            params,
            params,
            batch);
    }

    template <ck::index_t NDim, typename ck::enable_if<NDim == 3, bool>::type = false>
    static auto GetABCGridDesc()
    {
        const ck::index_t dim   = 1;
        const ck::index_t batch = 1;
        const std::array<ck::index_t, NDimSpatial> lengths{1, 1, 1};
        const std::array<ck::index_t, NDimSpatial + 3> strides{1, 1, 1, 1, 1, 1};
        const std::array<ck::index_t, NDimSpatial> params{1, 1, 1};
        return conv_to_gemm_transformer.template MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N<3>(
            dim,
            dim,
            dim,
            lengths,
            lengths,
            lengths,
            strides,
            strides,
            strides,
            params,
            params,
            params,
            params,
            batch);
    }

    using ABCGridDescs = decltype(GetABCGridDesc<NDimSpatial>());

    using AGridDesc_K0_M_K1 = remove_cvref_t<decltype(ABCGridDescs{}[I0])>;
    using BGridDesc_K0_N_K1 = remove_cvref_t<decltype(ABCGridDescs{}[I1])>;
    using CGridDesc_M_N     = remove_cvref_t<decltype(ABCGridDescs{}[I2])>;

    using GridwiseGemm = GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_bwd_weight<
        BlockSize,
        ADataType,
        BDataType,
        AccDataType,
        CDataType,
        InMemoryDataOperationEnum::AtomicAdd,
        AGridDesc_K0_M_K1,
        BGridDesc_K0_N_K1,
        CGridDesc_M_N,
        AElementwiseOperation,
        BElementwiseOperation,
        CElementwiseOperation,
        MPerBlock,
        NPerBlock,
        K0PerBlock,
        MPerXdl,
        NPerXdl,
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
        ABlockLdsM1PerBlock,
        ABlockLdsM0PerBlock,
        ABlockLdsM1Padding,
        BBlockTransferThreadClusterLengths_K0_N_K1,
        BBlockTransferThreadClusterArrangeOrder,
        BBlockTransferSrcAccessOrder,
        BBlockTransferSrcVectorDim,
        BBlockTransferSrcScalarPerVector,
        BBlockTransferDstScalarPerVector_K1,
        false, // BThreadTransferSrcResetCoordinateAfterRun,
        BBlockLdsAddExtraN,
        BBlockLdsN1PerBlock,
        BBlockLdsN0PerBlock,
        BBlockLdsN1Padding,
        CShuffleMXdlPerWavePerShuffle,
        CShuffleNXdlPerWavePerShuffle,
        CBlockTransferScalarPerVector_NWaveNPerXdl,
        CBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        true,
        true,
        1,
        PipelineVersion::v1,
        ComputeTypeA,
        ComputeTypeB>;

    // Argument
    using CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock =
        decltype(GridwiseGemm::MakeCGridDesc_MBlock_MPerBlock_NBlock_NPerBlock(CGridDesc_M_N{}));

    using Block2CTileMap =
        decltype(GridwiseGemm::MakeCBlockClusterAdaptor(CGridDesc_M_N{}, 1, 1, 1));

    struct Argument : public BaseArgument
    {
        Argument(const InDataType* p_in_grid,
                 WeiDataType* p_wei_grid,
                 const OutDataType* p_out_grid,
                 const std::array<index_t, NDimSpatial + 3>& b_g_n_c_wis_lengths, // input
                 const std::array<index_t, NDimSpatial + 3>& b_g_n_c_wis_strides,
                 const std::array<index_t, NDimSpatial + 3>& e_g_k_c_xs_lengths, // weight
                 const std::array<index_t, NDimSpatial + 3>& e_g_k_c_xs_strides,
                 const std::array<index_t, NDimSpatial + 3>& a_g_n_k_wos_lengths, // output
                 const std::array<index_t, NDimSpatial + 3>& a_g_n_k_wos_strides,
                 const std::array<ck::index_t, NDimSpatial>& conv_filter_strides,
                 const std::array<ck::index_t, NDimSpatial>& conv_filter_dilations,
                 const std::array<ck::index_t, NDimSpatial>& input_left_pads,
                 const std::array<ck::index_t, NDimSpatial>& input_right_pads,
                 const ck::index_t M01,
                 const ck::index_t N01,
                 InElementwiseOperation in_element_op,
                 WeiElementwiseOperation wei_element_op,
                 OutElementwiseOperation out_element_op,
                 ck::index_t split_k)
            : p_a_grid_{p_out_grid},
              p_b_grid_{p_in_grid},
              p_c_grid_{p_wei_grid},
              a_grid_desc_kbatch_k0_m_k1_{},
              b_grid_desc_kbatch_k0_n_k1_{},
              c_grid_desc_m_n_{},
              c_grid_desc_mblock_mperblock_nblock_nperblock_{},
              block_2_ctile_map_{},
              compute_ptr_offset_of_batch_{},
              M01_{M01},
              N01_{N01},
              a_element_op_{out_element_op},
              b_element_op_{in_element_op},
              c_element_op_{wei_element_op},
              Conv_G_{b_g_n_c_wis_lengths[0]},
              Conv_N_{b_g_n_c_wis_lengths[1]},
              Conv_K_{e_g_k_c_xs_lengths[1]},
              Conv_C_{b_g_n_c_wis_lengths[2]},
              input_spatial_lengths_{},
              filter_spatial_lengths_{},
              output_spatial_lengths_{},
              conv_filter_strides_{conv_filter_strides},
              input_left_pads_{input_left_pads},
              input_right_pads_{input_right_pads},
              k_batch_{split_k}
        {
            constexpr index_t spatial_offset = 3;
            std::copy(begin(b_g_n_c_wis_lengths) + spatial_offset,
                      end(b_g_n_c_wis_lengths),
                      begin(input_spatial_lengths_));
            std::copy(begin(e_g_k_c_xs_lengths) + spatial_offset,
                      end(e_g_k_c_xs_lengths),
                      begin(filter_spatial_lengths_));
            std::copy(begin(a_g_n_k_wos_lengths) + spatial_offset,
                      end(a_g_n_k_wos_lengths),
                      begin(output_spatial_lengths_));

            const auto descs =
                conv_to_gemm_transformer
                    .template MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N<NDimSpatial>(
                        Conv_N_,
                        Conv_K_,
                        Conv_C_,
                        input_spatial_lengths_,
                        filter_spatial_lengths_,
                        output_spatial_lengths_,
                        b_g_n_c_wis_strides,
                        e_g_k_c_xs_strides,
                        a_g_n_k_wos_strides,
                        conv_filter_strides,
                        conv_filter_dilations,
                        input_left_pads,
                        input_right_pads,
                        k_batch_);

            a_grid_desc_kbatch_k0_m_k1_ = descs[I0];
            b_grid_desc_kbatch_k0_n_k1_ = descs[I1];
            c_grid_desc_m_n_            = descs[I2];

            block_2_ctile_map_ =
                GridwiseGemm::MakeCBlockClusterAdaptor(c_grid_desc_m_n_, M01, N01, k_batch_);

            // A/B/C Batch Stride
            compute_ptr_offset_of_batch_.BatchStrideA_ = a_g_n_k_wos_strides[0];
            compute_ptr_offset_of_batch_.BatchStrideB_ = b_g_n_c_wis_strides[0];
            compute_ptr_offset_of_batch_.BatchStrideC_ =
                Conv_K_ * Conv_C_ *
                std::accumulate(begin(filter_spatial_lengths_),
                                end(filter_spatial_lengths_),
                                index_t{1},
                                std::multiplies<>{});

            if(GridwiseGemm::CheckValidity(a_grid_desc_kbatch_k0_m_k1_,
                                           b_grid_desc_kbatch_k0_n_k1_,
                                           c_grid_desc_m_n_,
                                           block_2_ctile_map_))
            {
                c_grid_desc_mblock_mperblock_nblock_nperblock_ =
                    GridwiseGemm::MakeCGridDesc_MBlock_MPerBlock_NBlock_NPerBlock(c_grid_desc_m_n_);
            }
        }

        const ADataType* p_a_grid_;
        const BDataType* p_b_grid_;
        CDataType* p_c_grid_;
        AGridDesc_K0_M_K1 a_grid_desc_kbatch_k0_m_k1_;
        BGridDesc_K0_N_K1 b_grid_desc_kbatch_k0_n_k1_;
        CGridDesc_M_N c_grid_desc_m_n_;
        CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock c_grid_desc_mblock_mperblock_nblock_nperblock_;

        Block2CTileMap block_2_ctile_map_;

        // for computing batch offset
        ComputePtrOffsetOfStridedBatch<> compute_ptr_offset_of_batch_;

        index_t M01_;
        index_t N01_;

        OutElementwiseOperation a_element_op_;
        InElementwiseOperation b_element_op_;
        WeiElementwiseOperation c_element_op_;

        // for checking IsSupportedArgument()
        const index_t Conv_G_;
        const index_t Conv_N_;
        const index_t Conv_K_;
        const index_t Conv_C_;
        std::array<ck::index_t, NDimSpatial> input_spatial_lengths_;
        std::array<ck::index_t, NDimSpatial> filter_spatial_lengths_;
        std::array<ck::index_t, NDimSpatial> output_spatial_lengths_;
        const std::array<ck::index_t, NDimSpatial>& conv_filter_strides_;
        const std::array<ck::index_t, NDimSpatial>& input_left_pads_;
        const std::array<ck::index_t, NDimSpatial>& input_right_pads_;
        const index_t k_batch_;
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        using Argument = DeviceOp::Argument;

        void ShowInfo(const Argument& arg)
        {
            std::cout << "arg.a_grid_desc_kbatch_k0_m_k1_{"
                      << arg.a_grid_desc_kbatch_k0_m_k1_.GetLength(I0) << ", "
                      << arg.a_grid_desc_kbatch_k0_m_k1_.GetLength(I1) << ", "
                      << arg.a_grid_desc_kbatch_k0_m_k1_.GetLength(I2) << ", "
                      << arg.a_grid_desc_kbatch_k0_m_k1_.GetLength(I3) << "}" << std::endl;

            std::cout << "arg.b_grid_desc_kbatch_k0_n_k1_{"
                      << arg.b_grid_desc_kbatch_k0_n_k1_.GetLength(I0) << ", "
                      << arg.b_grid_desc_kbatch_k0_n_k1_.GetLength(I1) << ", "
                      << arg.b_grid_desc_kbatch_k0_n_k1_.GetLength(I2) << ", "
                      << arg.b_grid_desc_kbatch_k0_n_k1_.GetLength(I3) << "}" << std::endl;

            std::cout << "arg.c_grid_desc_m_n_{" << arg.c_grid_desc_m_n_.GetLength(I0) << ", "
                      << arg.c_grid_desc_m_n_.GetLength(I1) << "}" << std::endl;
        }

        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            if(!GridwiseGemm::CheckValidity(arg.a_grid_desc_kbatch_k0_m_k1_,
                                            arg.b_grid_desc_kbatch_k0_n_k1_,
                                            arg.c_grid_desc_m_n_,
                                            arg.block_2_ctile_map_))
            {
                throw std::runtime_error(
                    "wrong! GridwiseGemm_km_kn_m0m1n0n1_xdlops_v3r1 has invalid setting");
            }

            const index_t grid_size =
                arg.block_2_ctile_map_.CalculateGridSize(arg.c_grid_desc_m_n_) * arg.Conv_G_;

            const auto K0 = arg.a_grid_desc_kbatch_k0_m_k1_.GetLength(I1);

            const bool has_main_k0_block_loop = GridwiseGemm::CalculateHasMainK0BlockLoop(K0);

            auto launch_kernel = [&](auto has_main_k_block_loop) {
                constexpr bool has_main_loop = has_main_k_block_loop.value;

                const auto kernel = kernel_batched_gemm_xdlops_bwd_weight<
                    GridwiseGemm,
                    ADataType,
                    BDataType,
                    CDataType,
                    OutElementwiseOperation,
                    InElementwiseOperation,
                    WeiElementwiseOperation,
                    remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                    remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                    remove_reference_t<DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                    remove_reference_t<DeviceOp::Block2CTileMap>,
                    ComputePtrOffsetOfStridedBatch<>,
                    has_main_loop>;

                return launch_and_time_kernel(stream_config,
                                              kernel,
                                              dim3(grid_size),
                                              dim3(BlockSize),
                                              0,
                                              arg.p_a_grid_,
                                              arg.p_b_grid_,
                                              arg.p_c_grid_,
                                              arg.a_element_op_,
                                              arg.b_element_op_,
                                              arg.c_element_op_,
                                              arg.Conv_G_,
                                              arg.a_grid_desc_kbatch_k0_m_k1_,
                                              arg.b_grid_desc_kbatch_k0_n_k1_,
                                              arg.c_grid_desc_mblock_mperblock_nblock_nperblock_,
                                              arg.block_2_ctile_map_,
                                              arg.compute_ptr_offset_of_batch_);
            };

            if(has_main_k0_block_loop)
            {
                return launch_kernel(integral_constant<bool, true>{});
            }
            else
            {
                return launch_kernel(integral_constant<bool, false>{});
            }
        }

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
        if constexpr(NDimSpatial == 1)
        {
            if constexpr(!is_GNWK_GKXC_GNWC<InLayout, WeiLayout, OutLayout>())
            {
                return false;
            }
        }
        else if constexpr(NDimSpatial == 2)
        {
            if constexpr(!(is_NHWGK_GKYXC_NHWGC<InLayout, WeiLayout, OutLayout>() ||
                           is_GNHWK_GKYXC_GNHWC<InLayout, WeiLayout, OutLayout>()))
            {
                return false;
            }
        }
        else if constexpr(NDimSpatial == 3)
        {
            if constexpr(!(is_NDHWGK_GKZYXC_NDHWGC<InLayout, WeiLayout, OutLayout>() ||
                           is_GNDHWK_GKZYXC_GNDHWC<InLayout, WeiLayout, OutLayout>()))
            {
                return false;
            }
        }
        else
        {
            return false;
        }

        if constexpr(ConvBackwardWeightSpecialization ==
                     ConvolutionBackwardWeightSpecialization::Filter1x1Stride1Pad0)
        {
            // check if it's 1x1, stride=1 pad = 0 conv
            for(int i = 0; i < NDimSpatial; i++)
            {
                if(!(arg.filter_spatial_lengths_[i] == 1 && arg.conv_filter_strides_[i] == 1 &&
                     arg.input_left_pads_[i] == 0 && arg.input_right_pads_[i] == 0))
                {
                    return false;
                }
            }
        }

        // vector load A/B matrix from global memory
        if(!(ABlockTransferSrcVectorDim == 2 && BBlockTransferSrcVectorDim == 2 &&
             arg.Conv_K_ % ABlockTransferSrcScalarPerVector == 0 &&
             arg.Conv_C_ % BBlockTransferSrcScalarPerVector == 0))
        {
            return false;
        }

        // vector store C matrix into global memory
        if(!(arg.Conv_C_ % CBlockTransferScalarPerVector_NWaveNPerXdl == 0))
        {
            return false;
        }

        // Gridwise GEMM size
        return GridwiseGemm::CheckValidity(arg.a_grid_desc_kbatch_k0_m_k1_,
                                           arg.b_grid_desc_kbatch_k0_n_k1_,
                                           arg.c_grid_desc_m_n_,
                                           arg.block_2_ctile_map_);
    }

    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static auto
    MakeArgument(const InDataType* p_in_grid,
                 WeiDataType* p_wei_grid,
                 const OutDataType* p_out_grid,
                 const std::array<index_t, NDimSpatial + 3>& b_g_n_c_wis_lengths, // input
                 const std::array<index_t, NDimSpatial + 3>& b_g_n_c_wis_strides,
                 const std::array<index_t, NDimSpatial + 3>& e_g_k_c_xs_lengths, // weight
                 const std::array<index_t, NDimSpatial + 3>& e_g_k_c_xs_strides,
                 const std::array<index_t, NDimSpatial + 3>& a_g_n_k_wos_lengths, // output
                 const std::array<index_t, NDimSpatial + 3>& a_g_n_k_wos_strides,
                 const std::array<ck::index_t, NDimSpatial>& conv_filter_strides,
                 const std::array<ck::index_t, NDimSpatial>& conv_filter_dilations,
                 const std::array<ck::index_t, NDimSpatial>& input_left_pads,
                 const std::array<ck::index_t, NDimSpatial>& input_right_pads,
                 InElementwiseOperation in_element_op,
                 WeiElementwiseOperation wei_element_op,
                 OutElementwiseOperation out_element_op,
                 const ck::index_t split_k)
    {
        return Argument{p_in_grid,
                        p_wei_grid,
                        p_out_grid,
                        b_g_n_c_wis_lengths, // input
                        b_g_n_c_wis_strides,
                        e_g_k_c_xs_lengths, // weight
                        e_g_k_c_xs_strides,
                        a_g_n_k_wos_lengths, // output
                        a_g_n_k_wos_strides,
                        conv_filter_strides,
                        conv_filter_dilations,
                        input_left_pads,
                        input_right_pads,
                        1,
                        1,
                        in_element_op,
                        wei_element_op,
                        out_element_op,
                        split_k};
    }

    static auto MakeInvoker() { return Invoker{}; }

    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_in_grid,
                        void* p_wei_grid,
                        const void* p_out_grid,
                        const std::array<index_t, NDimSpatial + 3>& b_g_n_c_wis_lengths, // input
                        const std::array<index_t, NDimSpatial + 3>& b_g_n_c_wis_strides,
                        const std::array<index_t, NDimSpatial + 3>& e_g_k_c_xs_lengths, // weight
                        const std::array<index_t, NDimSpatial + 3>& e_g_k_c_xs_strides,
                        const std::array<index_t, NDimSpatial + 3>& a_g_n_k_wos_lengths, // output
                        const std::array<index_t, NDimSpatial + 3>& a_g_n_k_wos_strides,
                        const std::array<ck::index_t, NDimSpatial>& conv_filter_strides,
                        const std::array<ck::index_t, NDimSpatial>& conv_filter_dilations,
                        const std::array<ck::index_t, NDimSpatial>& input_left_pads,
                        const std::array<ck::index_t, NDimSpatial>& input_right_pads,
                        InElementwiseOperation in_element_op,
                        WeiElementwiseOperation wei_element_op,
                        OutElementwiseOperation out_element_op,
                        const ck::index_t split_k) override
    {
        return std::make_unique<Argument>(static_cast<const InDataType*>(p_in_grid),
                                          static_cast<WeiDataType*>(p_wei_grid),
                                          static_cast<const OutDataType*>(p_out_grid),
                                          b_g_n_c_wis_lengths, // input
                                          b_g_n_c_wis_strides,
                                          e_g_k_c_xs_lengths, // weight
                                          e_g_k_c_xs_strides,
                                          a_g_n_k_wos_lengths, // output
                                          a_g_n_k_wos_strides,
                                          conv_filter_strides,
                                          conv_filter_dilations,
                                          input_left_pads,
                                          input_right_pads,
                                          1,
                                          1,
                                          in_element_op,
                                          wei_element_op,
                                          out_element_op,
                                          split_k);
    }

    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        // clang-format off
        str << "DeviceGroupedConvBwdWeight_Xdl_CShuffle"
            << "<"
            << BlockSize << ", "
            << MPerBlock << ", "
            << NPerBlock << ", "
            << K0PerBlock << ", "
            << getConvBackwardWeightSpecializationString(ConvBackwardWeightSpecialization) << ", "
            << K1 << ", "
            << MXdlPerWave << ", "
            << NXdlPerWave << ", "
            << ABlockTransferSrcScalarPerVector << ", "
            << ABlockTransferDstScalarPerVector_K1 << ", "
            << BBlockTransferSrcScalarPerVector << ", "
            << BBlockTransferDstScalarPerVector_K1 << ", "
            << CShuffleMXdlPerWavePerShuffle << ", "
            << CShuffleNXdlPerWavePerShuffle << ", "
            << CBlockTransferScalarPerVector_NWaveNPerXdl
            << ">";
        // clang-format on

        return str.str();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
