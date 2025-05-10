// SPDX-License-Identifier: MIT
// Copyright (c) 2023-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <functional>
#include <iostream>
#include <iterator>
#include <numeric>
#include <sstream>

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/convolution_forward_specialization.hpp"
#include "ck/tensor_operation/operator_transform/transform_conv_fwd_to_gemm.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_conv_fwd_multiple_abd.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/matrix_padder.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_multiple_d_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_multiple_abd_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_utils.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"
#include "ck/host_utility/io.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

namespace {

/*
 * \brief Wrapper function of GridwiseGemm::Run to realize BatchedGEMM.
 *
 * \tparam ComputePtrOffsetOfBatch Class that computes the base pointer offsets of A, B, C matrix
 * given the batch. For example, ComputePtrOffsetOfStridedBatch() computes the offsets of evenly
 * strided batched, but we can easily extend to other layouts. The returned offset can be either \p
 * index_t or \p long_index_t. If it returns \p long_index_t, we are not subject to the 2GB
 * limitations.
 *
 * \tparam Block2ETileMap Block2ETileMap::CalculateBottomIndex() takes in id of a workgroup and
 * returns the 2D index of the tile that it computes. \see
 * GridwiseGemm_k0mk1_k0nk1_mn_xdlops_v2r3::Run().
 *
 * \note Using \p ComputePtrOffsetOfBatch gives us the flexibility that 2 workgroups can compute 2
 * tiles from different matrices. Keep in mind that these 2 matrices can share the same grid
 * descriptor (like in BatchedGEMM), or use their own grid descriptors (in GroupedGemm). \link
 * impl/device_conv3d_fwd_xdl_ndhwc_kzyxc_ndhwk.hpp kernel_gemm_xdlops_v2r3_for_conv3d \endlink for
 * \link DeviceConv3d \endlink uses the same concept, but currently does NOT encapsulate the
 * computing of pointer offset into \p ComputePtrOffsetOfStridedBatch.
 *
 * \note \p Block2ETileMap allows customized mapping between a workgroup and the C-tile it computes.
 * Together with \p ComputePtrOffsetOfBatch, we can reuse GridwiseGemm (and GridwiseGemm fusion ) to
 * realize BatchedGemm and GroupedGemm (and the corresponding GEMM fusion).
 *
 */
template <typename GridwiseGemm,
          typename AsPointer, // tuples if multi AB, pointers if no
          typename BsPointer,
          typename DsPointer,
          typename EDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation,
          typename AGridDesc_AK0_M_AK1,
          typename BGridDesc_BK0_N_BK1,
          typename DsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock,
          typename EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
          typename Block2ETileMap,
          typename ComputePtrOffsetOfBatch,
          bool HasMainKBlockLoop,
          bool isMultiA,
          bool isMultiB>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
        kernel_grouped_conv_fwd_multiple_abd_xdl_cshuffle(
            AsPointer p_as_grid,
            BsPointer p_bs_grid,
            DsPointer p_ds_grid,
            EDataType* __restrict__ p_e_grid,
            const AElementwiseOperation a_element_op,
            const BElementwiseOperation b_element_op,
            const CDEElementwiseOperation cde_element_op,
            const index_t batch_count,
            const AGridDesc_AK0_M_AK1 a_grid_desc_k0_m_k1,
            const BGridDesc_BK0_N_BK1 b_grid_desc_k0_n_k1,
            const DsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock
                ds_grid_desc_mblock_mperblock_nblock_nperblock,
            const EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock
                e_grid_desc_mblock_mperblock_nblock_nperblock_,
            const Block2ETileMap block_2_ctile_map,
            const ComputePtrOffsetOfBatch compute_ptr_offset_of_batch)
{
#if(!defined(__HIP_DEVICE_COMPILE__) || defined(__gfx908__) || defined(__gfx90a__) || \
    defined(__gfx94__))
    // offset base pointer for each work-group
    const index_t num_blocks_per_batch =
        __builtin_amdgcn_readfirstlane(get_grid_size() / batch_count);
    const index_t g_idx = __builtin_amdgcn_readfirstlane(get_block_1d_id() / num_blocks_per_batch);

    const long_index_t e_batch_offset = __builtin_amdgcn_readfirstlane(
        static_cast<long_index_t>(compute_ptr_offset_of_batch.GetEPtrOffset(g_idx)));
    const auto& ds_batch_offset = compute_ptr_offset_of_batch.GetDsPtrOffset(g_idx);

    __shared__ char p_shared[GridwiseGemm::GetSharedMemoryNumberOfByte()];

    DsPointer p_ds_grid_grp;

    static constexpr index_t NumDTensor =
        DsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock::Size();

    static_for<0, NumDTensor, 1>{}(
        [&](auto i) { p_ds_grid_grp(i) = p_ds_grid[i] + ds_batch_offset[i]; });

    if constexpr(isMultiA || isMultiB)
    {
        AsPointer p_as_grid_grp;
        BsPointer p_bs_grid_grp;

        const auto& as_batch_offset = compute_ptr_offset_of_batch.GetAsPtrOffset(g_idx);

        static constexpr index_t NumATensor = AGridDesc_AK0_M_AK1::Size();
        static_for<0, NumATensor, 1>{}(
            [&](auto i) { p_as_grid_grp(i) = p_as_grid[i] + as_batch_offset[i]; });

        const auto& bs_batch_offset = compute_ptr_offset_of_batch.GetBsPtrOffset(g_idx);

        static constexpr index_t NumBTensor = BGridDesc_BK0_N_BK1::Size();
        static_for<0, NumBTensor, 1>{}(
            [&](auto i) { p_bs_grid_grp(i) = p_bs_grid[i] + bs_batch_offset[i]; });

        GridwiseGemm::template Run<HasMainKBlockLoop>(
            p_as_grid_grp,
            p_bs_grid_grp,
            p_ds_grid_grp,
            p_e_grid + e_batch_offset,
            p_shared,
            a_element_op,
            b_element_op,
            cde_element_op,
            a_grid_desc_k0_m_k1,
            b_grid_desc_k0_n_k1,
            ds_grid_desc_mblock_mperblock_nblock_nperblock,
            e_grid_desc_mblock_mperblock_nblock_nperblock_,
            block_2_ctile_map);
    }
    else
    {
        const long_index_t a_batch_offset = __builtin_amdgcn_readfirstlane(
            static_cast<long_index_t>(compute_ptr_offset_of_batch.GetAPtrOffset(g_idx)));
        const long_index_t b_batch_offset = __builtin_amdgcn_readfirstlane(
            static_cast<long_index_t>(compute_ptr_offset_of_batch.GetBPtrOffset(g_idx)));

        GridwiseGemm::template Run<HasMainKBlockLoop>(
            p_as_grid + a_batch_offset,
            p_bs_grid + b_batch_offset,
            p_ds_grid_grp,
            p_e_grid + e_batch_offset,
            p_shared,
            a_element_op,
            b_element_op,
            cde_element_op,
            a_grid_desc_k0_m_k1,
            b_grid_desc_k0_n_k1,
            ds_grid_desc_mblock_mperblock_nblock_nperblock,
            e_grid_desc_mblock_mperblock_nblock_nperblock_,
            block_2_ctile_map);
    }
#else
    ignore = p_as_grid;
    ignore = p_bs_grid;
    ignore = p_ds_grid;
    ignore = p_e_grid;
    ignore = batch_count;
    ignore = a_grid_desc_k0_m_k1;
    ignore = b_grid_desc_k0_n_k1;
    ignore = ds_grid_desc_mblock_mperblock_nblock_nperblock;
    ignore = e_grid_desc_mblock_mperblock_nblock_nperblock_;
    ignore = a_element_op;
    ignore = b_element_op;
    ignore = cde_element_op;
    ignore = compute_ptr_offset_of_batch;
    ignore = block_2_ctile_map;
#endif
}

} // namespace

template <typename T>
using is_tuple = decltype(std::declval<T&>().IsTuple());

//
// @brief      Device Convolution operation.
//
// Supports:
//  @li         Forward convolution with up to 3 spatial dimentions
//  @li         Input tensor in GNWC data format
//  @li         Weight tensor in GKXC data format
//  @li         Output tensor in GNWK data format
//
// 1D:
// out[N, Wo, K] = in[N, Wi, C] * wei[K, X, C]
// 2D:
// out[N, Ho, Wo, K] = in[N, Hi, Wi, C] * wei[K, Y, X, C]
// 3D:
// out[N, Do, Ho, Wo, K] = in[N, Di, Hi, Wi, C] * wei[K, Z, Y, X, C]
//
template <index_t NDimSpatial,
          typename ALayout,
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
          ConvolutionForwardSpecialization ConvForwardSpecialization,
          GemmSpecialization GemmSpec,
          index_t NumGemmKPrefetchStage,
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
          index_t CDEBlockTransferScalarPerVector_NPerBlock,
          typename AComputeDataType =
              decltype(UnpackDataType<is_detected<is_tuple, ADataType>::value,
                                      Number<0>,
                                      ADataType>()), // ComputeType is InputType by default (first
                                                     // in tuple for MultiAB), unpack if tuple was
                                                     // passed
          typename BComputeDataType = AComputeDataType,
          LoopScheduler LoopSched   = make_default_loop_scheduler()>
struct DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle
    : public DeviceGroupedConvFwdMultipleABD<NDimSpatial,
                                             ALayout,
                                             BLayout,
                                             DsLayout,
                                             ELayout,
                                             ADataType,
                                             BDataType,
                                             DsDataType,
                                             EDataType,
                                             AElementwiseOperation,
                                             BElementwiseOperation,
                                             CDEElementwiseOperation,
                                             AComputeDataType,
                                             BComputeDataType>
{
    using DeviceOp = DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle;

    static constexpr bool isMultiA = is_detected<is_tuple, ADataType>::value;
    static constexpr bool isMultiB = is_detected<is_tuple, BDataType>::value;

    static constexpr index_t NumATensor = GetNumABTensors<isMultiA, ADataType>();
    static constexpr index_t NumBTensor = GetNumABTensors<isMultiB, BDataType>();
    static constexpr index_t NumDTensor = DsDataType::Size();

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};

    static constexpr auto conv_to_gemm_transformer =
        TransformConvFwdToGemm<NDimSpatial, ConvForwardSpecialization>{};

    static constexpr auto matrix_padder =
        MatrixPadder<GemmSpec, index_t, index_t, index_t>{MPerBlock, NPerBlock, KPerBlock};

    template <typename ALay>
    static auto
    MakeAGridDescriptor_M_K(const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_lengths,
                            const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_strides,
                            const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_lengths,
                            const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_strides,
                            const std::array<index_t, NDimSpatial + 3>& e_g_n_k_wos_lengths,
                            const std::array<index_t, NDimSpatial + 3>& e_g_n_k_wos_strides,
                            const std::array<index_t, NDimSpatial>& conv_filter_strides,
                            const std::array<index_t, NDimSpatial>& conv_filter_dilations,
                            const std::array<index_t, NDimSpatial>& input_left_pads,
                            const std::array<index_t, NDimSpatial>& input_right_pads)
    {
        const auto in_gemmmraw_gemmkraw_desc =
            conv_to_gemm_transformer.template MakeADescriptor_M_K<ALay>(a_g_n_c_wis_lengths,
                                                                        a_g_n_c_wis_strides,
                                                                        b_g_k_c_xs_lengths,
                                                                        b_g_k_c_xs_strides,
                                                                        e_g_n_k_wos_lengths,
                                                                        e_g_n_k_wos_strides,
                                                                        conv_filter_strides,
                                                                        conv_filter_dilations,
                                                                        input_left_pads,
                                                                        input_right_pads);

        const auto in_gemmm_gemmk_desc =
            matrix_padder.PadADescriptor_M_K(in_gemmmraw_gemmkraw_desc);

        return in_gemmm_gemmk_desc;
    }

    template <typename BLay>
    static auto
    MakeBGridDescriptor_N_K(const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_lengths,
                            const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_strides)
    {
        const auto wei_gemmnraw_gemmkraw_desc =
            conv_to_gemm_transformer.template MakeBDescriptor_N_K<BLay>(b_g_k_c_xs_lengths,
                                                                        b_g_k_c_xs_strides);

        const auto wei_gemmn_gemmk_desc =
            matrix_padder.PadBDescriptor_N_K(wei_gemmnraw_gemmkraw_desc);

        return wei_gemmn_gemmk_desc;
    }

    template <typename ELay>
    static auto
    MakeEGridDescriptor_M_N(const std::array<index_t, NDimSpatial + 3>& e_g_n_k_wos_lengths,
                            const std::array<index_t, NDimSpatial + 3>& e_g_n_k_wos_strides)
    {
        const auto out_gemmmraw_gemmnraw_desc =
            conv_to_gemm_transformer.template MakeCDescriptor_M_N<ELay>(e_g_n_k_wos_lengths,
                                                                        e_g_n_k_wos_strides);

        const auto out_gemmm_gemmn_desc =
            matrix_padder.PadCDescriptor_M_N(out_gemmmraw_gemmnraw_desc);

        return out_gemmm_gemmn_desc;
    }

    // Shape of Ds and E must be aligned. Strides can be different.
    // Pass e_g_n_k_wos_lengths for logical broadcast.
    static auto MakeDsGridDescriptor_M_N(
        const std::array<index_t, NDimSpatial + 3>& e_g_n_k_wos_lengths,
        const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor>& ds_g_n_k_wos_strides)
    {
        return generate_tuple(
            [&](auto i) {
                using DLayout = remove_cvref_t<tuple_element_t<i.value, DsLayout>>;

                return DeviceOp::MakeEGridDescriptor_M_N<DLayout>(e_g_n_k_wos_lengths,
                                                                  ds_g_n_k_wos_strides[i]);
            },
            Number<NumDTensor>{});
    }

    // desc for problem definition
    using AGridDesc_M_K  = remove_cvref_t<decltype(MakeAGridDescriptor_M_K<ALayout>(
        {}, {}, {}, {}, {}, {}, {}, {}, {}, {}))>;
    using BGridDesc_N_K  = remove_cvref_t<decltype(MakeBGridDescriptor_N_K<BLayout>({}, {}))>;
    using DsGridDesc_M_N = remove_cvref_t<decltype(MakeDsGridDescriptor_M_N({}, {}))>;
    using EGridDesc_M_N  = remove_cvref_t<decltype(MakeEGridDescriptor_M_N<ELayout>({}, {}))>;

    // If we are using multiAB and one of the template datatype parameters is not a tuple, convert
    // it to it
    using GemmADataType = std::conditional_t<!isMultiA && isMultiB, Tuple<ADataType>, ADataType>;
    using GemmBDataType = std::conditional_t<!isMultiB && isMultiA, Tuple<BDataType>, BDataType>;

#define GridwiseGemmTemplateParameters                                                          \
    GemmADataType, GemmBDataType, AComputeDataType, AccDataType, CShuffleDataType, DsDataType,  \
        EDataType, AElementwiseOperation, BElementwiseOperation, CDEElementwiseOperation,       \
        InMemoryDataOperationEnum::Set, NumGemmKPrefetchStage, BlockSize, MPerBlock, NPerBlock, \
        KPerBlock, AK1, BK1, MPerXDL, NPerXDL, MXdlPerWave, NXdlPerWave,                        \
        ABlockTransferThreadClusterLengths_AK0_M_AK1, ABlockTransferThreadClusterArrangeOrder,  \
        ABlockTransferSrcAccessOrder, ABlockTransferSrcVectorDim,                               \
        ABlockTransferSrcScalarPerVector, ABlockTransferDstScalarPerVector_AK1, false,          \
        ABlockLdsExtraM, BBlockTransferThreadClusterLengths_BK0_N_BK1,                          \
        BBlockTransferThreadClusterArrangeOrder, BBlockTransferSrcAccessOrder,                  \
        BBlockTransferSrcVectorDim, BBlockTransferSrcScalarPerVector,                           \
        BBlockTransferDstScalarPerVector_BK1, false, BBlockLdsExtraN,                           \
        CShuffleMXdlPerWavePerShuffle, CShuffleNXdlPerWavePerShuffle,                           \
        CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,                       \
        CDEBlockTransferScalarPerVector_NPerBlock, LoopSched, PipelineVersion::v1,              \
        BComputeDataType
    // Use appropriate gridwise gemm
    using GridwiseGemm =
        std::conditional_t<isMultiA || isMultiB,
                           GridwiseGemmMultipleABD_xdl_cshuffle<GridwiseGemmTemplateParameters>,
                           GridwiseGemmMultipleD_xdl_cshuffle<GridwiseGemmTemplateParameters>>;

    // If ADataTypes or BDataTypes is tuple, user has to pass std::array with pointers.
    using APointers =
        std::conditional_t<isMultiA, std::array<const void*, NumATensor>&, const void*>;
    using BPointers =
        std::conditional_t<isMultiB, std::array<const void*, NumBTensor>&, const void*>;
    // Use Tuple for the both cases for GridPointer to initialize it in Argument constructor (not
    // in initializer list what is required for single const pointer).
    using AGridPointer = remove_cvref_t<
        decltype(GetAGridPointer < isMultiA || isMultiB, GridwiseGemm, ADataType > ())>;
    using BGridPointer = remove_cvref_t<
        decltype(GetBGridPointer < isMultiA || isMultiB, GridwiseGemm, BDataType > ())>;

    // desc for blockwise copy
    using AGridDesc_AK0_M_AK1 =
        remove_cvref_t<decltype(GridwiseGemm::MakeDefaultAGridDescriptor_AK0_M_AK1(
            AGridDesc_M_K{}))>;
    using BGridDesc_BK0_N_BK1 =
        remove_cvref_t<decltype(GridwiseGemm::MakeDefaultBGridDescriptor_BK0_N_BK1(
            BGridDesc_N_K{}))>;
    using DsGridDesc_MBlock_MPerBlock_NBlock_NPerBlock = remove_cvref_t<
        decltype(GridwiseGemm::MakeDsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
            DsGridDesc_M_N{}))>;
    using EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock =
        remove_cvref_t<decltype(GridwiseGemm::MakeEGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
            EGridDesc_M_N{}))>;

    // block-to-e-tile map
    using Block2ETileMap =
        remove_cvref_t<decltype(GridwiseGemm::MakeDefaultBlock2ETileMap(EGridDesc_M_N{}))>;

    // Argument
    struct Argument : public BaseArgument
    {
        Argument(APointers p_as,
                 BPointers p_bs,
                 const std::array<const void*, NumDTensor>& p_ds,
                 void* p_e,
                 const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_lengths,
                 const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_strides,
                 const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_lengths,
                 const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_strides,
                 const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor>&
                     ds_g_n_k_wos_lengths,
                 const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor>&
                     ds_g_n_k_wos_strides,
                 const std::array<index_t, NDimSpatial + 3>& e_g_n_k_wos_lengths,
                 const std::array<index_t, NDimSpatial + 3>& e_g_n_k_wos_strides,
                 const std::array<index_t, NDimSpatial>& conv_filter_strides,
                 const std::array<index_t, NDimSpatial>& conv_filter_dilations,
                 const std::array<index_t, NDimSpatial>& input_left_pads,
                 const std::array<index_t, NDimSpatial>& input_right_pads,
                 const AElementwiseOperation& a_element_op,
                 const BElementwiseOperation& b_element_op,
                 const CDEElementwiseOperation& cde_element_op)
            : p_as_grid_{},
              p_bs_grid_{},
              p_ds_grid_{},
              p_e_grid_{static_cast<EDataType*>(p_e)},
              num_group_{a_g_n_c_wis_lengths[0]},
              a_grid_desc_m_k_{DeviceOp::MakeAGridDescriptor_M_K<ALayout>(a_g_n_c_wis_lengths,
                                                                          a_g_n_c_wis_strides,
                                                                          b_g_k_c_xs_lengths,
                                                                          b_g_k_c_xs_strides,
                                                                          e_g_n_k_wos_lengths,
                                                                          e_g_n_k_wos_strides,
                                                                          conv_filter_strides,
                                                                          conv_filter_dilations,
                                                                          input_left_pads,
                                                                          input_right_pads)},
              b_grid_desc_n_k_{DeviceOp::MakeBGridDescriptor_N_K<BLayout>(b_g_k_c_xs_lengths,
                                                                          b_g_k_c_xs_strides)},
              ds_grid_desc_m_n_{},
              e_grid_desc_m_n_{DeviceOp::MakeEGridDescriptor_M_N<ELayout>(e_g_n_k_wos_lengths,
                                                                          e_g_n_k_wos_strides)},
              a_grid_desc_ak0_m_ak1_{
                  GridwiseGemm::MakeDefaultAGridDescriptor_AK0_M_AK1(a_grid_desc_m_k_)},
              b_grid_desc_bk0_n_bk1_{
                  GridwiseGemm::MakeDefaultBGridDescriptor_BK0_N_BK1(b_grid_desc_n_k_)},
              ds_grid_desc_mblock_mperblock_nblock_nperblock_{},
              e_grid_desc_mblock_mperblock_nblock_nperblock_{},
              block_2_etile_map_{GridwiseGemm::MakeDefaultBlock2ETileMap(e_grid_desc_m_n_)},
              compute_ptr_offset_of_batch_{},
              a_element_op_{a_element_op},
              b_element_op_{b_element_op},
              cde_element_op_{cde_element_op},
              a_g_n_c_wis_lengths_{a_g_n_c_wis_lengths},
              a_g_n_c_wis_strides_{a_g_n_c_wis_strides},
              b_g_k_c_xs_lengths_{b_g_k_c_xs_lengths},
              b_g_k_c_xs_strides_{b_g_k_c_xs_strides},
              ds_g_n_k_wos_lengths_{ds_g_n_k_wos_lengths},
              ds_g_n_k_wos_strides_{ds_g_n_k_wos_strides},
              e_g_n_k_wos_lengths_{e_g_n_k_wos_lengths},
              e_g_n_k_wos_strides_{e_g_n_k_wos_strides},
              conv_filter_strides_{conv_filter_strides},
              conv_filter_dilations_{conv_filter_dilations},
              input_left_pads_{input_left_pads},
              input_right_pads_{input_right_pads}
        {
            // A/B/E Batch Stride
            if constexpr(isMultiA || isMultiB)
            {
                static_for<0, NumATensor, 1>{}([&](auto i) {
                    // Init compute_ptr_offset_of_batch_ for multiple AB
                    compute_ptr_offset_of_batch_.BatchStrideA_(i) = a_g_n_c_wis_strides[0];

                    // Use GemmADataType/GemmBDataType to iterate over tuple (even if passed data
                    // type is not tuple)
                    using DataType = remove_cvref_t<tuple_element_t<i.value, GemmADataType>>;
                    // It is possible that one of the AB is a pointer and one is a tuple.
                    // Then also use multiAB but we have to cast single pointer instead of tuple of
                    // pointer.
                    if constexpr(isMultiA)
                    {
                        // p_as is tuple
                        p_as_grid_(i) = static_cast<const DataType*>(p_as[i.value]);
                    }
                    else
                    {
                        // if MultiB and not MultiA then p_as is single pointer
                        p_as_grid_(i) = static_cast<const DataType*>(p_as);
                    }
                });
                static_for<0, NumBTensor, 1>{}([&](auto i) {
                    // Init compute_ptr_offset_of_batch_ for multiple AB
                    compute_ptr_offset_of_batch_.BatchStrideB_(i) = b_g_k_c_xs_strides[0];

                    using DataType = remove_cvref_t<tuple_element_t<i.value, GemmBDataType>>;
                    // It is possible that one of the AB is a pointer and one is a tuple.
                    // Then also use multiAB but we have to cast single pointer instead of tuple of
                    // pointer.
                    if constexpr(isMultiB)
                    {
                        // p_bs is tuple
                        p_bs_grid_(i) = static_cast<const DataType*>(p_bs[i.value]);
                    }
                    else
                    {
                        // if MultiA and not MultiB then p_bs is single pointer
                        p_bs_grid_(i) = static_cast<const DataType*>(p_bs);
                    }
                });
            }
            else
            {
                compute_ptr_offset_of_batch_.BatchStrideA_ = a_g_n_c_wis_strides[0];
                compute_ptr_offset_of_batch_.BatchStrideB_ = b_g_k_c_xs_strides[0];

                // p_as and p_bs are pointers
                p_as_grid_(I0) = static_cast<const ADataType*>(p_as);
                p_bs_grid_(I0) = static_cast<const BDataType*>(p_bs);
            }

            // populate pointer, batch stride, desc for Ds
            static_for<0, NumDTensor, 1>{}([&](auto i) {
                using DLayout   = remove_cvref_t<tuple_element_t<i.value, DsLayout>>;
                using DDataType = remove_cvref_t<tuple_element_t<i.value, DsDataType>>;

                // D pointer
                p_ds_grid_(i) = static_cast<const DDataType*>(p_ds[i]);

                // D batch stride
                compute_ptr_offset_of_batch_.BatchStrideDs_(i) = ds_g_n_k_wos_strides[i][0];

                // D desc
                ds_grid_desc_m_n_(i) = DeviceOp::MakeEGridDescriptor_M_N<DLayout>(
                    e_g_n_k_wos_lengths, ds_g_n_k_wos_strides[i]);
            });
            compute_ptr_offset_of_batch_.BatchStrideE_ = e_g_n_k_wos_strides[0];

            // populate desc for Ds/E
            if constexpr(isMultiA || isMultiB)
            {
                const auto as_grid_desc_ak0_m_ak1 =
                    generate_tuple([&](auto) { return a_grid_desc_m_k_; }, Number<NumATensor>{});
                const auto bs_grid_desc_bk0_n_bk1 =
                    generate_tuple([&](auto) { return b_grid_desc_n_k_; }, Number<NumBTensor>{});

                if(GridwiseGemm::CheckValidity(as_grid_desc_ak0_m_ak1,
                                               bs_grid_desc_bk0_n_bk1,
                                               ds_grid_desc_m_n_,
                                               e_grid_desc_m_n_,
                                               block_2_etile_map_))
                {
                    e_grid_desc_mblock_mperblock_nblock_nperblock_ =
                        GridwiseGemm::MakeEGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                            e_grid_desc_m_n_);

                    ds_grid_desc_mblock_mperblock_nblock_nperblock_ =
                        GridwiseGemm::MakeDsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                            ds_grid_desc_m_n_);
                }
            }
            else
            {
                if(GridwiseGemm::CheckValidity(a_grid_desc_m_k_,
                                               b_grid_desc_n_k_,
                                               ds_grid_desc_m_n_,
                                               e_grid_desc_m_n_,
                                               block_2_etile_map_))
                {
                    e_grid_desc_mblock_mperblock_nblock_nperblock_ =
                        GridwiseGemm::MakeEGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                            e_grid_desc_m_n_);

                    ds_grid_desc_mblock_mperblock_nblock_nperblock_ =
                        GridwiseGemm::MakeDsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                            ds_grid_desc_m_n_);
                }
            }
        }

        void Print() const
        {
            std::cout << "A[M, K]: " << a_grid_desc_m_k_ << std::endl;
            std::cout << "B[N, K]: " << b_grid_desc_n_k_ << std::endl;
            static_for<0, NumDTensor, 1>{}(
                [&](auto i) { std::cout << "Ds[M, N]: " << ds_grid_desc_m_n_[i] << std::endl; });
            std::cout << "E[M, N]: " << e_grid_desc_m_n_ << std::endl;
        }

        //  private:
        // pointers (tuple if multi AB, pointer if no)
        AGridPointer p_as_grid_;
        BGridPointer p_bs_grid_;
        typename GridwiseGemm::DsGridPointer p_ds_grid_;
        EDataType* p_e_grid_;

        // tensor descriptors for problem definiton
        index_t num_group_;
        AGridDesc_M_K a_grid_desc_m_k_;
        BGridDesc_N_K b_grid_desc_n_k_;
        DsGridDesc_M_N ds_grid_desc_m_n_;
        EGridDesc_M_N e_grid_desc_m_n_;

        // tensor descriptors for block/thread-wise copy
        AGridDesc_AK0_M_AK1 a_grid_desc_ak0_m_ak1_;
        BGridDesc_BK0_N_BK1 b_grid_desc_bk0_n_bk1_;
        DsGridDesc_MBlock_MPerBlock_NBlock_NPerBlock
            ds_grid_desc_mblock_mperblock_nblock_nperblock_;
        EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock e_grid_desc_mblock_mperblock_nblock_nperblock_;

        // block-to-e-tile map
        Block2ETileMap block_2_etile_map_;

        // for computing batch offset
        ComputePtrOffsetOfStridedBatch<NumATensor, NumBTensor, NumDTensor>
            compute_ptr_offset_of_batch_;

        // element-wise op
        AElementwiseOperation a_element_op_;
        BElementwiseOperation b_element_op_;
        CDEElementwiseOperation cde_element_op_;

        // for checking IsSupportedArgument()
        std::array<index_t, NDimSpatial + 3> a_g_n_c_wis_lengths_;
        std::array<index_t, NDimSpatial + 3> a_g_n_c_wis_strides_;
        std::array<index_t, NDimSpatial + 3> b_g_k_c_xs_lengths_;
        std::array<index_t, NDimSpatial + 3> b_g_k_c_xs_strides_;
        std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor> ds_g_n_k_wos_lengths_;
        std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor> ds_g_n_k_wos_strides_;
        std::array<index_t, NDimSpatial + 3> e_g_n_k_wos_lengths_;
        std::array<index_t, NDimSpatial + 3> e_g_n_k_wos_strides_;
        std::array<index_t, NDimSpatial> conv_filter_strides_;
        std::array<index_t, NDimSpatial> conv_filter_dilations_;
        std::array<index_t, NDimSpatial> input_left_pads_;
        std::array<index_t, NDimSpatial> input_right_pads_;
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        using Argument = DeviceOp::Argument;

        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            if(stream_config.log_level_ > 0)
            {
                arg.Print();
            }

            const index_t grid_size =
                arg.block_2_etile_map_.CalculateGridSize(arg.e_grid_desc_m_n_) * arg.num_group_;

            const auto K =
                arg.a_grid_desc_ak0_m_ak1_.GetLength(I0) * arg.a_grid_desc_ak0_m_ak1_.GetLength(I2);

            auto launch_kernel = [&](auto has_main_k_block_loop) {
                constexpr bool has_main_loop = has_main_k_block_loop.value;

                if constexpr(isMultiA || isMultiB)
                {
                    // Generate tuples with grid descriptors for each A and B
                    const auto as_grid_desc_ak0_m_ak1 = generate_tuple(
                        [&](auto) { return arg.a_grid_desc_ak0_m_ak1_; }, Number<NumATensor>{});
                    const auto bs_grid_desc_bk0_n_bk1 = generate_tuple(
                        [&](auto) { return arg.b_grid_desc_bk0_n_bk1_; }, Number<NumBTensor>{});

                    const auto kernel = kernel_grouped_conv_fwd_multiple_abd_xdl_cshuffle<
                        GridwiseGemm,
                        AGridPointer,
                        BGridPointer,
                        typename GridwiseGemm::DsGridPointer,
                        EDataType,
                        AElementwiseOperation,
                        BElementwiseOperation,
                        CDEElementwiseOperation,
                        decltype(as_grid_desc_ak0_m_ak1),
                        decltype(bs_grid_desc_bk0_n_bk1),
                        DeviceOp::DsGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
                        DeviceOp::EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
                        Block2ETileMap,
                        ComputePtrOffsetOfStridedBatch<NumATensor, NumBTensor, NumDTensor>,
                        has_main_loop,
                        isMultiA,
                        isMultiB>;

                    return launch_and_time_kernel(
                        stream_config,
                        kernel,
                        dim3(grid_size),
                        dim3(BlockSize),
                        0,
                        arg.p_as_grid_,
                        arg.p_bs_grid_,
                        arg.p_ds_grid_,
                        arg.p_e_grid_,
                        arg.a_element_op_,
                        arg.b_element_op_,
                        arg.cde_element_op_,
                        arg.a_g_n_c_wis_lengths_[0], // Group count
                        as_grid_desc_ak0_m_ak1,
                        bs_grid_desc_bk0_n_bk1,
                        arg.ds_grid_desc_mblock_mperblock_nblock_nperblock_,
                        arg.e_grid_desc_mblock_mperblock_nblock_nperblock_,
                        arg.block_2_etile_map_,
                        arg.compute_ptr_offset_of_batch_);
                }
                else
                {
                    const auto kernel = kernel_grouped_conv_fwd_multiple_abd_xdl_cshuffle<
                        GridwiseGemm,
                        const ADataType*,
                        const BDataType*,
                        typename GridwiseGemm::DsGridPointer,
                        EDataType,
                        AElementwiseOperation,
                        BElementwiseOperation,
                        CDEElementwiseOperation,
                        DeviceOp::AGridDesc_AK0_M_AK1,
                        DeviceOp::BGridDesc_BK0_N_BK1,
                        DeviceOp::DsGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
                        DeviceOp::EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
                        Block2ETileMap,
                        ComputePtrOffsetOfStridedBatch<NumATensor, NumBTensor, NumDTensor>,
                        has_main_loop,
                        isMultiA,
                        isMultiB>;

                    return launch_and_time_kernel(
                        stream_config,
                        kernel,
                        dim3(grid_size),
                        dim3(BlockSize),
                        0,
                        arg.p_as_grid_.At(I0), // Pass just A descriptor instead of tuple
                        arg.p_bs_grid_.At(I0), // Pass just B descriptor instead of tuple
                        arg.p_ds_grid_,
                        arg.p_e_grid_,
                        arg.a_element_op_,
                        arg.b_element_op_,
                        arg.cde_element_op_,
                        arg.a_g_n_c_wis_lengths_[0], // Group count
                        arg.a_grid_desc_ak0_m_ak1_,
                        arg.b_grid_desc_bk0_n_bk1_,
                        arg.ds_grid_desc_mblock_mperblock_nblock_nperblock_,
                        arg.e_grid_desc_mblock_mperblock_nblock_nperblock_,
                        arg.block_2_etile_map_,
                        arg.compute_ptr_offset_of_batch_);
                }
            };

            if(GridwiseGemm::CalculateHasMainKBlockLoop(K))
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

    static bool IsSupportedArgument(const Argument& arg)
    {
        namespace ctc = tensor_layout::convolution;

        // check device
        if(get_device_name() == "gfx908")
        {
            // FIXME: re-enable fp64 when SWDEV-335738 is fixed
            if constexpr(!(is_same_v<AccDataType, float> || is_same_v<AccDataType, int32_t>))
            {
                return false;
            }
        }
        else if(ck::is_lds_direct_load_supported())
        {
            if constexpr(!(is_same_v<AccDataType, float> || is_same_v<AccDataType, float> ||
                           is_same_v<AccDataType, int32_t> || is_same_v<AccDataType, double>))
            {
                return false;
            }
        }
        else
        {
            return false;
        }

        // check ConvolutionForwardSpecialization
        if constexpr(ConvForwardSpecialization ==
                     ConvolutionForwardSpecialization::Filter1x1Stride1Pad0)
        {
            // check if it's 1x1, stride=1 conv
            for(index_t i = 0; i < NDimSpatial; ++i)
            {
                const index_t X          = arg.b_g_k_c_xs_lengths_[i + 3];
                const index_t ConvStride = arg.conv_filter_strides_[i];
                const index_t LeftPad    = arg.input_left_pads_[i];
                const index_t RightPad   = arg.input_right_pads_[i];

                if(!(X == 1 && ConvStride == 1 && LeftPad == 0 && RightPad == 0))
                {
                    return false;
                }
            }
        }
        else if constexpr(ConvForwardSpecialization ==
                          ConvolutionForwardSpecialization::Filter1x1Pad0)
        {
            // check if it's 1x1 conv
            for(index_t i = 0; i < NDimSpatial; ++i)
            {
                const index_t X        = arg.b_g_k_c_xs_lengths_[i + 3];
                const index_t LeftPad  = arg.input_left_pads_[i];
                const index_t RightPad = arg.input_right_pads_[i];

                if(!(X == 1 && LeftPad == 0 && RightPad == 0))
                {
                    return false;
                }
            }
        }

        // check vector access of A
        // FIXME: layout
        if constexpr(is_same_v<ALayout, ctc::G_NW_C> || is_same_v<ALayout, ctc::G_NHW_C> ||
                     is_same_v<ALayout, ctc::G_NDHW_C> || is_same_v<ALayout, ctc::GNWC> ||
                     is_same_v<ALayout, ctc::GNHWC> || is_same_v<ALayout, ctc::GNDHWC> ||
                     is_same_v<ALayout, ctc::NWGC> || is_same_v<ALayout, ctc::NHWGC> ||
                     is_same_v<ALayout, ctc::NDHWGC>)
        {
            const index_t C = arg.a_g_n_c_wis_lengths_[2];

            if(!(ABlockTransferSrcVectorDim == 2 && C % ABlockTransferSrcScalarPerVector == 0))
            {
                return false;
            }
        }
        else
        {
            return false;
        }

        // check vector access of B
        // FIXME: layout
        if constexpr(is_same_v<BLayout, ctc::G_K_X_C> || is_same_v<BLayout, ctc::G_K_YX_C> ||
                     is_same_v<BLayout, ctc::G_K_ZYX_C> || is_same_v<BLayout, ctc::GKXC> ||
                     is_same_v<BLayout, ctc::GKYXC> || is_same_v<BLayout, ctc::GKZYXC> ||
                     is_same_v<BLayout, ctc::KXGC> || is_same_v<BLayout, ctc::KYXGC> ||
                     is_same_v<BLayout, ctc::KZYXGC>)

        {
            const index_t C = arg.b_g_k_c_xs_lengths_[2];

            if(!(BBlockTransferSrcVectorDim == 2 && C % BBlockTransferSrcScalarPerVector == 0))
            {
                return false;
            }
        }
        else
        {
            return false;
        }

        //  check vector access of Ds
        bool valid = true;

        static_for<0, NumDTensor, 1>{}([&](auto i) {
            using DLayout = remove_cvref_t<tuple_element_t<i.value, DsLayout>>;

            // FIXME: layout
            if constexpr(is_same_v<DLayout, ctc::G_NW_K> || is_same_v<DLayout, ctc::G_NHW_K> ||
                         is_same_v<DLayout, ctc::G_NDHW_K> || is_same_v<DLayout, ctc::GNWK> ||
                         is_same_v<DLayout, ctc::GNHWK> || is_same_v<DLayout, ctc::GNDHWK> ||
                         is_same_v<DLayout, ctc::NWGK> || is_same_v<DLayout, ctc::NHWGK> ||
                         is_same_v<DLayout, ctc::NDHWGK> || is_same_v<DLayout, ctc::G_K>)
            {
                const index_t K = arg.ds_g_n_k_wos_lengths_[i][2];

                if(!(K % CDEBlockTransferScalarPerVector_NPerBlock == 0))
                {
                    valid = false;
                }

                if constexpr(is_same_v<DLayout, ctc::G_K>)
                {
                    // G and K must be the same
                    if(arg.ds_g_n_k_wos_lengths_[i][0] != arg.e_g_n_k_wos_lengths_[0] ||
                       arg.ds_g_n_k_wos_lengths_[i][2] != arg.e_g_n_k_wos_lengths_[2])
                    {
                        valid = false;
                    }
                }
                else
                {
                    // E and D must have the same shape
                    for(index_t d = 0; d < NDimSpatial + 3; d++)
                    {
                        if(arg.ds_g_n_k_wos_lengths_[i][d] != arg.e_g_n_k_wos_lengths_[d])
                        {
                            valid = false;
                        }
                    }
                }
            }
            else
            {
                valid = false;
            }
        });

        if(!valid)
        {
            return false;
        }

        // check vector access of E
        if constexpr(is_same_v<ELayout, ctc::G_NW_K> || is_same_v<ELayout, ctc::G_NHW_K> ||
                     is_same_v<ELayout, ctc::G_NDHW_K> || is_same_v<ELayout, ctc::GNWK> ||
                     is_same_v<ELayout, ctc::GNHWK> || is_same_v<ELayout, ctc::GNDHWK> ||
                     is_same_v<ELayout, ctc::NWGK> || is_same_v<ELayout, ctc::NHWGK> ||
                     is_same_v<ELayout, ctc::NDHWGK>)
        {
            const index_t K = arg.e_g_n_k_wos_lengths_[2];

            if(!(K % CDEBlockTransferScalarPerVector_NPerBlock == 0))
            {
                return false;
            }
        }
        else
        {
            return false;
        }

        // check Gridwise GEMM
        if constexpr(isMultiA || isMultiB)
        {
            // Genarate tuples with the same descriptors
            const auto as_grid_desc_ak0_m_ak1 =
                generate_tuple([&](auto) { return arg.a_grid_desc_m_k_; }, Number<NumATensor>{});
            const auto bs_grid_desc_bk0_n_bk1 =
                generate_tuple([&](auto) { return arg.b_grid_desc_n_k_; }, Number<NumBTensor>{});
            return GridwiseGemm::CheckValidity(as_grid_desc_ak0_m_ak1,
                                               bs_grid_desc_bk0_n_bk1,
                                               arg.ds_grid_desc_m_n_,
                                               arg.e_grid_desc_m_n_,
                                               arg.block_2_etile_map_);
        }
        else
        {
            return GridwiseGemm::CheckValidity(arg.a_grid_desc_m_k_,
                                               arg.b_grid_desc_n_k_,
                                               arg.ds_grid_desc_m_n_,
                                               arg.e_grid_desc_m_n_,
                                               arg.block_2_etile_map_);
        }
    }

    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static auto MakeArgument(
        APointers p_as,
        BPointers p_bs,
        const std::array<const void*, NumDTensor>& p_ds,
        void* p_e,
        const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_lengths,
        const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_strides,
        const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_lengths,
        const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_strides,
        const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor>& ds_g_n_k_wos_lengths,
        const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor>& ds_g_n_k_wos_strides,
        const std::array<index_t, NDimSpatial + 3>& e_g_n_k_wos_lengths,
        const std::array<index_t, NDimSpatial + 3>& e_g_n_k_wos_strides,
        const std::array<index_t, NDimSpatial>& conv_filter_strides,
        const std::array<index_t, NDimSpatial>& conv_filter_dilations,
        const std::array<index_t, NDimSpatial>& input_left_pads,
        const std::array<index_t, NDimSpatial>& input_right_pads,
        const AElementwiseOperation& a_element_op,
        const BElementwiseOperation& b_element_op,
        const CDEElementwiseOperation& cde_element_op)
    {
        return Argument{p_as,
                        p_bs,
                        p_ds,
                        p_e,
                        a_g_n_c_wis_lengths,
                        a_g_n_c_wis_strides,
                        b_g_k_c_xs_lengths,
                        b_g_k_c_xs_strides,
                        ds_g_n_k_wos_lengths,
                        ds_g_n_k_wos_strides,
                        e_g_n_k_wos_lengths,
                        e_g_n_k_wos_strides,
                        conv_filter_strides,
                        conv_filter_dilations,
                        input_left_pads,
                        input_right_pads,
                        a_element_op,
                        b_element_op,
                        cde_element_op};
    }

    static auto MakeInvoker() { return Invoker{}; }

    std::unique_ptr<BaseArgument> MakeArgumentPointer(
        APointers p_a,
        BPointers p_b,
        const std::array<const void*, NumDTensor>& p_ds,
        void* p_e,
        const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_lengths,
        const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_strides,
        const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_lengths,
        const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_strides,
        const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor>& ds_g_n_k_wos_lengths,
        const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor>& ds_g_n_k_wos_strides,
        const std::array<index_t, NDimSpatial + 3>& e_g_n_k_wos_lengths,
        const std::array<index_t, NDimSpatial + 3>& e_g_n_k_wos_strides,
        const std::array<index_t, NDimSpatial>& conv_filter_strides,
        const std::array<index_t, NDimSpatial>& conv_filter_dilations,
        const std::array<index_t, NDimSpatial>& input_left_pads,
        const std::array<index_t, NDimSpatial>& input_right_pads,
        const AElementwiseOperation& a_element_op,
        const BElementwiseOperation& b_element_op,
        const CDEElementwiseOperation& cde_element_op) override
    {
        return std::make_unique<Argument>(p_a,
                                          p_b,
                                          p_ds,
                                          p_e,
                                          a_g_n_c_wis_lengths,
                                          a_g_n_c_wis_strides,
                                          b_g_k_c_xs_lengths,
                                          b_g_k_c_xs_strides,
                                          ds_g_n_k_wos_lengths,
                                          ds_g_n_k_wos_strides,
                                          e_g_n_k_wos_lengths,
                                          e_g_n_k_wos_strides,
                                          conv_filter_strides,
                                          conv_filter_dilations,
                                          input_left_pads,
                                          input_right_pads,
                                          a_element_op,
                                          b_element_op,
                                          cde_element_op);
    }

    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        // clang-format off
        str << "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle"
            << "<"
            << BlockSize << ", "
            << MPerBlock << ", "
            << NPerBlock << ", "
            << KPerBlock << ", "
            << getConvForwardSpecializationString(ConvForwardSpecialization) << ", "
            << MPerXDL << ", "
            << NPerXDL << ", "
            << MXdlPerWave << ", "
            << NXdlPerWave << ", "
            << ABlockTransferSrcScalarPerVector << ", "
            << BBlockTransferSrcScalarPerVector << ", "
            << CDEBlockTransferScalarPerVector_NPerBlock << ", "
            << CShuffleMXdlPerWavePerShuffle << ", "
            << CShuffleNXdlPerWavePerShuffle
            << ">";
        // clang-format on

        return str.str();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
