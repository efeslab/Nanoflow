// SPDX-License-Identifier: MIT
// Copyright (c) 2023-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <functional>
#include <iostream>
#include <iterator>
#include <numeric>
#include <sstream>

#include "ck/library/utility/numeric.hpp"
#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/convolution_forward_specialization.hpp"
#include "ck/tensor_operation/operator_transform/transform_conv_fwd_to_gemm.hpp"
#include "ck/tensor_operation/operator_transform/transform_conv_ngchw_to_nhwgc.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_conv_fwd_multiple_abd.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/matrix_padder.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_v3.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_elementwise_2d.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_utils.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"
#include "ck/host_utility/flush_cache.hpp"
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
          typename AGridDesc_AK0_M_K1,
          typename BGridDesc_BK0_N_K1,
          typename CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
          typename ComputePtrOffset,
          bool HasMainKBlockLoop,
          InMemoryDataOperationEnum CGlobalMemoryDataOperation,
          index_t MinimumOccupancy = 1,
          TailNumber TailNum       = TailNumber::Full>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, MinimumOccupancy)
#endif
        kernel_grouped_conv_fwd_xdl_cshuffle_v3(
            typename GridwiseGemm::Argument karg,
            [[maybe_unused]] const AGridDesc_AK0_M_K1 a_grid_desc_ak0_m_ak1,
            [[maybe_unused]] const BGridDesc_BK0_N_K1 b_grid_desc_bk0_n_bk1,
            [[maybe_unused]] const CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock
                c_grid_desc_mblock_mperblock_nblock_nperblock,
            [[maybe_unused]] const ComputePtrOffset compute_ptr_offset_of_groups,
            [[maybe_unused]] const ComputePtrOffset compute_ptr_offset_of_n,
            [[maybe_unused]] const index_t groups_count)
{
#if(!defined(__HIP_DEVICE_COMPILE__) || defined(__gfx9__))
    // offset base pointer for each work-group
    const index_t num_blocks_per_batch = __builtin_amdgcn_readfirstlane(gridDim.y / groups_count);
    const index_t& num_blocks_per_n    = groups_count;
    const index_t g_idx = __builtin_amdgcn_readfirstlane(blockIdx.y / num_blocks_per_batch);
    const index_t n_idx = __builtin_amdgcn_readfirstlane(blockIdx.y / num_blocks_per_n);

    const long_index_t a_batch_offset =
        amd_wave_read_first_lane(compute_ptr_offset_of_groups.GetAPtrOffset(g_idx));
    const long_index_t b_batch_offset =
        amd_wave_read_first_lane(compute_ptr_offset_of_groups.GetBPtrOffset(g_idx));
    const long_index_t e_batch_offset =
        amd_wave_read_first_lane(compute_ptr_offset_of_groups.GetEPtrOffset(g_idx));

    const long_index_t a_n_offset =
        amd_wave_read_first_lane(compute_ptr_offset_of_n.GetAPtrOffset(n_idx));
    const long_index_t e_n_offset =
        amd_wave_read_first_lane(compute_ptr_offset_of_n.GetEPtrOffset(n_idx));

    __shared__ char p_shared[GridwiseGemm::GetSharedMemoryNumberOfByte()];

    GridwiseGemm::template Run<AGridDesc_AK0_M_K1,
                               BGridDesc_BK0_N_K1,
                               CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
                               HasMainKBlockLoop,
                               CGlobalMemoryDataOperation,
                               TailNum>(karg.p_a_grid + a_batch_offset + a_n_offset,
                                        karg.p_b_grid + b_batch_offset,
                                        karg.p_c_grid + e_batch_offset + e_n_offset,
                                        p_shared,
                                        karg,
                                        a_grid_desc_ak0_m_ak1,
                                        b_grid_desc_bk0_n_bk1,
                                        c_grid_desc_mblock_mperblock_nblock_nperblock);
#else
    ignore = karg;
#endif // end of if (defined(__gfx908__) || defined(__gfx90a__))
}

template <typename GridwiseGemm,
          typename AGridDesc_AK0_M_K1,
          typename BGridDesc_BK0_N_K1,
          typename CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
          typename ComputePtrOffset,
          bool HasMainKBlockLoop,
          InMemoryDataOperationEnum CGlobalMemoryDataOperation,
          index_t MinimumOccupancy = 1,
          TailNumber TailNum       = TailNumber::Full>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, MinimumOccupancy)
#endif
        kernel_grouped_conv_fwd_xdl_cshuffle_v3_2lds(
            typename GridwiseGemm::Argument karg,
            [[maybe_unused]] const AGridDesc_AK0_M_K1 a_grid_desc_ak0_m_ak1,
            [[maybe_unused]] const BGridDesc_BK0_N_K1 b_grid_desc_bk0_n_bk1,
            [[maybe_unused]] const CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock
                c_grid_desc_mblock_mperblock_nblock_nperblock,
            [[maybe_unused]] const ComputePtrOffset compute_ptr_offset_of_groups,
            [[maybe_unused]] const ComputePtrOffset compute_ptr_offset_of_n,
            [[maybe_unused]] const index_t groups_count)
{
#if(!defined(__HIP_DEVICE_COMPILE__) || defined(__gfx9__))
    // offset base pointer for each work-group
    const index_t num_blocks_per_batch = __builtin_amdgcn_readfirstlane(gridDim.y / groups_count);
    const index_t& num_blocks_per_n    = groups_count;
    const index_t g_idx = __builtin_amdgcn_readfirstlane(blockIdx.y / num_blocks_per_batch);
    const index_t n_idx = __builtin_amdgcn_readfirstlane(blockIdx.y / num_blocks_per_n);

    const long_index_t a_batch_offset =
        amd_wave_read_first_lane(compute_ptr_offset_of_groups.GetAPtrOffset(g_idx));
    const long_index_t b_batch_offset =
        amd_wave_read_first_lane(compute_ptr_offset_of_groups.GetBPtrOffset(g_idx));
    const long_index_t e_batch_offset =
        amd_wave_read_first_lane(compute_ptr_offset_of_groups.GetEPtrOffset(g_idx));

    const long_index_t a_n_offset =
        amd_wave_read_first_lane(compute_ptr_offset_of_n.GetAPtrOffset(n_idx));
    const long_index_t e_n_offset =
        amd_wave_read_first_lane(compute_ptr_offset_of_n.GetEPtrOffset(n_idx));

    // Pass two lds pointer is the key to tell compiler that ds_read/write
    // operate on different lds chunk at same time without order dependecy
    __shared__ char p_shared_0[GridwiseGemm::GetSharedMemoryNumberOfByte()];
    __shared__ char p_shared_1[GridwiseGemm::GetSharedMemoryNumberOfByte()];

    GridwiseGemm::template Run_2Lds<AGridDesc_AK0_M_K1,
                                    BGridDesc_BK0_N_K1,
                                    CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
                                    HasMainKBlockLoop,
                                    CGlobalMemoryDataOperation,
                                    TailNum>(karg.p_a_grid + a_batch_offset + a_n_offset,
                                             karg.p_b_grid + b_batch_offset,
                                             karg.p_c_grid + e_batch_offset + e_n_offset,
                                             p_shared_0,
                                             p_shared_1,
                                             karg,
                                             a_grid_desc_ak0_m_ak1,
                                             b_grid_desc_bk0_n_bk1,
                                             c_grid_desc_mblock_mperblock_nblock_nperblock);
#else
    ignore = karg;
#endif // end of if (defined(__gfx908__) || defined(__gfx90a__))
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
          BlockGemmPipelineScheduler BlkGemmPipeSched = BlockGemmPipelineScheduler::Intrawave,
          BlockGemmPipelineVersion BlkGemmPipelineVer = BlockGemmPipelineVersion::v1,
          typename AComputeDataType =
              decltype(UnpackDataType<is_detected<is_tuple, ADataType>::value,
                                      Number<0>,
                                      ADataType>()), // ComputeType is InputType by default (first
                                                     // in tuple for MultiAB), unpack if tuple was
                                                     // passed
          typename BComputeDataType = AComputeDataType>
struct DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3
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
    using DeviceOp = DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3;

    static constexpr bool isMultiA   = is_detected<is_tuple, ADataType>::value;
    static constexpr bool isMultiB   = is_detected<is_tuple, BDataType>::value;
    static constexpr bool isMultiD   = DsDataType::Size() > 0;
    static constexpr bool isMultiABD = isMultiA || isMultiB || isMultiD;

    // multi ABD not supported
    static_assert(!isMultiABD, "Multi A, Mutli B and Multi D are not supported");

    static constexpr index_t NumATensor = GetNumABTensors<isMultiA, ADataType>();
    static constexpr index_t NumBTensor = GetNumABTensors<isMultiB, BDataType>();
    static constexpr index_t NumDTensor = DsDataType::Size();

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};
    static constexpr auto I5 = Number<5>{};

    using ConvToGemmFwdTransformer = TransformConvFwdToGemm<NDimSpatial,
                                                            ConvForwardSpecialization,
                                                            true /*SplitN*/,
                                                            ADataType,
                                                            EDataType>;

    static constexpr auto matrix_padder =
        MatrixPadder<GemmSpec, index_t, index_t, index_t>{MPerBlock, NPerBlock, KPerBlock};

    static constexpr index_t ClusterLengthNPerBlock =
        CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock::At(3);

    static constexpr auto conv_ngchw_to_nhwgc_transformer =
        TransformConvNGCHWToNHWGC<ALayout,
                                  BLayout,
                                  ELayout,
                                  NDimSpatial,
                                  MPerBlock / ClusterLengthNPerBlock,
                                  NPerBlock / ClusterLengthNPerBlock>{};

    template <typename ALay>
    static auto
    MakeAGridDescriptor_AK0_M_AK1(const ConvToGemmFwdTransformer& conv_to_gemm_transformer)

    {
        namespace ctc = tensor_layout::convolution;
        using Layout  = std::conditional_t<
            is_NGCHW_GKYXC_NGKHW<ALayout, BLayout, ELayout>(),
            ctc::NHWGC,
            std::conditional_t<is_NGCDHW_GKZYXC_NGKDHW<ALayout, BLayout, ELayout>(),
                               ctc::NDHWGC,
                               ALay>>;

        const auto in_gemmmraw_gemmkraw_desc =
            conv_to_gemm_transformer.template MakeADescriptor_M_K<Layout>();

        const auto in_gemmm_gemmk_desc =
            matrix_padder.PadADescriptor_M_K(in_gemmmraw_gemmkraw_desc);

        const auto M = in_gemmm_gemmk_desc.GetLength(I0);
        const auto K = in_gemmm_gemmk_desc.GetLength(I1);

        const auto AK0 = K / AK1;

        return transform_tensor_descriptor(in_gemmm_gemmk_desc,
                                           make_tuple(make_unmerge_transform(make_tuple(AK0, AK1)),
                                                      make_pass_through_transform(M)),
                                           make_tuple(Sequence<1>{}, Sequence<0>{}),
                                           make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
    }

    template <typename BLay>
    static auto
    MakeBGridDescriptor_BK0_N_BK1(const ConvToGemmFwdTransformer& conv_to_gemm_transformer)
    {
        const auto wei_gemmnraw_gemmkraw_desc =
            conv_to_gemm_transformer.template MakeBDescriptor_N_K<BLay>();

        const auto wei_gemmn_gemmk_desc =
            matrix_padder.PadBDescriptor_N_K(wei_gemmnraw_gemmkraw_desc);

        const auto N = wei_gemmn_gemmk_desc.GetLength(I0);
        const auto K = wei_gemmn_gemmk_desc.GetLength(I1);

        const auto BK0 = K / BK1;

        return transform_tensor_descriptor(wei_gemmn_gemmk_desc,
                                           make_tuple(make_unmerge_transform(make_tuple(BK0, BK1)),
                                                      make_pass_through_transform(N)),
                                           make_tuple(Sequence<1>{}, Sequence<0>{}),
                                           make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
    }

    template <typename ELay>
    static auto MakeEGridDescriptor_M_N(const ConvToGemmFwdTransformer& conv_to_gemm_transformer)

    {
        namespace ctc = tensor_layout::convolution;
        using Layout  = std::conditional_t<
            is_NGCHW_GKYXC_NGKHW<ALayout, BLayout, ELayout>(),
            ctc::NHWGK,
            std::conditional_t<is_NGCDHW_GKZYXC_NGKDHW<ALayout, BLayout, ELayout>(),
                               ctc::NDHWGK,
                               ELay>>;

        const auto out_gemmmraw_gemmnraw_desc =
            conv_to_gemm_transformer.template MakeCDescriptor_M_N<Layout>();

        const auto out_gemmm_gemmn_desc =
            matrix_padder.PadCDescriptor_M_N(out_gemmmraw_gemmnraw_desc);

        return out_gemmm_gemmn_desc;
    }

    // desc for problem definition
    constexpr static ConvToGemmFwdTransformer dummy_conv_to_gemm_transformer;
    using EGridDesc_M_N =
        remove_cvref_t<decltype(MakeEGridDescriptor_M_N<ELayout>(dummy_conv_to_gemm_transformer))>;

#define GridwiseGemmV3TemplateParams                                                           \
    tensor_layout::gemm::RowMajor, tensor_layout::gemm::ColumnMajor,                           \
        tensor_layout::gemm::RowMajor, ADataType, BDataType, AccDataType, CShuffleDataType,    \
        EDataType, AElementwiseOperation, BElementwiseOperation, CDEElementwiseOperation,      \
        GemmSpec, BlockSize, MPerBlock, NPerBlock, KPerBlock, AK1, BK1, MPerXDL, NPerXDL,      \
        MXdlPerWave, NXdlPerWave, ABlockTransferThreadClusterLengths_AK0_M_AK1,                \
        ABlockTransferThreadClusterArrangeOrder, ABlockTransferSrcAccessOrder,                 \
        ABlockTransferSrcVectorDim, ABlockTransferSrcScalarPerVector,                          \
        ABlockTransferDstScalarPerVector_AK1, false, ABlockLdsExtraM,                          \
        BBlockTransferThreadClusterLengths_BK0_N_BK1, BBlockTransferThreadClusterArrangeOrder, \
        BBlockTransferSrcAccessOrder, BBlockTransferSrcVectorDim,                              \
        BBlockTransferSrcScalarPerVector, BBlockTransferDstScalarPerVector_BK1, false,         \
        BBlockLdsExtraN, CShuffleMXdlPerWavePerShuffle, CShuffleNXdlPerWavePerShuffle,         \
        CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,                      \
        CDEBlockTransferScalarPerVector_NPerBlock, BlkGemmPipeSched, BlkGemmPipelineVer,       \
        AComputeDataType, BComputeDataType

    // Use appropriate gridwise gemm
    using GridwiseGemm = GridwiseGemm_xdl_cshuffle_v3<GridwiseGemmV3TemplateParams>;

    using Block2TileMapElementwise = BlockToCTileMap_M00_N0_M01Adapt<NPerBlock, NPerBlock>;

    using NGCHWTransposeDescType =
        remove_cvref_t<decltype(conv_ngchw_to_nhwgc_transformer
                                    .template MakeNGCHWTransposeDesc<NDimSpatial>({}, {}))>;
    using NHWGCTransposeDescType =
        remove_cvref_t<decltype(conv_ngchw_to_nhwgc_transformer
                                    .template MakeNHWGCTransposeDesc<NDimSpatial>({}, {}))>;

    static constexpr index_t ElementwiseBlocksize = ClusterLengthNPerBlock * ClusterLengthNPerBlock;

    using GridwiseElementwiseInputTranspose =
        GridwiseElementwise<Tuple<NGCHWTransposeDescType>,
                            Tuple<NHWGCTransposeDescType>,
                            Tuple<const ADataType*>,
                            Tuple<ADataType*>,
                            Block2TileMapElementwise,
                            element_wise::PassThrough,
                            ElementwiseBlocksize,
                            NPerBlock,
                            NPerBlock,
                            NPerBlock / ClusterLengthNPerBlock,
                            NPerBlock / ClusterLengthNPerBlock,
                            Sequence<1, 0>,
                            Sequence<CDEBlockTransferScalarPerVector_NPerBlock>,
                            Sequence<CDEBlockTransferScalarPerVector_NPerBlock>,
                            I1,
                            I0>;

    using GridwiseElementwiseOutputTranspose =
        GridwiseElementwise<Tuple<NHWGCTransposeDescType>,
                            Tuple<NGCHWTransposeDescType>,
                            Tuple<const EDataType*>,
                            Tuple<EDataType*>,
                            Block2TileMapElementwise,
                            element_wise::PassThrough,
                            ElementwiseBlocksize,
                            NPerBlock,
                            NPerBlock,
                            NPerBlock / ClusterLengthNPerBlock,
                            NPerBlock / ClusterLengthNPerBlock,
                            Sequence<1, 0>,
                            Sequence<CDEBlockTransferScalarPerVector_NPerBlock>,
                            Sequence<CDEBlockTransferScalarPerVector_NPerBlock>,
                            I0,
                            I1>;

    static auto
    MakeEGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(const EGridDesc_M_N& e_grid_desc_m_n)
    {
        const index_t M = e_grid_desc_m_n.GetLength(I0);
        const index_t N = e_grid_desc_m_n.GetLength(I1);
        return GridwiseGemm::MakeCGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
            e_grid_desc_m_n, GridwiseGemm::CalculateMBlock(M), GridwiseGemm::CalculateNBlock(N));
    }

    // desc for blockwise copy
    using AGridDesc_AK0_M_AK1 = remove_cvref_t<decltype(MakeAGridDescriptor_AK0_M_AK1<ALayout>(
        dummy_conv_to_gemm_transformer))>;
    using BGridDesc_BK0_N_BK1 = remove_cvref_t<decltype(MakeBGridDescriptor_BK0_N_BK1<BLayout>(
        dummy_conv_to_gemm_transformer))>;
    using EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock =
        remove_cvref_t<decltype(MakeEGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
            EGridDesc_M_N{}))>;

    // Argument
    struct Argument : public BaseArgument
    {
        Argument(const void* p_as,
                 const void* p_bs,
                 const std::array<const void*, NumDTensor>&,
                 void* p_e,
                 const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_lengths,
                 const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_strides,
                 const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_lengths,
                 const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_strides,
                 const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor>&,
                 const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor>&,
                 const std::array<index_t, NDimSpatial + 3>& e_g_n_k_wos_lengths,
                 const std::array<index_t, NDimSpatial + 3>& e_g_n_k_wos_strides,
                 const std::array<index_t, NDimSpatial>& conv_filter_strides,
                 const std::array<index_t, NDimSpatial>& conv_filter_dilations,
                 const std::array<index_t, NDimSpatial>& input_left_pads,
                 const std::array<index_t, NDimSpatial>& input_right_pads,
                 const AElementwiseOperation& a_element_op,
                 const BElementwiseOperation& b_element_op,
                 const CDEElementwiseOperation& cde_element_op)
            : p_a_grid_{},
              p_b_grid_{},
              p_e_grid_{static_cast<EDataType*>(p_e)},
              a_g_n_c_wis_lengths_{a_g_n_c_wis_lengths},
              a_g_n_c_wis_strides_{conv_ngchw_to_nhwgc_transformer.TransposeStrides(
                  a_g_n_c_wis_lengths, a_g_n_c_wis_strides)},
              b_g_k_c_xs_lengths_{b_g_k_c_xs_lengths},
              b_g_k_c_xs_strides_{b_g_k_c_xs_strides},
              e_g_n_k_wos_lengths_{e_g_n_k_wos_lengths},
              e_g_n_k_wos_strides_{conv_ngchw_to_nhwgc_transformer.TransposeStrides(
                  e_g_n_k_wos_lengths, e_g_n_k_wos_strides)},
              conv_filter_strides_{conv_filter_strides},
              conv_filter_dilations_{conv_filter_dilations},
              input_left_pads_{input_left_pads},
              input_right_pads_{input_right_pads},
              num_group_{a_g_n_c_wis_lengths_[0]},
              conv_to_gemm_transformer_{a_g_n_c_wis_lengths_,
                                        a_g_n_c_wis_strides_,
                                        b_g_k_c_xs_lengths_,
                                        b_g_k_c_xs_strides_,
                                        e_g_n_k_wos_lengths_,
                                        e_g_n_k_wos_strides_,
                                        conv_filter_strides_,
                                        conv_filter_dilations_,
                                        input_left_pads_,
                                        input_right_pads_},
              conv_N_per_block_{conv_to_gemm_transformer_.N_},
              a_grid_desc_ak0_m_ak1_{
                  MakeAGridDescriptor_AK0_M_AK1<ALayout>(conv_to_gemm_transformer_)},
              b_grid_desc_bk0_n_bk1_{
                  MakeBGridDescriptor_BK0_N_BK1<BLayout>(conv_to_gemm_transformer_)},
              e_grid_desc_m_n_{
                  DeviceOp::MakeEGridDescriptor_M_N<ELayout>(conv_to_gemm_transformer_)},
              e_grid_desc_mblock_mperblock_nblock_nperblock_{},
              compute_ptr_offset_of_groups_{},
              compute_ptr_offset_of_n_{},
              a_element_op_{a_element_op},
              b_element_op_{b_element_op},
              cde_element_op_{cde_element_op}
        {
            // A/B/E Batch/N Stride
            compute_ptr_offset_of_groups_.BatchStrideA_ = a_g_n_c_wis_strides_[0];
            compute_ptr_offset_of_groups_.BatchStrideB_ = b_g_k_c_xs_strides_[0];
            compute_ptr_offset_of_n_.BatchStrideA_ = a_g_n_c_wis_strides_[1] * conv_N_per_block_;

            // p_as and p_bs are pointers
            p_a_grid_ = static_cast<const ADataType*>(p_as);
            p_b_grid_ = static_cast<const BDataType*>(p_bs);

            compute_ptr_offset_of_groups_.BatchStrideE_ = e_g_n_k_wos_strides_[0];
            compute_ptr_offset_of_n_.BatchStrideE_ = e_g_n_k_wos_strides_[1] * conv_N_per_block_;

            e_grid_desc_mblock_mperblock_nblock_nperblock_ =
                MakeEGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(e_grid_desc_m_n_);

            if constexpr(is_NGCHW_GKYXC_NGKHW<ALayout, BLayout, ELayout>() ||
                         is_NGCDHW_GKZYXC_NGKDHW<ALayout, BLayout, ELayout>())
            {
                // Use not modified base strides
                a_in_transpose_desc_ =
                    conv_ngchw_to_nhwgc_transformer.template MakeNGCHWTransposeDesc<NDimSpatial>(
                        a_g_n_c_wis_lengths, a_g_n_c_wis_strides);
                a_out_transpose_desc_ =
                    conv_ngchw_to_nhwgc_transformer.template MakeNHWGCTransposeDesc<NDimSpatial>(
                        a_g_n_c_wis_lengths, a_g_n_c_wis_strides);

                e_in_transpose_desc_ =
                    conv_ngchw_to_nhwgc_transformer.template MakeNHWGCTransposeDesc<NDimSpatial>(
                        e_g_n_k_wos_lengths, e_g_n_k_wos_strides);
                e_out_transpose_desc_ =
                    conv_ngchw_to_nhwgc_transformer.template MakeNGCHWTransposeDesc<NDimSpatial>(
                        e_g_n_k_wos_lengths, e_g_n_k_wos_strides);

                elementwise_block_2_ctile_map_transpose_a_ = Block2TileMapElementwise{
                    a_in_transpose_desc_.GetLength(I0), a_in_transpose_desc_.GetLength(I1)};
                elementwise_block_2_ctile_map_transpose_e_ = Block2TileMapElementwise{
                    e_in_transpose_desc_.GetLength(I0), e_in_transpose_desc_.GetLength(I1)};
            }
        }

        std::size_t GetWorkspaceATensorSizeBytes() const
        {
            return sizeof(ADataType) * a_in_transpose_desc_.GetElementSpaceSize();
        }

        std::size_t GetWorkspaceETensorSizeBytes() const
        {
            return sizeof(EDataType) * e_out_transpose_desc_.GetElementSpaceSize();
        }

        std::size_t GetWorkspaceSizeBytes() const
        {
            // Transpose require workspace for A and B
            if constexpr(is_NGCHW_GKYXC_NGKHW<ALayout, BLayout, ELayout>() ||
                         is_NGCDHW_GKZYXC_NGKDHW<ALayout, BLayout, ELayout>())
            {
                return GetWorkspaceATensorSizeBytes() + GetWorkspaceETensorSizeBytes();
            }
            else
            {
                return 0;
            }
        }

        void Print() const
        {
            std::cout << "A[AK0, M, AK1]: " << a_grid_desc_ak0_m_ak1_ << std::endl;
            std::cout << "B[BK0, N, BK1]: " << b_grid_desc_bk0_n_bk1_ << std::endl;
            std::cout << "E[M, N]: " << e_grid_desc_m_n_ << std::endl;
        }

        //  private:
        // pointers (tuple if multi AB, pointer if no)
        const ADataType* p_a_grid_;
        const BDataType* p_b_grid_;
        EDataType* p_e_grid_;

        // for checking IsSupportedArgument()
        std::array<index_t, NDimSpatial + 3> a_g_n_c_wis_lengths_;
        std::array<index_t, NDimSpatial + 3> a_g_n_c_wis_strides_;
        std::array<index_t, NDimSpatial + 3> b_g_k_c_xs_lengths_;
        std::array<index_t, NDimSpatial + 3> b_g_k_c_xs_strides_;
        std::array<index_t, NDimSpatial + 3> e_g_n_k_wos_lengths_;
        std::array<index_t, NDimSpatial + 3> e_g_n_k_wos_strides_;
        std::array<index_t, NDimSpatial> conv_filter_strides_;
        std::array<index_t, NDimSpatial> conv_filter_dilations_;
        std::array<index_t, NDimSpatial> input_left_pads_;
        std::array<index_t, NDimSpatial> input_right_pads_;

        // tensor descriptors for problem definiton
        index_t num_group_;

        ConvToGemmFwdTransformer conv_to_gemm_transformer_;

        index_t conv_N_per_block_;

        // tensor descriptors for block/thread-wise copy
        AGridDesc_AK0_M_AK1 a_grid_desc_ak0_m_ak1_;
        BGridDesc_BK0_N_BK1 b_grid_desc_bk0_n_bk1_;
        EGridDesc_M_N e_grid_desc_m_n_;
        EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock e_grid_desc_mblock_mperblock_nblock_nperblock_;

        // for computing batch offset
        ComputePtrOffsetOfStridedBatch<I1, I1, I0> compute_ptr_offset_of_groups_;
        ComputePtrOffsetOfStridedBatch<I1, I1, I0> compute_ptr_offset_of_n_;

        // element-wise op
        AElementwiseOperation a_element_op_;
        BElementwiseOperation b_element_op_;
        CDEElementwiseOperation cde_element_op_;

        // block-to-e-tile map
        Block2TileMapElementwise elementwise_block_2_ctile_map_transpose_a_,
            elementwise_block_2_ctile_map_transpose_e_;

        NGCHWTransposeDescType a_in_transpose_desc_, e_out_transpose_desc_;
        NHWGCTransposeDescType a_out_transpose_desc_, e_in_transpose_desc_;
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        using Argument = DeviceOp::Argument;

        float RunGemm(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            if(stream_config.log_level_ > 0)
            {
                arg.Print();
            }

            float ave_time = 0;

            constexpr index_t minimum_occupancy =
                BlkGemmPipeSched == BlockGemmPipelineScheduler::Intrawave ? 1 : 2;

            const index_t GemmM = arg.a_grid_desc_ak0_m_ak1_.GetLength(I1);
            const index_t GemmN = arg.b_grid_desc_bk0_n_bk1_.GetLength(I1);
            const index_t GemmK =
                arg.a_grid_desc_ak0_m_ak1_.GetLength(I0) * arg.a_grid_desc_ak0_m_ak1_.GetLength(I2);

            const index_t num_workgroups_per_Conv_N =
                arg.a_g_n_c_wis_lengths_[I1] / arg.conv_N_per_block_;

            index_t gdx, gdy, gdz;
            std::tie(gdx, gdy, gdz) =
                GridwiseGemm::CalculateGridSize(GemmM, GemmN, I1 /*arg.KBatch*/);

            gdy *= arg.num_group_ * num_workgroups_per_Conv_N;

            index_t K_split                  = (GemmK + KPerBlock - 1) / KPerBlock * KPerBlock;
            const bool has_main_k_block_loop = GridwiseGemm::CalculateHasMainKBlockLoop(K_split);

            const ADataType* p_a_grid = arg.p_a_grid_;
            EDataType* p_e_grid       = arg.p_e_grid_;

            if constexpr(is_NGCHW_GKYXC_NGKHW<ALayout, BLayout, ELayout>() ||
                         is_NGCDHW_GKZYXC_NGKDHW<ALayout, BLayout, ELayout>())
            {
                p_a_grid = type_convert<const ADataType*>(arg.p_workspace_);
                p_e_grid = type_convert<EDataType*>(arg.p_workspace_) +
                           arg.GetWorkspaceATensorSizeBytes() / sizeof(EDataType);
            }

            typename GridwiseGemm::Argument gemm_arg{
                p_a_grid, arg.p_b_grid_, p_e_grid, GemmM, GemmN, GemmK, I0, I0, I0, I1};

            const auto Run = [&](const auto& kernel) {
                if(stream_config.flush_cache)
                {
                    typename GridwiseGemm::Argument gemm_arg_ = gemm_arg;
                    ck::utility::RotatingMemWrapper<typename GridwiseGemm::Argument> rotating_mem(
                        gemm_arg_,
                        stream_config.rotating_count,
                        gemm_arg_.M * gemm_arg_.K * sizeof(ADataType),
                        gemm_arg_.K * gemm_arg_.N * sizeof(BDataType));
                    rotating_mem.Print();

                    auto run_flush_cache = [&]() {
                        // flush icache
                        ck::utility::flush_icache();
                        // rotating mem
                        rotating_mem.Next();
                    };

                    ave_time += ck::utility::launch_and_time_kernel_with_preprocess<false>(
                        stream_config,
                        run_flush_cache,
                        kernel,
                        dim3(gdx, gdy, gdz),
                        dim3(BlockSize),
                        0,
                        gemm_arg_,
                        arg.a_grid_desc_ak0_m_ak1_,
                        arg.b_grid_desc_bk0_n_bk1_,
                        arg.e_grid_desc_mblock_mperblock_nblock_nperblock_,
                        arg.compute_ptr_offset_of_groups_,
                        arg.compute_ptr_offset_of_n_,
                        arg.num_group_);
                }
                else
                {
                    ave_time +=
                        launch_and_time_kernel(stream_config,
                                               kernel,
                                               dim3(gdx, gdy, gdz),
                                               dim3(BlockSize),
                                               0,
                                               gemm_arg,
                                               arg.a_grid_desc_ak0_m_ak1_,
                                               arg.b_grid_desc_bk0_n_bk1_,
                                               arg.e_grid_desc_mblock_mperblock_nblock_nperblock_,
                                               arg.compute_ptr_offset_of_groups_,
                                               arg.compute_ptr_offset_of_n_,
                                               arg.num_group_);
                }
            };

            if(has_main_k_block_loop)
            {
                // Tail number always full
                if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v1 ||
                             BlkGemmPipelineVer == BlockGemmPipelineVersion::v3)
                {
                    const auto kernel = kernel_grouped_conv_fwd_xdl_cshuffle_v3<
                        GridwiseGemm,
                        DeviceOp::AGridDesc_AK0_M_AK1,
                        DeviceOp::BGridDesc_BK0_N_BK1,
                        DeviceOp::EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
                        ComputePtrOffsetOfStridedBatch<I1, I1, I0>,
                        true,
                        InMemoryDataOperationEnum::Set,
                        minimum_occupancy>;
                    Run(kernel);
                }
                // Tail number could be One to Seven
                else if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v2)
                {
                    if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::One)
                    {
                        const auto kernel = kernel_grouped_conv_fwd_xdl_cshuffle_v3<
                            GridwiseGemm,
                            DeviceOp::AGridDesc_AK0_M_AK1,
                            DeviceOp::BGridDesc_BK0_N_BK1,
                            DeviceOp::EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
                            ComputePtrOffsetOfStridedBatch<I1, I1, I0>,
                            true,
                            InMemoryDataOperationEnum::Set,
                            minimum_occupancy,
                            TailNumber::One>;
                        Run(kernel);
                    }
                    else if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::Full)
                    {
                        const auto kernel = kernel_grouped_conv_fwd_xdl_cshuffle_v3<
                            GridwiseGemm,
                            DeviceOp::AGridDesc_AK0_M_AK1,
                            DeviceOp::BGridDesc_BK0_N_BK1,
                            DeviceOp::EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
                            ComputePtrOffsetOfStridedBatch<I1, I1, I0>,
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
                            const auto kernel = kernel_grouped_conv_fwd_xdl_cshuffle_v3<
                                GridwiseGemm,
                                DeviceOp::AGridDesc_AK0_M_AK1,
                                DeviceOp::BGridDesc_BK0_N_BK1,
                                DeviceOp::EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
                                ComputePtrOffsetOfStridedBatch<I1, I1, I0>,
                                true,
                                InMemoryDataOperationEnum::Set,
                                minimum_occupancy,
                                TailNumber::Two>;
                            Run(kernel);
                        }
                    }

                    if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 3)
                    {
                        if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::Three)
                        {
                            const auto kernel = kernel_grouped_conv_fwd_xdl_cshuffle_v3<
                                GridwiseGemm,
                                DeviceOp::AGridDesc_AK0_M_AK1,
                                DeviceOp::BGridDesc_BK0_N_BK1,
                                DeviceOp::EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
                                ComputePtrOffsetOfStridedBatch<I1, I1, I0>,
                                true,
                                InMemoryDataOperationEnum::Set,
                                minimum_occupancy,
                                TailNumber::Three>;
                            Run(kernel);
                        }
                    }

                    if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 4)
                    {
                        if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::Four)
                        {
                            const auto kernel = kernel_grouped_conv_fwd_xdl_cshuffle_v3<
                                GridwiseGemm,
                                DeviceOp::AGridDesc_AK0_M_AK1,
                                DeviceOp::BGridDesc_BK0_N_BK1,
                                DeviceOp::EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
                                ComputePtrOffsetOfStridedBatch<I1, I1, I0>,
                                true,
                                InMemoryDataOperationEnum::Set,
                                minimum_occupancy,
                                TailNumber::Four>;
                            Run(kernel);
                        }
                    }

                    if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 5)
                    {
                        if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::Five)
                        {
                            const auto kernel = kernel_grouped_conv_fwd_xdl_cshuffle_v3<
                                GridwiseGemm,
                                DeviceOp::AGridDesc_AK0_M_AK1,
                                DeviceOp::BGridDesc_BK0_N_BK1,
                                DeviceOp::EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
                                ComputePtrOffsetOfStridedBatch<I1, I1, I0>,
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
                            const auto kernel = kernel_grouped_conv_fwd_xdl_cshuffle_v3<
                                GridwiseGemm,
                                DeviceOp::AGridDesc_AK0_M_AK1,
                                DeviceOp::BGridDesc_BK0_N_BK1,
                                DeviceOp::EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
                                ComputePtrOffsetOfStridedBatch<I1, I1, I0>,
                                true,
                                InMemoryDataOperationEnum::Set,
                                minimum_occupancy,
                                TailNumber::Six>;
                            Run(kernel);
                        }
                    }

                    if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 7)
                    {
                        if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::Seven)
                        {
                            const auto kernel = kernel_grouped_conv_fwd_xdl_cshuffle_v3<
                                GridwiseGemm,
                                DeviceOp::AGridDesc_AK0_M_AK1,
                                DeviceOp::BGridDesc_BK0_N_BK1,
                                DeviceOp::EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
                                ComputePtrOffsetOfStridedBatch<I1, I1, I0>,
                                true,
                                InMemoryDataOperationEnum::Set,
                                minimum_occupancy,
                                TailNumber::Seven>;
                            Run(kernel);
                        }
                    }
                }
                // Tail number could be Odd or Even
                else if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v4)
                {
                    if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::Odd)
                    {
                        const auto kernel = kernel_grouped_conv_fwd_xdl_cshuffle_v3_2lds<
                            GridwiseGemm,
                            DeviceOp::AGridDesc_AK0_M_AK1,
                            DeviceOp::BGridDesc_BK0_N_BK1,
                            DeviceOp::EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
                            ComputePtrOffsetOfStridedBatch<I1, I1, I0>,
                            true,
                            InMemoryDataOperationEnum::Set,
                            minimum_occupancy,
                            TailNumber::Odd>;
                        Run(kernel);
                    }
                    else
                    {
                        const auto kernel = kernel_grouped_conv_fwd_xdl_cshuffle_v3_2lds<
                            GridwiseGemm,
                            DeviceOp::AGridDesc_AK0_M_AK1,
                            DeviceOp::BGridDesc_BK0_N_BK1,
                            DeviceOp::EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
                            ComputePtrOffsetOfStridedBatch<I1, I1, I0>,
                            true,
                            InMemoryDataOperationEnum::Set,
                            minimum_occupancy,
                            TailNumber::Even>;
                        Run(kernel);
                    }
                }
                else
                {
                    if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::Odd)
                    {
                        const auto kernel = kernel_grouped_conv_fwd_xdl_cshuffle_v3<
                            GridwiseGemm,
                            DeviceOp::AGridDesc_AK0_M_AK1,
                            DeviceOp::BGridDesc_BK0_N_BK1,
                            DeviceOp::EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
                            ComputePtrOffsetOfStridedBatch<I1, I1, I0>,
                            true,
                            InMemoryDataOperationEnum::Set,
                            minimum_occupancy,
                            TailNumber::Odd>;
                        Run(kernel);
                    }
                    else
                    {
                        const auto kernel = kernel_grouped_conv_fwd_xdl_cshuffle_v3<
                            GridwiseGemm,
                            DeviceOp::AGridDesc_AK0_M_AK1,
                            DeviceOp::BGridDesc_BK0_N_BK1,
                            DeviceOp::EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
                            ComputePtrOffsetOfStridedBatch<I1, I1, I0>,
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
                // Tail number always 1
                if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v1)
                {
                    const auto kernel = kernel_grouped_conv_fwd_xdl_cshuffle_v3<
                        GridwiseGemm,
                        DeviceOp::AGridDesc_AK0_M_AK1,
                        DeviceOp::BGridDesc_BK0_N_BK1,
                        DeviceOp::EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
                        ComputePtrOffsetOfStridedBatch<I1, I1, I0>,
                        false,
                        InMemoryDataOperationEnum::Set,
                        minimum_occupancy>;
                    Run(kernel);
                }
            }

            return ave_time;
        }

        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            float avg_time = 0.f;

            if constexpr(is_NGCHW_GKYXC_NGKHW<ALayout, BLayout, ELayout>() ||
                         is_NGCDHW_GKZYXC_NGKDHW<ALayout, BLayout, ELayout>())
            {
                const index_t grid_size =
                    arg.elementwise_block_2_ctile_map_transpose_a_.CalculateGridSize(
                        arg.a_in_transpose_desc_);

                ADataType* p_a_out_grid = type_convert<ADataType*>(arg.p_workspace_);

                auto kernel_transpose = kernel_elementwise<GridwiseElementwiseInputTranspose,
                                                           ck::Tuple<NGCHWTransposeDescType>,
                                                           ck::Tuple<NHWGCTransposeDescType>,
                                                           ck::Tuple<const ADataType*>,
                                                           ck::Tuple<ADataType*>,
                                                           Block2TileMapElementwise,
                                                           element_wise::PassThrough>;

                avg_time += launch_and_time_kernel(stream_config,
                                                   kernel_transpose,
                                                   dim3(grid_size),
                                                   dim3(ElementwiseBlocksize),
                                                   0,
                                                   make_tuple(arg.a_in_transpose_desc_),
                                                   make_tuple(arg.a_out_transpose_desc_),
                                                   make_tuple(arg.p_a_grid_),
                                                   make_tuple(p_a_out_grid),
                                                   arg.elementwise_block_2_ctile_map_transpose_a_,
                                                   element_wise::PassThrough{});
            }

            avg_time += RunGemm(arg, stream_config);

            if constexpr(is_NGCHW_GKYXC_NGKHW<ALayout, BLayout, ELayout>() ||
                         is_NGCDHW_GKZYXC_NGKDHW<ALayout, BLayout, ELayout>())
            {
                const index_t grid_size =
                    arg.elementwise_block_2_ctile_map_transpose_e_.CalculateGridSize(
                        arg.e_in_transpose_desc_);

                const EDataType* p_e_out_grid =
                    type_convert<EDataType*>(arg.p_workspace_) +
                    arg.GetWorkspaceATensorSizeBytes() / sizeof(EDataType);

                EDataType* p_e_in_grid = arg.p_e_grid_;

                auto kernel_transpose = kernel_elementwise<GridwiseElementwiseOutputTranspose,
                                                           ck::Tuple<NHWGCTransposeDescType>,
                                                           ck::Tuple<NGCHWTransposeDescType>,
                                                           ck::Tuple<const EDataType*>,
                                                           ck::Tuple<EDataType*>,
                                                           Block2TileMapElementwise,
                                                           element_wise::PassThrough>;

                avg_time += launch_and_time_kernel(stream_config,
                                                   kernel_transpose,
                                                   dim3(grid_size),
                                                   dim3(ElementwiseBlocksize),
                                                   0,
                                                   make_tuple(arg.e_in_transpose_desc_),
                                                   make_tuple(arg.e_out_transpose_desc_),
                                                   make_tuple(p_e_out_grid),
                                                   make_tuple(p_e_in_grid),
                                                   arg.elementwise_block_2_ctile_map_transpose_e_,
                                                   element_wise::PassThrough{});
            }

            return avg_time;
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

        const index_t G = arg.b_g_k_c_xs_lengths_[I0];
        const index_t K = arg.b_g_k_c_xs_lengths_[I1];
        const index_t C = arg.b_g_k_c_xs_lengths_[I2];

        // check device
        if(get_device_name() == "gfx908")
        {
            // FIXME: re-enable fp64 when SWDEV-335738 is fixed
            if constexpr(!(is_same_v<AccDataType, float> || is_same_v<AccDataType, int32_t>))
            {
                return false;
            }
        }

        if(!ck::is_xdl_supported())
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
                     is_same_v<ALayout, ctc::NDHWGC> || is_same_v<ALayout, ctc::NGCW> ||
                     is_same_v<ALayout, ctc::NGCHW> || is_same_v<ALayout, ctc::NGCDHW>)
        {
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
            if(!(BBlockTransferSrcVectorDim == 2 && C % BBlockTransferSrcScalarPerVector == 0))
            {
                return false;
            }
        }
        else
        {
            return false;
        }

        if constexpr(is_NGCHW_GKYXC_NGKHW<ALayout, BLayout, ELayout>() ||
                     is_NGCDHW_GKZYXC_NGKDHW<ALayout, BLayout, ELayout>())
        {
            if((G * C) % CDEBlockTransferScalarPerVector_NPerBlock != 0)
            {
                return false;
            }

            if((G * K) % CDEBlockTransferScalarPerVector_NPerBlock != 0)
            {
                return false;
            }

            const index_t input_spatial_acum = ck::accumulate_n<index_t>(
                arg.a_g_n_c_wis_lengths_.begin() + I3, NDimSpatial, 1, std::multiplies<>());
            const index_t output_spatial_acum = ck::accumulate_n<index_t>(
                arg.e_g_n_k_wos_lengths_.begin() + I3, NDimSpatial, 1, std::multiplies<>());

            if(input_spatial_acum % CDEBlockTransferScalarPerVector_NPerBlock != 0)
            {
                return false;
            }

            if(output_spatial_acum % CDEBlockTransferScalarPerVector_NPerBlock != 0)
            {
                return false;
            }
        }

        // check vector access of E
        if constexpr(is_same_v<ELayout, ctc::G_NW_K> || is_same_v<ELayout, ctc::G_NHW_K> ||
                     is_same_v<ELayout, ctc::G_NDHW_K> || is_same_v<ELayout, ctc::GNWK> ||
                     is_same_v<ELayout, ctc::GNHWK> || is_same_v<ELayout, ctc::GNDHWK> ||
                     is_same_v<ELayout, ctc::NWGK> || is_same_v<ELayout, ctc::NHWGK> ||
                     is_same_v<ELayout, ctc::NDHWGK> || is_same_v<ELayout, ctc::NGKW> ||
                     is_same_v<ELayout, ctc::NGKHW> || is_same_v<ELayout, ctc::NGKDHW>)
        {
            if(!(K % CDEBlockTransferScalarPerVector_NPerBlock == 0))
            {
                return false;
            }
        }
        else
        {
            return false;
        }

        // Gridwise gemm v3 doesn't verify descriptors size
        if(!arg.conv_to_gemm_transformer_.AreDescriptorsSmallerThan2GB())
        {
            return false;
        }

        // check Gridwise GEMM
        const index_t GemmM = arg.a_grid_desc_ak0_m_ak1_.GetLength(I1);
        const index_t GemmN = arg.b_grid_desc_bk0_n_bk1_.GetLength(I1);
        const index_t GemmK =
            arg.a_grid_desc_ak0_m_ak1_.GetLength(I0) * arg.a_grid_desc_ak0_m_ak1_.GetLength(I2);

        typename GridwiseGemm::Argument gemm_arg{
            nullptr, nullptr, nullptr, GemmM, GemmN, GemmK, I0, I0, I0, I1 /*KBatch*/};

        return GridwiseGemm::CheckValidity(gemm_arg);
    }

    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static auto MakeArgument(
        const void* p_as,
        const void* p_bs,
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

    static auto
    MakeArgument(const void* p_as,
                 const void* p_bs,
                 const std::array<const void*, NumDTensor>& p_ds,
                 void* p_e,
                 const std::array<long_index_t, NDimSpatial + 3>& a_g_n_c_wis_lengths,
                 const std::array<long_index_t, NDimSpatial + 3>& a_g_n_c_wis_strides,
                 const std::array<long_index_t, NDimSpatial + 3>& b_g_k_c_xs_lengths,
                 const std::array<long_index_t, NDimSpatial + 3>& b_g_k_c_xs_strides,
                 const std::array<std::array<long_index_t, NDimSpatial + 3>, NumDTensor>&
                     ds_g_n_k_wos_lengths,
                 const std::array<std::array<long_index_t, NDimSpatial + 3>, NumDTensor>&
                     ds_g_n_k_wos_strides,
                 const std::array<long_index_t, NDimSpatial + 3>& e_g_n_k_wos_lengths,
                 const std::array<long_index_t, NDimSpatial + 3>& e_g_n_k_wos_strides,
                 const std::array<long_index_t, NDimSpatial>& conv_filter_strides,
                 const std::array<long_index_t, NDimSpatial>& conv_filter_dilations,
                 const std::array<long_index_t, NDimSpatial>& input_left_pads,
                 const std::array<long_index_t, NDimSpatial>& input_right_pads,
                 const AElementwiseOperation& a_element_op,
                 const BElementwiseOperation& b_element_op,
                 const CDEElementwiseOperation& cde_element_op)
    {
        std::array<index_t, NDimSpatial + 3> a_g_n_c_wis_lengths_i32;
        std::array<index_t, NDimSpatial + 3> a_g_n_c_wis_strides_i32;
        std::array<index_t, NDimSpatial + 3> b_g_k_c_xs_lengths_i32;
        std::array<index_t, NDimSpatial + 3> b_g_k_c_xs_strides_i32;
        std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor> ds_g_n_k_wos_lengths_i32;
        std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor> ds_g_n_k_wos_strides_i32;
        std::array<index_t, NDimSpatial + 3> e_g_n_k_wos_lengths_i32;
        std::array<index_t, NDimSpatial + 3> e_g_n_k_wos_strides_i32;
        std::array<index_t, NDimSpatial> conv_filter_strides_i32;
        std::array<index_t, NDimSpatial> conv_filter_dilations_i32;
        std::array<index_t, NDimSpatial> input_left_pads_i32;
        std::array<index_t, NDimSpatial> input_right_pads_i32;

        array_convert(a_g_n_c_wis_lengths_i32, a_g_n_c_wis_lengths);
        array_convert(a_g_n_c_wis_strides_i32, a_g_n_c_wis_strides);
        array_convert(b_g_k_c_xs_lengths_i32, b_g_k_c_xs_lengths);
        array_convert(b_g_k_c_xs_strides_i32, b_g_k_c_xs_strides);
        for(index_t d = 0; d < NumDTensor; d++)
        {
            array_convert(ds_g_n_k_wos_lengths_i32[d], ds_g_n_k_wos_lengths[d]);
            array_convert(ds_g_n_k_wos_strides_i32[d], ds_g_n_k_wos_strides[d]);
        }
        array_convert(e_g_n_k_wos_lengths_i32, e_g_n_k_wos_lengths);
        array_convert(e_g_n_k_wos_strides_i32, e_g_n_k_wos_strides);
        array_convert(conv_filter_strides_i32, conv_filter_strides);
        array_convert(conv_filter_dilations_i32, conv_filter_dilations);
        array_convert(input_left_pads_i32, input_left_pads);
        array_convert(input_right_pads_i32, input_right_pads);

        return Argument{p_as,
                        p_bs,
                        p_ds,
                        p_e,
                        a_g_n_c_wis_lengths_i32,
                        a_g_n_c_wis_strides_i32,
                        b_g_k_c_xs_lengths_i32,
                        b_g_k_c_xs_strides_i32,
                        ds_g_n_k_wos_lengths_i32,
                        ds_g_n_k_wos_strides_i32,
                        e_g_n_k_wos_lengths_i32,
                        e_g_n_k_wos_strides_i32,
                        conv_filter_strides_i32,
                        conv_filter_dilations_i32,
                        input_left_pads_i32,
                        input_right_pads_i32,
                        a_element_op,
                        b_element_op,
                        cde_element_op};
    }

    static auto MakeInvoker() { return Invoker{}; }

    std::unique_ptr<BaseArgument> MakeArgumentPointer(
        const void* p_a,
        const void* p_b,
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

    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_a,
                        const void* p_b,
                        const std::array<const void*, NumDTensor>& p_ds,
                        void* p_e,
                        const std::array<long_index_t, NDimSpatial + 3>& a_g_n_c_wis_lengths,
                        const std::array<long_index_t, NDimSpatial + 3>& a_g_n_c_wis_strides,
                        const std::array<long_index_t, NDimSpatial + 3>& b_g_k_c_xs_lengths,
                        const std::array<long_index_t, NDimSpatial + 3>& b_g_k_c_xs_strides,
                        const std::array<std::array<long_index_t, NDimSpatial + 3>, NumDTensor>&
                            ds_g_n_k_wos_lengths,
                        const std::array<std::array<long_index_t, NDimSpatial + 3>, NumDTensor>&
                            ds_g_n_k_wos_strides,
                        const std::array<long_index_t, NDimSpatial + 3>& e_g_n_k_wos_lengths,
                        const std::array<long_index_t, NDimSpatial + 3>& e_g_n_k_wos_strides,
                        const std::array<long_index_t, NDimSpatial>& conv_filter_strides,
                        const std::array<long_index_t, NDimSpatial>& conv_filter_dilations,
                        const std::array<long_index_t, NDimSpatial>& input_left_pads,
                        const std::array<long_index_t, NDimSpatial>& input_right_pads,
                        const AElementwiseOperation& a_element_op,
                        const BElementwiseOperation& b_element_op,
                        const CDEElementwiseOperation& cde_element_op) override
    {
        std::array<index_t, NDimSpatial + 3> a_g_n_c_wis_lengths_i32;
        std::array<index_t, NDimSpatial + 3> a_g_n_c_wis_strides_i32;
        std::array<index_t, NDimSpatial + 3> b_g_k_c_xs_lengths_i32;
        std::array<index_t, NDimSpatial + 3> b_g_k_c_xs_strides_i32;
        std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor> ds_g_n_k_wos_lengths_i32;
        std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor> ds_g_n_k_wos_strides_i32;
        std::array<index_t, NDimSpatial + 3> e_g_n_k_wos_lengths_i32;
        std::array<index_t, NDimSpatial + 3> e_g_n_k_wos_strides_i32;
        std::array<index_t, NDimSpatial> conv_filter_strides_i32;
        std::array<index_t, NDimSpatial> conv_filter_dilations_i32;
        std::array<index_t, NDimSpatial> input_left_pads_i32;
        std::array<index_t, NDimSpatial> input_right_pads_i32;

        array_convert(a_g_n_c_wis_lengths_i32, a_g_n_c_wis_lengths);
        array_convert(a_g_n_c_wis_strides_i32, a_g_n_c_wis_strides);
        array_convert(b_g_k_c_xs_lengths_i32, b_g_k_c_xs_lengths);
        array_convert(b_g_k_c_xs_strides_i32, b_g_k_c_xs_strides);
        for(index_t d = 0; d < NumDTensor; d++)
        {
            array_convert(ds_g_n_k_wos_lengths_i32[d], ds_g_n_k_wos_lengths[d]);
            array_convert(ds_g_n_k_wos_strides_i32[d], ds_g_n_k_wos_strides[d]);
        }
        array_convert(e_g_n_k_wos_lengths_i32, e_g_n_k_wos_lengths);
        array_convert(e_g_n_k_wos_strides_i32, e_g_n_k_wos_strides);
        array_convert(conv_filter_strides_i32, conv_filter_strides);
        array_convert(conv_filter_dilations_i32, conv_filter_dilations);
        array_convert(input_left_pads_i32, input_left_pads);
        array_convert(input_right_pads_i32, input_right_pads);

        return std::make_unique<Argument>(p_a,
                                          p_b,
                                          p_ds,
                                          p_e,
                                          a_g_n_c_wis_lengths_i32,
                                          a_g_n_c_wis_strides_i32,
                                          b_g_k_c_xs_lengths_i32,
                                          b_g_k_c_xs_strides_i32,
                                          ds_g_n_k_wos_lengths_i32,
                                          ds_g_n_k_wos_strides_i32,
                                          e_g_n_k_wos_lengths_i32,
                                          e_g_n_k_wos_strides_i32,
                                          conv_filter_strides_i32,
                                          conv_filter_dilations_i32,
                                          input_left_pads_i32,
                                          input_right_pads_i32,
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
        str << "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3"
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
            << CShuffleNXdlPerWavePerShuffle << ", "
            << "BlkGemmPipelineScheduler: "
            << BlkGemmPipelineSchedulerToString[BlkGemmPipeSched] << ", "
            << "BlkGemmPipelineVersion: "
            << BlkGemmPipelineVersionToString[BlkGemmPipelineVer]
            << ">";
        // clang-format on

        return str.str();
    }

    size_t GetWorkSpaceSize(const BaseArgument* p_arg) const override
    {
        auto arg = dynamic_cast<const Argument*>(p_arg);
        if(arg)
        {
            return arg->GetWorkspaceSizeBytes();
        }
        else
            throw std::runtime_error(
                "The argument pointer is not an object of "
                "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle::Argument structure!");
    }

    void SetWorkSpacePointer(BaseArgument* p_arg,
                             void* p_workspace,
                             const StreamConfig& = StreamConfig{}) const override
    {
        auto p_arg_ = dynamic_cast<Argument*>(p_arg);
        if(p_arg_)
        {
            p_arg_->p_workspace_ = p_workspace;
        }
        else
            throw std::runtime_error(
                "The argument pointer is not an object of "
                "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle::Argument structure!");
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
