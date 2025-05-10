// SPDX-License-Identifier: MIT
// Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <sstream>

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_conv_bwd_data_multiple_d.hpp"
#include "ck/tensor_operation/gpu/device/convolution_backward_data_specialization.hpp"
#include "ck/tensor_operation/operator_transform/transform_conv_bwd_data_to_gemm_v1.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_multiple_d_wmma_cshuffle.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_utils.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"
#include "ck/host_utility/io.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

// Conv backward data multiple D:
//   input : output image A: [G, N, K, Ho, Wo]
//   input : weight B: [G, K, C, Y, X],
//   input : D0, D1, ... : [G, N, K, Ho, Wo]
//   output : input image E: [G, N, C, Hi, Wi]
//   C = a_op(A) * b_op(B)
//   E = cde_op(C, D0, D1, ...)
template <index_t NDimSpatial,
          typename ALayout,   // output image
          typename BLayout,   // weight
          typename DsLayout,  // bias
          typename ELayout,   // input image
          typename ADataType, // output image
          typename BDataType, // weight
          typename AccDataType,
          typename CShuffleDataType,
          typename DsDataType,       // bias
          typename EDataType,        // input image
          typename AElementwiseOp,   // output image
          typename BElementwiseOp,   // weight
          typename CDEElementwiseOp, // C, bias, and input image
          ConvolutionBackwardDataSpecialization ConvBackwardDataSpecialization,
          ck::index_t BlockSize,
          ck::index_t MPerBlock,
          ck::index_t NPerBlock,
          ck::index_t K0PerBlock,
          ck::index_t K1,
          ck::index_t MPerWMMA,
          ck::index_t NPerWMMA,
          ck::index_t MRepeat,
          ck::index_t NRepeat,
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
          index_t CShuffleMRepeatPerShuffle,
          index_t CShuffleNRepeatPerShuffle,
          typename CDEShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          index_t CDEShuffleBlockTransferScalarPerVector_NPerBlock,
          index_t NumGemmKPrefetchStage   = 1,
          LoopScheduler LoopSched         = make_default_loop_scheduler(),
          ck::PipelineVersion PipelineVer = ck::PipelineVersion::v1>
struct DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffle
    : public DeviceGroupedConvBwdDataMultipleD<NDimSpatial,
                                               ALayout,    // output image
                                               BLayout,    // weight
                                               DsLayout,   // bias
                                               ELayout,    // input image
                                               ADataType,  // output image
                                               BDataType,  // weight
                                               DsDataType, // bias
                                               EDataType,  // input image
                                               AElementwiseOp,
                                               BElementwiseOp,
                                               CDEElementwiseOp>
{
    // TODO: Extend support for more spatial dimensions.
    static_assert(NDimSpatial == 2 || NDimSpatial == 3,
                  "wrong! only implemented for 2D and 3D now");

    using DeviceOp = DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffle;

    static constexpr index_t NumDTensor = DsDataType::Size();

    // TODO: Add support for different A and B data types.
    using ABDataType = ADataType;

    static constexpr auto I0           = Number<0>{};
    static constexpr auto I1           = Number<1>{};
    static constexpr auto I2           = Number<2>{};
    static constexpr auto I3           = Number<3>{};
    static constexpr index_t KPerBlock = K0PerBlock * K1;

    static constexpr auto transform_conv_to_gemm =
        TransformConvBwdDataToGemm_v1<NDimSpatial,
                                      ConvBackwardDataSpecialization,
                                      K1,
                                      K1,
                                      MPerBlock,
                                      NPerBlock,
                                      KPerBlock,
                                      true /* DoPadGemmM */,
                                      true /* DoPadGemmN */>{};

    static auto GetDummyABDsEGridDescriptor()
    {
        const std::array<index_t, NDimSpatial + 3> dummy_tensor_lengths = {1};
        const std::array<index_t, NDimSpatial + 3> dummy_tensor_strides = {1};
        const std::array<index_t, NDimSpatial> dummy_spatial_lengths    = {1};

        const auto a_grid_desc_ak0_m_ak1 =
            transform_conv_to_gemm.template MakeADescriptor_AK0_M_AK1<ALayout>(
                dummy_tensor_lengths,
                dummy_tensor_strides,
                dummy_tensor_lengths,
                dummy_tensor_strides,
                dummy_tensor_lengths,
                dummy_tensor_strides,
                dummy_spatial_lengths,
                dummy_spatial_lengths,
                dummy_spatial_lengths,
                dummy_spatial_lengths,
                dummy_spatial_lengths);

        const auto b_grid_desc_bk0_n_bk1 =
            transform_conv_to_gemm.template MakeBDescriptor_BK0_N_BK1<BLayout>(
                dummy_tensor_lengths,
                dummy_tensor_strides,
                dummy_tensor_lengths,
                dummy_tensor_strides,
                dummy_tensor_lengths,
                dummy_tensor_strides,
                dummy_spatial_lengths,
                dummy_spatial_lengths,
                dummy_spatial_lengths,
                dummy_spatial_lengths,
                dummy_spatial_lengths);

        const auto ds_grid_desc_m_n = generate_tuple(
            [&](auto i) {
                using DLayout = remove_cvref_t<tuple_element_t<i.value, DsLayout>>;

                return transform_conv_to_gemm.template MakeCDescriptor_M_N<DLayout>(
                    dummy_tensor_lengths,
                    dummy_tensor_strides,
                    dummy_tensor_lengths,
                    dummy_tensor_strides,
                    dummy_tensor_lengths,
                    dummy_tensor_strides,
                    dummy_spatial_lengths,
                    dummy_spatial_lengths,
                    dummy_spatial_lengths,
                    dummy_spatial_lengths,
                    dummy_spatial_lengths);
            },
            Number<NumDTensor>{});

        const auto e_grid_desc_m_n =
            transform_conv_to_gemm.template MakeCDescriptor_M_N<ELayout>(dummy_tensor_lengths,
                                                                         dummy_tensor_strides,
                                                                         dummy_tensor_lengths,
                                                                         dummy_tensor_strides,
                                                                         dummy_tensor_lengths,
                                                                         dummy_tensor_strides,
                                                                         dummy_spatial_lengths,
                                                                         dummy_spatial_lengths,
                                                                         dummy_spatial_lengths,
                                                                         dummy_spatial_lengths,
                                                                         dummy_spatial_lengths);

        return make_tuple(
            a_grid_desc_ak0_m_ak1, b_grid_desc_bk0_n_bk1, ds_grid_desc_m_n, e_grid_desc_m_n);
    }

    // desc
    using ABDsEGridDesc = decltype(GetDummyABDsEGridDescriptor());

    using AGridDesc_AK0_M_AK1 = remove_cvref_t<tuple_element_t<0, ABDsEGridDesc>>;
    using BGridDesc_BK0_N_BK1 = remove_cvref_t<tuple_element_t<1, ABDsEGridDesc>>;
    using DsGridDesc_M_N      = remove_cvref_t<tuple_element_t<2, ABDsEGridDesc>>;
    using EGridDesc_M_N       = remove_cvref_t<tuple_element_t<3, ABDsEGridDesc>>;

    // GridwiseGemm
    using GridwiseGemm = GridwiseGemmMultipleD_Wmma<
        // DataType Family
        ADataType,
        BDataType,
        AccDataType,
        CShuffleDataType,
        DsDataType,
        EDataType,
        // InMemory Data Descriptor
        AGridDesc_AK0_M_AK1,
        BGridDesc_BK0_N_BK1,
        DsGridDesc_M_N,
        EGridDesc_M_N,
        // ElementwiseOp Family
        AElementwiseOp,
        BElementwiseOp,
        CDEElementwiseOp,
        InMemoryDataOperationEnum::Set,
        // Tiling Family
        MPerBlock,
        NPerBlock,
        KPerBlock,
        MPerWMMA,
        NPerWMMA,
        K1,
        MRepeat,
        NRepeat,
        // ThreadCluster Family
        BlockSize,
        ABlockTransferThreadClusterLengths_AK0_M_AK1,
        ABlockTransferThreadClusterArrangeOrder,
        ABlockTransferSrcAccessOrder,
        ABlockTransferSrcVectorDim,
        ABlockTransferSrcScalarPerVector,
        ABlockTransferDstScalarPerVector_AK1,
        false,
        true,
        ABlockLdsExtraM,
        BBlockTransferThreadClusterLengths_BK0_N_BK1,
        BBlockTransferThreadClusterArrangeOrder,
        BBlockTransferSrcAccessOrder,
        BBlockTransferSrcVectorDim,
        BBlockTransferSrcScalarPerVector,
        BBlockTransferDstScalarPerVector_BK1,
        false,
        true,
        BBlockLdsExtraN,
        CShuffleMRepeatPerShuffle,
        CShuffleNRepeatPerShuffle,
        CDEShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        CDEShuffleBlockTransferScalarPerVector_NPerBlock,
        NumGemmKPrefetchStage,
        LoopSched,
        PipelineVer>;

    using DsGridDesc_MBlock_MPerBlock_NBlock_NPerBlock =
        decltype(GridwiseGemm::MakeDsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
            DsGridDesc_M_N{}));
    using EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock =
        decltype(GridwiseGemm::MakeEGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
            EGridDesc_M_N{}));

    // Argument
    struct Argument : public BaseArgument
    {
        Argument(const void* p_a,                                 // output image
                 const void* p_b,                                 // weight
                 const std::array<const void*, NumDTensor>& p_ds, // bias
                 void* p_e,                                       // input image
                 const std::array<index_t, NDimSpatial + 3>& a_g_n_k_wos_lengths,
                 const std::array<index_t, NDimSpatial + 3>& a_g_n_k_wos_strides,
                 const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_lengths,
                 const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_strides,
                 const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor>&
                     ds_g_n_c_wis_lengths,
                 const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor>&
                     ds_g_n_c_wis_strides,
                 const std::array<index_t, NDimSpatial + 3>& e_g_n_c_wis_lengths,
                 const std::array<index_t, NDimSpatial + 3>& e_g_n_c_wis_strides,
                 const std::array<index_t, NDimSpatial>& conv_filter_strides,
                 const std::array<index_t, NDimSpatial>& conv_filter_dilations,
                 const std::array<index_t, NDimSpatial>& input_left_pads,
                 const std::array<index_t, NDimSpatial>& input_right_pads,
                 const AElementwiseOp& a_element_op,
                 const BElementwiseOp& b_element_op,
                 const CDEElementwiseOp& cde_element_op)
            : p_a_grid_{static_cast<const ADataType*>(p_a)},
              p_b_grid_{static_cast<const BDataType*>(p_b)},
              p_ds_grid_{},
              p_e_grid_{static_cast<EDataType*>(p_e)},
              num_group_{a_g_n_k_wos_lengths[0]},
              a_element_op_{a_element_op},
              b_element_op_{b_element_op},
              cde_element_op_{cde_element_op},
              a_g_n_k_wos_lengths_{a_g_n_k_wos_lengths},
              a_g_n_k_wos_strides_{a_g_n_k_wos_strides},
              b_g_k_c_xs_lengths_{b_g_k_c_xs_lengths},
              b_g_k_c_xs_strides_{b_g_k_c_xs_strides},
              ds_g_n_c_wis_lengths_{ds_g_n_c_wis_lengths},
              ds_g_n_c_wis_strides_{ds_g_n_c_wis_strides},
              e_g_n_c_wis_lengths_{e_g_n_c_wis_lengths},
              e_g_n_c_wis_strides_{e_g_n_c_wis_strides},
              conv_filter_strides_{conv_filter_strides},
              conv_filter_dilations_{conv_filter_dilations},
              input_left_pads_{input_left_pads},
              input_right_pads_{input_right_pads}
        {
            // populate Ds pointer
            static_for<0, NumDTensor, 1>{}([&](auto i) {
                using DDataType = remove_cvref_t<tuple_element_t<i.value, DsDataType>>;

                p_ds_grid_(i) = static_cast<const DDataType*>(p_ds[i]);
            });

            // A/B/Ds/E Batch Stride
            compute_ptr_offset_of_batch_.BatchStrideA_ = a_g_n_k_wos_strides[0];
            compute_ptr_offset_of_batch_.BatchStrideB_ = b_g_k_c_xs_strides[0];
            compute_ptr_offset_of_batch_.BatchStrideE_ = e_g_n_c_wis_strides[0];

            static_for<0, NumDTensor, 1>{}([&](auto i) {
                compute_ptr_offset_of_batch_.BatchStrideDs_(i) = ds_g_n_c_wis_strides[i][0];
            });

            static constexpr auto NonSpatialDimsNum = Number<3>{};

            static constexpr auto DIdx = Number<NonSpatialDimsNum>{};
            static constexpr auto HIdx =
                NDimSpatial == 2 ? Number<NonSpatialDimsNum>{} : Number<NonSpatialDimsNum + 1>{};
            static constexpr auto WIdx = NDimSpatial == 2 ? Number<NonSpatialDimsNum + 1>{}
                                                          : Number<NonSpatialDimsNum + 2>{};

            static constexpr auto ZIdx = Number<NonSpatialDimsNum>{};
            static constexpr auto YIdx =
                NDimSpatial == 2 ? Number<NonSpatialDimsNum>{} : Number<NonSpatialDimsNum + 1>{};
            static constexpr auto XIdx = NDimSpatial == 2 ? Number<NonSpatialDimsNum + 1>{}
                                                          : Number<NonSpatialDimsNum + 2>{};

            // problem definition
            const index_t Z = b_g_k_c_xs_lengths[ZIdx];
            const index_t Y = b_g_k_c_xs_lengths[YIdx];
            const index_t X = b_g_k_c_xs_lengths[XIdx];

            const index_t ConvStrideD = conv_filter_strides[DIdx - NonSpatialDimsNum];
            const index_t ConvStrideH = conv_filter_strides[HIdx - NonSpatialDimsNum];
            const index_t ConvStrideW = conv_filter_strides[WIdx - NonSpatialDimsNum];

            const index_t ConvDilationD = conv_filter_dilations[DIdx - NonSpatialDimsNum];
            const index_t ConvDilationH = conv_filter_dilations[HIdx - NonSpatialDimsNum];
            const index_t ConvDilationW = conv_filter_dilations[WIdx - NonSpatialDimsNum];

            const auto GcdStrideDilationD = math::gcd(ConvStrideD, ConvDilationD);
            const auto GcdStrideDilationH = math::gcd(ConvStrideH, ConvDilationH);
            const auto GcdStrideDilationW = math::gcd(ConvStrideW, ConvDilationW);

            const auto ZTilde = NDimSpatial == 3 ? ConvStrideD / GcdStrideDilationD : 1;
            const auto YTilde = ConvStrideH / GcdStrideDilationH;
            const auto XTilde = ConvStrideW / GcdStrideDilationW;

            for(index_t i_ztilde = 0; i_ztilde < ZTilde; ++i_ztilde)
            {

                for(index_t i_ytilde = 0; i_ytilde < YTilde; ++i_ytilde)
                {
                    for(index_t i_xtilde = 0; i_xtilde < XTilde; ++i_xtilde)
                    {
                        // check slice is valid
                        const auto ZDotSlice =
                            NDimSpatial == 3 ? math::integer_divide_ceil(Z - i_ztilde, ZTilde) : 1;
                        const auto YDotSlice = math::integer_divide_ceil(Y - i_ytilde, YTilde);
                        const auto XDotSlice = math::integer_divide_ceil(X - i_xtilde, XTilde);

                        if(YDotSlice * XDotSlice * ZDotSlice <= 0)
                        {
                            continue;
                        }

                        std::array<index_t, NDimSpatial> tildes;
                        if constexpr(NDimSpatial == 2)
                        {
                            tildes = {i_ytilde, i_xtilde};
                        }
                        else if constexpr(NDimSpatial == 3)
                        {
                            tildes = {i_ztilde, i_ytilde, i_xtilde};
                        }
                        else
                        {
                            throw std::runtime_error("wrong! only implemented for 2D and 3D now");
                        }

                        const auto a_grid_desc_ak0_m_ak1 =
                            transform_conv_to_gemm.template MakeADescriptor_AK0_M_AK1<ALayout>(
                                a_g_n_k_wos_lengths,
                                a_g_n_k_wos_strides,
                                b_g_k_c_xs_lengths,
                                b_g_k_c_xs_strides,
                                e_g_n_c_wis_lengths,
                                e_g_n_c_wis_strides,
                                conv_filter_strides,
                                conv_filter_dilations,
                                input_left_pads,
                                input_right_pads,
                                tildes);

                        const auto b_grid_desc_bk0_n_bk1 =
                            transform_conv_to_gemm.template MakeBDescriptor_BK0_N_BK1<BLayout>(
                                a_g_n_k_wos_lengths,
                                a_g_n_k_wos_strides,
                                b_g_k_c_xs_lengths,
                                b_g_k_c_xs_strides,
                                e_g_n_c_wis_lengths,
                                e_g_n_c_wis_strides,
                                conv_filter_strides,
                                conv_filter_dilations,
                                input_left_pads,
                                input_right_pads,
                                tildes);

                        DsGridDesc_M_N ds_grid_desc_m_n;

                        // populate Ds desc
                        static_for<0, NumDTensor, 1>{}([&](auto i) {
                            using DLayout = remove_cvref_t<tuple_element_t<i.value, DsLayout>>;

                            ds_grid_desc_m_n(i) =
                                transform_conv_to_gemm.template MakeCDescriptor_M_N<DLayout>(
                                    a_g_n_k_wos_lengths,
                                    a_g_n_k_wos_strides,
                                    b_g_k_c_xs_lengths,
                                    b_g_k_c_xs_strides,
                                    ds_g_n_c_wis_lengths[i],
                                    ds_g_n_c_wis_strides[i],
                                    conv_filter_strides,
                                    conv_filter_dilations,
                                    input_left_pads,
                                    input_right_pads,
                                    tildes);
                        });

                        const auto e_grid_desc_m_n =
                            transform_conv_to_gemm.template MakeCDescriptor_M_N<ELayout>(
                                a_g_n_k_wos_lengths,
                                a_g_n_k_wos_strides,
                                b_g_k_c_xs_lengths,
                                b_g_k_c_xs_strides,
                                e_g_n_c_wis_lengths,
                                e_g_n_c_wis_strides,
                                conv_filter_strides,
                                conv_filter_dilations,
                                input_left_pads,
                                input_right_pads,
                                tildes);

                        // for check validity
                        ds_grid_desc_m_n_container_.push_back(ds_grid_desc_m_n);
                        e_grid_desc_m_n_container_.push_back(e_grid_desc_m_n);

                        // desc for blockwise copy
                        a_grid_desc_ak0_m_ak1_container_.push_back(a_grid_desc_ak0_m_ak1);
                        b_grid_desc_bk0_n_bk1_container_.push_back(b_grid_desc_bk0_n_bk1);

                        // block-to-e-tile-map
                        auto block_2_ctile_map = GridwiseGemm::MakeDefaultBlock2CTileMap(
                            e_grid_desc_m_n, 1 /* M01 */, 1 /* N01 */);

                        block_2_ctile_map_container_.push_back(block_2_ctile_map);

                        ds_grid_desc_mblock_mperblock_nblock_nperblock_container_.push_back(
                            GridwiseGemm::MakeDsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                                ds_grid_desc_m_n));
                        e_grid_desc_mblock_mperblock_nblock_nperblock_container_.push_back(
                            GridwiseGemm::MakeEGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                                e_grid_desc_m_n));
                    }
                }
            }
        }

        void Print() const
        {
            for(std::size_t i = 0; i < a_grid_desc_ak0_m_ak1_container_.size(); i++)
            {
                std::cout << "a_grid_desc_ak0_m_ak1_container_"
                          << a_grid_desc_ak0_m_ak1_container_[i] << std::endl;

                std::cout << "b_grid_desc_bk0_n_bk1_container_"
                          << b_grid_desc_bk0_n_bk1_container_[i] << std::endl;

                static_for<0, NumDTensor, 1>{}([&](auto j) {
                    std::cout << "ds_grid_desc_mblock_mperblock_nblock_nperblock_container_"
                              << ds_grid_desc_mblock_mperblock_nblock_nperblock_container_[i][j]
                              << std::endl;
                });

                std::cout << "e_grid_desc_mblock_mperblock_nblock_nperblock_container_"
                          << e_grid_desc_mblock_mperblock_nblock_nperblock_container_[i]
                          << std::endl;
            }
        }

        // pointers
        const ADataType* p_a_grid_;
        const BDataType* p_b_grid_;
        typename GridwiseGemm::DsGridPointer p_ds_grid_;
        EDataType* p_e_grid_;

        // tensor descriptor for problem definition
        index_t num_group_;
        std::vector<DsGridDesc_M_N> ds_grid_desc_m_n_container_;
        std::vector<EGridDesc_M_N> e_grid_desc_m_n_container_;

        // tensor descriptor for block-wise copy
        std::vector<AGridDesc_AK0_M_AK1> a_grid_desc_ak0_m_ak1_container_;
        std::vector<BGridDesc_BK0_N_BK1> b_grid_desc_bk0_n_bk1_container_;
        std::vector<DsGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>
            ds_grid_desc_mblock_mperblock_nblock_nperblock_container_;
        std::vector<EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>
            e_grid_desc_mblock_mperblock_nblock_nperblock_container_;

        // block-to-e-tile map
        std::vector<typename GridwiseGemm::DefaultBlock2CTileMap> block_2_ctile_map_container_;

        // for computing batch offset
        ComputePtrOffsetOfStridedBatch<I1, I1, NumDTensor> compute_ptr_offset_of_batch_;

        // element-wise op
        AElementwiseOp a_element_op_;
        BElementwiseOp b_element_op_;
        CDEElementwiseOp cde_element_op_;

        // for checking IsSupportedArgument()
        std::array<index_t, NDimSpatial + 3> a_g_n_k_wos_lengths_;
        std::array<index_t, NDimSpatial + 3> a_g_n_k_wos_strides_;
        std::array<index_t, NDimSpatial + 3> b_g_k_c_xs_lengths_;
        std::array<index_t, NDimSpatial + 3> b_g_k_c_xs_strides_;
        std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor> ds_g_n_c_wis_lengths_;
        std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor> ds_g_n_c_wis_strides_;
        std::array<index_t, NDimSpatial + 3> e_g_n_c_wis_lengths_;
        std::array<index_t, NDimSpatial + 3> e_g_n_c_wis_strides_;
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

            float ave_time = 0;

            for(std::size_t i = 0; i < arg.a_grid_desc_ak0_m_ak1_container_.size(); i++)
            {
                const index_t grid_size = arg.block_2_ctile_map_container_[i].CalculateGridSize(
                                              arg.e_grid_desc_m_n_container_[i]) *
                                          arg.num_group_;

                const auto GemmK = arg.a_grid_desc_ak0_m_ak1_container_[i].GetLength(I0) *
                                   arg.a_grid_desc_ak0_m_ak1_container_[i].GetLength(I2);

                auto launch_kernel = [&](auto has_main_k_block_loop) {
                    constexpr bool has_main_loop = has_main_k_block_loop.value;

                    const auto kernel = kernel_grouped_conv_multiple_d_wmma_cshuffle<
                        GridwiseGemm,
                        ADataType,
                        BDataType,
                        typename GridwiseGemm::DsGridPointer,
                        EDataType,
                        AElementwiseOp,
                        BElementwiseOp,
                        CDEElementwiseOp,
                        DeviceOp::AGridDesc_AK0_M_AK1,
                        DeviceOp::BGridDesc_BK0_N_BK1,
                        typename GridwiseGemm::DsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock,
                        typename GridwiseGemm::EGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock,
                        remove_reference_t<typename GridwiseGemm::DefaultBlock2CTileMap>,
                        ComputePtrOffsetOfStridedBatch<I1, I1, NumDTensor>,
                        has_main_loop>;

                    return launch_and_time_kernel(
                        stream_config,
                        kernel,
                        dim3(grid_size),
                        dim3(BlockSize),
                        0,
                        arg.p_a_grid_,
                        arg.p_b_grid_,
                        arg.p_ds_grid_,
                        arg.p_e_grid_,
                        arg.a_element_op_,
                        arg.b_element_op_,
                        arg.cde_element_op_,
                        arg.a_g_n_k_wos_lengths_[0], // Group count
                        arg.a_grid_desc_ak0_m_ak1_container_[i],
                        arg.b_grid_desc_bk0_n_bk1_container_[i],
                        arg.ds_grid_desc_mblock_mperblock_nblock_nperblock_container_[i],
                        arg.e_grid_desc_mblock_mperblock_nblock_nperblock_container_[i],
                        arg.block_2_ctile_map_container_[i],
                        arg.compute_ptr_offset_of_batch_);
                };

                if(GridwiseGemm::CalculateHasMainKBlockLoop(GemmK))
                {
                    ave_time += launch_kernel(integral_constant<bool, true>{});
                }
                else
                {
                    ave_time += launch_kernel(integral_constant<bool, false>{});
                }
            }

            return ave_time;
        }

        float Run(const BaseArgument* p_arg,
                  const StreamConfig& stream_config = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), stream_config);
        }
    };

    static bool IsSupportedArgument(const Argument& arg)
    {
        // check device
        if(ck::is_gfx11_supported())
        {
            if constexpr(!(is_same_v<AccDataType, float> || is_same_v<AccDataType, int32_t>))
            {
                return false;
            }
        }
        else
        {
            return false;
        }

        const index_t ConvK = arg.b_g_k_c_xs_lengths_[1];
        const index_t ConvC = arg.b_g_k_c_xs_lengths_[2];

        // Specialization
        if constexpr(ConvBackwardDataSpecialization ==
                     ConvolutionBackwardDataSpecialization::Filter1x1Stride1Pad0)
        {
            // check if it's a 1x1 convolution with stride=1 and no padding
            for(int i = 0; i < NDimSpatial; i++)
            {
                if(!(arg.b_g_k_c_xs_lengths_[3 + i] == 1 && arg.conv_filter_strides_[i] == 1 &&
                     arg.input_left_pads_[i] == 0 && arg.input_right_pads_[i] == 0))
                {
                    return false;
                }
            }
        }

        // vector load for A matrix from global memory to LDS
        if constexpr(is_same_v<ALayout, tensor_layout::convolution::GNHWK> ||
                     is_same_v<ALayout, tensor_layout::convolution::GNDHWK> ||
                     is_same_v<ALayout, tensor_layout::convolution::NHWGK> ||
                     is_same_v<ALayout, tensor_layout::convolution::NDHWGK>)
        {
            if(!(ABlockTransferSrcVectorDim == 2 && ConvK % ABlockTransferSrcScalarPerVector == 0))
            {
                return false;
            }
        }
        else
        {
            return false;
        }

        // vector load for B matrix from global memory to LDS
        if constexpr(is_same_v<BLayout, tensor_layout::convolution::GKYXC> ||
                     is_same_v<BLayout, tensor_layout::convolution::GKZYXC>)
        {
            if(!(BBlockTransferSrcVectorDim == 1 && ConvC % BBlockTransferSrcScalarPerVector == 0))
            {
                return false;
            }
        }
        else
        {
            return false;
        }

        // vector store for Ds
        bool ds_valid = true;

        static_for<0, NumDTensor, 1>{}([&](auto i) {
            using DLayout = remove_cvref_t<tuple_element_t<i.value, DsLayout>>;

            if constexpr(is_same_v<DLayout, tensor_layout::convolution::GNHWC> ||
                         is_same_v<DLayout, tensor_layout::convolution::GNDHWC> ||
                         is_same_v<DLayout, tensor_layout::convolution::NHWGC> ||
                         is_same_v<DLayout, tensor_layout::convolution::NDHWGC> ||
                         is_same_v<DLayout, tensor_layout::convolution::G_NHW_C> ||
                         is_same_v<DLayout, tensor_layout::convolution::GC> ||
                         is_same_v<DLayout, tensor_layout::convolution::G_C>)
            {
                // vector load D matrix from global memory
                if(!(ConvC % CDEShuffleBlockTransferScalarPerVector_NPerBlock == 0))
                {
                    ds_valid = false;
                }
            }
            else
            {
                ds_valid = false;
            }
        });

        if(!ds_valid)
        {
            return false;
        }

        // vector store for E
        if constexpr(is_same_v<ELayout, tensor_layout::convolution::GNHWC> ||
                     is_same_v<ELayout, tensor_layout::convolution::GNDHWC> ||
                     is_same_v<ELayout, tensor_layout::convolution::NHWGC> ||
                     is_same_v<ELayout, tensor_layout::convolution::NDHWGC>)
        {
            // vector store C matrix into global memory
            if(!(ConvC % CDEShuffleBlockTransferScalarPerVector_NPerBlock == 0))
            {
                return false;
            }
        }
        else
        {
            return false;
        }

        // Gridwise GEMM size
        for(std::size_t i = 0; i < arg.a_grid_desc_ak0_m_ak1_container_.size(); i++)
        {
            if(!GridwiseGemm::CheckValidity(arg.a_grid_desc_ak0_m_ak1_container_[i],
                                            arg.b_grid_desc_bk0_n_bk1_container_[i],
                                            arg.ds_grid_desc_m_n_container_[i],
                                            arg.e_grid_desc_m_n_container_[i],
                                            arg.block_2_ctile_map_container_[i]))
            {
                return false;
            }
        }

        return true;
    }

    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static auto
    MakeArgument(const void* p_a,                                                 // output image
                 const void* p_b,                                                 // weight
                 const std::array<const void*, NumDTensor>& p_ds,                 // bias
                 void* p_e,                                                       // input image
                 const std::array<index_t, NDimSpatial + 3>& a_g_n_k_wos_lengths, // output image
                 const std::array<index_t, NDimSpatial + 3>& a_g_n_k_wos_strides, // output image
                 const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_lengths,  // weight
                 const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_strides,  // weight
                 const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor>&
                     ds_g_n_c_wis_lengths, // bias
                 const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor>&
                     ds_g_n_c_wis_strides,                                        // bias
                 const std::array<index_t, NDimSpatial + 3>& e_g_n_c_wis_lengths, // input image
                 const std::array<index_t, NDimSpatial + 3>& e_g_n_c_wis_strides, // input image
                 const std::array<index_t, NDimSpatial>& conv_filter_strides,
                 const std::array<index_t, NDimSpatial>& conv_filter_dilations,
                 const std::array<index_t, NDimSpatial>& input_left_pads,
                 const std::array<index_t, NDimSpatial>& input_right_pads,
                 const AElementwiseOp& a_element_op,
                 const BElementwiseOp& b_element_op,
                 const CDEElementwiseOp& cde_element_op)
    {
        return Argument{p_a,
                        p_b,
                        p_ds,
                        p_e,
                        a_g_n_k_wos_lengths,
                        a_g_n_k_wos_strides,
                        b_g_k_c_xs_lengths,
                        b_g_k_c_xs_strides,
                        ds_g_n_c_wis_lengths,
                        ds_g_n_c_wis_strides,
                        e_g_n_c_wis_lengths,
                        e_g_n_c_wis_strides,
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
        const void* p_a,                                                 // output image
        const void* p_b,                                                 // weight
        const std::array<const void*, NumDTensor>& p_ds,                 // bias
        void* p_e,                                                       // input image
        const std::array<index_t, NDimSpatial + 3>& a_g_n_k_wos_lengths, // output image
        const std::array<index_t, NDimSpatial + 3>& a_g_n_k_wos_strides, // output image
        const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_lengths,  // weight
        const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_strides,  // weight
        const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor>&
            ds_g_n_c_wis_lengths, // bias
        const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor>&
            ds_g_n_c_wis_strides,                                        // bias
        const std::array<index_t, NDimSpatial + 3>& e_g_n_c_wis_lengths, // input image
        const std::array<index_t, NDimSpatial + 3>& e_g_n_c_wis_strides, // input image
        const std::array<index_t, NDimSpatial>& conv_filter_strides,
        const std::array<index_t, NDimSpatial>& conv_filter_dilations,
        const std::array<index_t, NDimSpatial>& input_left_pads,
        const std::array<index_t, NDimSpatial>& input_right_pads,
        const AElementwiseOp& a_element_op,
        const BElementwiseOp& b_element_op,
        const CDEElementwiseOp& cde_element_op) override
    {
        return std::make_unique<Argument>(p_a,
                                          p_b,
                                          p_ds,
                                          p_e,
                                          a_g_n_k_wos_lengths,
                                          a_g_n_k_wos_strides,
                                          b_g_k_c_xs_lengths,
                                          b_g_k_c_xs_strides,
                                          ds_g_n_c_wis_lengths,
                                          ds_g_n_c_wis_strides,
                                          e_g_n_c_wis_lengths,
                                          e_g_n_c_wis_strides,
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
        str << "DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffle"
            << "<"
            << BlockSize << ", "
            << MPerBlock << ", "
            << NPerBlock << ", "
            << KPerBlock << ", "
            << getConvBackwardDataSpecializationString(ConvBackwardDataSpecialization) << ", "
            << K1 << ", "
            << ABlockTransferSrcScalarPerVector << ", "
            << BBlockTransferSrcScalarPerVector
            << ">";
        // clang-format on

        return str.str();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
