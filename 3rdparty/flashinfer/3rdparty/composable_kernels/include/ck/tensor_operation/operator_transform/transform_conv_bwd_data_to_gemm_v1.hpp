// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/convolution_backward_data_specialization.hpp"
#include "ck/tensor_operation/gpu/device/matrix_padder.hpp"

namespace ck {
namespace tensor_operation {

namespace {
template <
    index_t NDimSpatial,
    typename ALayout,
    ck::tensor_operation::device::ConvolutionBackwardDataSpecialization ConvBwdDataSpecialization>
constexpr auto make_out_grid_desc(const index_t N,
                                  const index_t Do,
                                  const index_t Ho,
                                  const index_t Wo,
                                  const index_t K,
                                  const std::array<index_t, NDimSpatial + 3>& out_g_n_k_wos_strides)
{
    const auto KStride = Number<1>{};

    if constexpr(is_same_v<ALayout, tensor_layout::convolution::NHWGK>)
    {
        const index_t NStride  = out_g_n_k_wos_strides[1];
        const index_t HiStride = out_g_n_k_wos_strides[3];
        const index_t WiStride = out_g_n_k_wos_strides[4];
        if constexpr(ConvBwdDataSpecialization ==
                     ck::tensor_operation::device::ConvolutionBackwardDataSpecialization::
                         Filter1x1Stride1Pad0)
        {

            return make_naive_tensor_descriptor(make_tuple(N * Ho * Wo, K),
                                                make_tuple(WiStride, KStride));
        }
        else
        {
            return make_naive_tensor_descriptor(make_tuple(N, Ho, Wo, K),
                                                make_tuple(NStride, HiStride, WiStride, KStride));
        }
    }
    else if constexpr(is_same_v<ALayout, tensor_layout::convolution::NDHWGK>)
    {
        const index_t NStride  = out_g_n_k_wos_strides[1];
        const index_t DoStride = out_g_n_k_wos_strides[3];
        const index_t HoStride = out_g_n_k_wos_strides[4];
        const index_t WoStride = out_g_n_k_wos_strides[5];
        if constexpr(ConvBwdDataSpecialization ==
                     ck::tensor_operation::device::ConvolutionBackwardDataSpecialization::
                         Filter1x1Stride1Pad0)
        {

            return make_naive_tensor_descriptor(make_tuple(N * Do * Ho * Wo, K),
                                                make_tuple(WoStride, KStride));
        }
        else
        {
            return make_naive_tensor_descriptor(
                make_tuple(N, Do, Ho, Wo, K),
                make_tuple(NStride, DoStride, HoStride, WoStride, KStride));
        }
    }
    else if constexpr(is_same_v<ALayout, tensor_layout::convolution::GNHWK>)
    {
        // assume packed
        if constexpr(ConvBwdDataSpecialization ==
                     ck::tensor_operation::device::ConvolutionBackwardDataSpecialization::
                         Filter1x1Stride1Pad0)
        {
            return make_naive_tensor_descriptor_packed(make_tuple(N * Ho * Wo, K));
        }
        else
        {
            return make_naive_tensor_descriptor_packed(make_tuple(N, Ho, Wo, K));
        }
    }
    else if constexpr(is_same_v<ALayout, tensor_layout::convolution::GNDHWK>)
    {
        // assume packed
        if constexpr(ConvBwdDataSpecialization ==
                     ck::tensor_operation::device::ConvolutionBackwardDataSpecialization::
                         Filter1x1Stride1Pad0)
        {
            return make_naive_tensor_descriptor_packed(make_tuple(N * Do * Ho * Wo, K));
        }
        else
        {
            return make_naive_tensor_descriptor_packed(make_tuple(N, Do, Ho, Wo, K));
        }
    }
    else
    {
        throw std::runtime_error("wrong! unsupported layout: " + ALayout::name());
    }
}

template <typename BLayout>
constexpr auto make_wei_grid_desc(
    const index_t K, const index_t Z, const index_t Y, const index_t X, const index_t C)
{

    if constexpr(is_same_v<BLayout, tensor_layout::convolution::GKYXC>)
    {
        return make_naive_tensor_descriptor_packed(make_tuple(K, Y, X, C));
    }
    else if constexpr(is_same_v<BLayout, tensor_layout::convolution::GKZYXC>)
    {
        return make_naive_tensor_descriptor_packed(make_tuple(K, Z, Y, X, C));
    }
    else
    {
        throw std::runtime_error("wrong! unsupported layout: " + BLayout::name());
    }
}

template <index_t NDimSpatial, typename CLayout>
constexpr auto make_in_grid_desc(const index_t N,
                                 const index_t Di,
                                 const index_t Hi,
                                 const index_t Wi,
                                 const index_t C,
                                 const std::array<index_t, NDimSpatial + 3>& in_g_n_c_wis_strides)
{

    if constexpr(is_same_v<CLayout, tensor_layout::convolution::GNHWC> ||
                 is_same_v<CLayout, tensor_layout::convolution::NHWGC> ||
                 is_same_v<CLayout, tensor_layout::convolution::G_NHW_C>)
    {
        return make_naive_tensor_descriptor(make_tuple(N, Hi, Wi, C),
                                            make_tuple(in_g_n_c_wis_strides[1],
                                                       in_g_n_c_wis_strides[3],
                                                       in_g_n_c_wis_strides[4],
                                                       in_g_n_c_wis_strides[2]));
    }
    else if constexpr(is_same_v<CLayout, tensor_layout::convolution::GNDHWC> ||
                      is_same_v<CLayout, tensor_layout::convolution::NDHWGC>)
    {
        return make_naive_tensor_descriptor(make_tuple(N, Di, Hi, Wi, C),
                                            make_tuple(in_g_n_c_wis_strides[1],
                                                       in_g_n_c_wis_strides[3],
                                                       in_g_n_c_wis_strides[4],
                                                       in_g_n_c_wis_strides[5],
                                                       in_g_n_c_wis_strides[2]));
    }
    else
    {
        throw std::runtime_error("wrong! unsupported layout: " + CLayout::name());
    }
}

} // namespace

template <
    index_t NDimSpatial,
    ck::tensor_operation::device::ConvolutionBackwardDataSpecialization ConvBwdDataSpecialization,
    index_t AK1,
    index_t BK1,
    index_t GemmMPerBlock,
    index_t GemmNPerBlock,
    index_t GemmKPerBlock,
    bool DoPadGemmM,
    bool DoPadGemmN>
struct TransformConvBwdDataToGemm_v1
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};

    static constexpr auto NonSpatialDimsNum = Number<3>{};

    static constexpr auto DIdx = Number<NonSpatialDimsNum>{};
    static constexpr auto HIdx =
        NDimSpatial == 2 ? Number<NonSpatialDimsNum>{} : Number<NonSpatialDimsNum + 1>{};
    static constexpr auto WIdx =
        NDimSpatial == 2 ? Number<NonSpatialDimsNum + 1>{} : Number<NonSpatialDimsNum + 2>{};

    static constexpr auto ZIdx = Number<NonSpatialDimsNum>{};
    static constexpr auto YIdx =
        NDimSpatial == 2 ? Number<NonSpatialDimsNum>{} : Number<NonSpatialDimsNum + 1>{};
    static constexpr auto XIdx =
        NDimSpatial == 2 ? Number<NonSpatialDimsNum + 1>{} : Number<NonSpatialDimsNum + 2>{};

    template <typename ALayout,
              typename std::enable_if<(NDimSpatial == 2 || NDimSpatial == 3) &&
                                          (is_same_v<ALayout, tensor_layout::convolution::GNHWK> ||
                                           is_same_v<ALayout, tensor_layout::convolution::GNDHWK> ||
                                           is_same_v<ALayout, tensor_layout::convolution::NHWGK> ||
                                           is_same_v<ALayout, tensor_layout::convolution::NDHWGK>),
                                      bool>::type = false>
    static auto MakeADescriptor_AK0_M_AK1(
        const std::array<index_t, NDimSpatial + 3>& out_g_n_k_wos_lengths,
        const std::array<index_t, NDimSpatial + 3>& out_g_n_k_wos_strides,
        const std::array<index_t, NDimSpatial + 3>& wei_g_k_c_xs_lengths,
        const std::array<index_t, NDimSpatial + 3>& /* wei_g_k_c_xs_strides */,
        const std::array<index_t, NDimSpatial + 3>& in_g_n_c_wis_lengths,
        const std::array<index_t, NDimSpatial + 3>& /* in_g_n_c_wis_strides */,
        const std::array<index_t, NDimSpatial>& conv_filter_strides,
        const std::array<index_t, NDimSpatial>& conv_filter_dilations,
        const std::array<index_t, NDimSpatial>& input_left_pads,
        const std::array<index_t, NDimSpatial>& /* input_right_pads */,
        const std::array<index_t, NDimSpatial>& tildes)
    {
        index_t i_ztilde = tildes[ZIdx - NonSpatialDimsNum];
        index_t i_ytilde = tildes[YIdx - NonSpatialDimsNum];
        index_t i_xtilde = tildes[XIdx - NonSpatialDimsNum];

        const index_t N = in_g_n_c_wis_lengths[1];
        const index_t K = wei_g_k_c_xs_lengths[1];

        const index_t Di = NDimSpatial == 3 ? in_g_n_c_wis_lengths[DIdx] : 1;
        const index_t Hi = in_g_n_c_wis_lengths[HIdx];
        const index_t Wi = in_g_n_c_wis_lengths[WIdx];

        const index_t Do = NDimSpatial == 3 ? out_g_n_k_wos_lengths[DIdx] : 1;
        const index_t Ho = out_g_n_k_wos_lengths[HIdx];
        const index_t Wo = out_g_n_k_wos_lengths[WIdx];

        const index_t Z = NDimSpatial == 3 ? wei_g_k_c_xs_lengths[ZIdx] : 1;
        const index_t Y = wei_g_k_c_xs_lengths[YIdx];
        const index_t X = wei_g_k_c_xs_lengths[XIdx];

        const index_t InLeftPadD = input_left_pads[DIdx - NonSpatialDimsNum];
        const index_t InLeftPadH = input_left_pads[HIdx - NonSpatialDimsNum];
        const index_t InLeftPadW = input_left_pads[WIdx - NonSpatialDimsNum];

        const index_t ConvStrideD = conv_filter_strides[DIdx - NonSpatialDimsNum];
        const index_t ConvStrideH = conv_filter_strides[HIdx - NonSpatialDimsNum];
        const index_t ConvStrideW = conv_filter_strides[WIdx - NonSpatialDimsNum];

        const index_t ConvDilationD = conv_filter_dilations[DIdx - NonSpatialDimsNum];
        const index_t ConvDilationH = conv_filter_dilations[HIdx - NonSpatialDimsNum];
        const index_t ConvDilationW = conv_filter_dilations[WIdx - NonSpatialDimsNum];

        // n_do_ho_wo_k for 3d or n_ho_wo_k for 2d
        const auto out_grid_desc =
            make_out_grid_desc<NDimSpatial, ALayout, ConvBwdDataSpecialization>(
                N, Do, Ho, Wo, K, out_g_n_k_wos_strides);

        if constexpr(ConvBwdDataSpecialization ==
                     ck::tensor_operation::device::ConvolutionBackwardDataSpecialization::
                         Filter1x1Stride1Pad0)
        {
            const index_t AK0 = math::integer_divide_ceil(K, AK1);

            // A: output tensor
            const auto out_gemmak0_gemmmraw_gemmak1_grid_desc = transform_tensor_descriptor(
                out_grid_desc,
                make_tuple(make_pass_through_transform(N * Do * Ho * Wo),
                           make_unmerge_transform(make_tuple(AK0, AK1))),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<1>{}, Sequence<0, 2>{}));

            const auto out_gemmak0_gemmm_gemmak1_grid_desc =
                ck::tensor_operation::device::PadTensorDescriptor(
                    out_gemmak0_gemmmraw_gemmak1_grid_desc,
                    make_tuple(AK0, GemmMPerBlock, AK1),
                    Sequence<false, DoPadGemmM, false>{});

            return out_gemmak0_gemmm_gemmak1_grid_desc;
        }
        else
        {
            const auto GcdStrideDilationD = math::gcd(ConvStrideD, ConvDilationD);
            const auto GcdStrideDilationH = math::gcd(ConvStrideH, ConvDilationH);
            const auto GcdStrideDilationW = math::gcd(ConvStrideW, ConvDilationW);

            const auto ZTilde = ConvStrideD / GcdStrideDilationD;
            const auto YTilde = ConvStrideH / GcdStrideDilationH;
            const auto XTilde = ConvStrideW / GcdStrideDilationW;

            const auto ZDot = math::integer_divide_ceil(Z, ZTilde);
            const auto YDot = math::integer_divide_ceil(Y, YTilde);
            const auto XDot = math::integer_divide_ceil(X, XTilde);

            const auto DTilde =
                Do + math::integer_divide_ceil(ConvDilationD * (Z - I1), ConvStrideD);
            const auto HTilde =
                Ho + math::integer_divide_ceil(ConvDilationH * (Y - I1), ConvStrideH);
            const auto WTilde =
                Wo + math::integer_divide_ceil(ConvDilationW * (X - I1), ConvStrideW);

            // only work on HTilde and WTilde that contribute to non-padding area of input tensor
            const auto IDTildeSliceBegin = math::integer_divide_floor(
                math::max(I0, InLeftPadD - ConvDilationD * (ZTilde - I1)), ConvStrideD);
            const auto IHTildeSliceBegin = math::integer_divide_floor(
                math::max(I0, InLeftPadH - ConvDilationH * (YTilde - I1)), ConvStrideH);
            const auto IWTildeSliceBegin = math::integer_divide_floor(
                math::max(I0, InLeftPadW - ConvDilationW * (XTilde - I1)), ConvStrideW);

            const auto IDTildeSliceEnd = math::min(
                DTilde, math::integer_divide_ceil(InLeftPadD + Di - I1, ConvStrideD) + I1);
            const auto IHTildeSliceEnd = math::min(
                HTilde, math::integer_divide_ceil(InLeftPadH + Hi - I1, ConvStrideH) + I1);
            const auto IWTildeSliceEnd = math::min(
                WTilde, math::integer_divide_ceil(InLeftPadW + Wi - I1, ConvStrideW) + I1);

            const auto DTildeSlice = IDTildeSliceEnd - IDTildeSliceBegin;
            const auto HTildeSlice = IHTildeSliceEnd - IHTildeSliceBegin;
            const auto WTildeSlice = IWTildeSliceEnd - IWTildeSliceBegin;

            // GemmK is different for each GEMM
            const auto ZDotSlice = math::integer_divide_ceil(Z - i_ztilde, ZTilde);
            const auto YDotSlice = math::integer_divide_ceil(Y - i_ytilde, YTilde);
            const auto XDotSlice = math::integer_divide_ceil(X - i_xtilde, XTilde);

            if constexpr(NDimSpatial == 2)
            {
                // A: output tensor
                const auto out_n_hop_wop_k_grid_desc = transform_tensor_descriptor(
                    out_grid_desc,
                    make_tuple(make_pass_through_transform(N),
                               make_pad_transform(Ho, I0, I0),
                               make_pad_transform(Wo, I0, I0),
                               make_pass_through_transform(K)),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

                const auto out_n_ydot_htilde_xdot_wtilde_k_grid_desc = transform_tensor_descriptor(
                    out_n_hop_wop_k_grid_desc,
                    make_tuple(
                        make_pass_through_transform(N),
                        make_embed_transform(make_tuple(YDot, HTilde),
                                             make_tuple(-ConvDilationH / GcdStrideDilationH, I1)),
                        make_embed_transform(make_tuple(XDot, WTilde),
                                             make_tuple(-ConvDilationW / GcdStrideDilationW, I1)),
                        make_pass_through_transform(K)),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                    make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3, 4>{}, Sequence<5>{}));

                const auto out_n_ydotslice_htildeslice_xdotslice_wtildeslice_k_grid_desc =
                    transform_tensor_descriptor(
                        out_n_ydot_htilde_xdot_wtilde_k_grid_desc,
                        make_tuple(make_pass_through_transform(N),
                                   make_slice_transform(YDot, I0, YDotSlice),
                                   make_slice_transform(HTilde, IHTildeSliceBegin, HTildeSlice),
                                   make_slice_transform(XDot, I0, XDotSlice),
                                   make_slice_transform(WTilde, IWTildeSliceBegin, WTildeSlice),
                                   make_pass_through_transform(K)),
                        make_tuple(Sequence<0>{},
                                   Sequence<1>{},
                                   Sequence<2>{},
                                   Sequence<3>{},
                                   Sequence<4>{},
                                   Sequence<5>{}),
                        make_tuple(Sequence<0>{},
                                   Sequence<1>{},
                                   Sequence<2>{},
                                   Sequence<3>{},
                                   Sequence<4>{},
                                   Sequence<5>{}));

                const auto out_gemmk_gemmmraw_grid_desc = transform_tensor_descriptor(
                    out_n_ydotslice_htildeslice_xdotslice_wtildeslice_k_grid_desc,
                    make_tuple(make_merge_transform(make_tuple(YDotSlice, XDotSlice, K)),
                               make_merge_transform(make_tuple(N, HTildeSlice, WTildeSlice))),
                    make_tuple(Sequence<1, 3, 5>{}, Sequence<0, 2, 4>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}));

                const auto out_gemmk_gemmm_padded_grid_desc =
                    ck::tensor_operation::device::PadTensorDescriptor(
                        out_gemmk_gemmmraw_grid_desc,
                        make_tuple(GemmKPerBlock, GemmMPerBlock),
                        Sequence<true, DoPadGemmM>{});

                const index_t AK0 = out_gemmk_gemmm_padded_grid_desc.GetLength(I0) / AK1;

                const auto out_gemmak0_gemmm_gemmak1_grid_desc = transform_tensor_descriptor(
                    out_gemmk_gemmm_padded_grid_desc,
                    make_tuple(make_unmerge_transform(make_tuple(AK0, AK1)),
                               make_pass_through_transform(
                                   out_gemmk_gemmm_padded_grid_desc.GetLength(I1))),
                    make_tuple(Sequence<0>{}, Sequence<1>{}),
                    make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

                return out_gemmak0_gemmm_gemmak1_grid_desc;
            }
            else if constexpr(NDimSpatial == 3)
            {
                // A: output tensor
                const auto out_n_hop_wop_k_grid_desc = transform_tensor_descriptor(
                    out_grid_desc,
                    make_tuple(make_pass_through_transform(N),
                               make_pad_transform(Do, I0, I0),
                               make_pad_transform(Ho, I0, I0),
                               make_pad_transform(Wo, I0, I0),
                               make_pass_through_transform(K)),
                    make_tuple(
                        Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
                    make_tuple(
                        Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}));

                const auto out_n_zdot_dtilde_ydot_htilde_xdot_wtilde_k_grid_desc =
                    transform_tensor_descriptor(
                        out_n_hop_wop_k_grid_desc,
                        make_tuple(make_pass_through_transform(N),
                                   make_embed_transform(
                                       make_tuple(ZDot, DTilde),
                                       make_tuple(-ConvDilationD / GcdStrideDilationD, I1)),
                                   make_embed_transform(
                                       make_tuple(YDot, HTilde),
                                       make_tuple(-ConvDilationH / GcdStrideDilationH, I1)),
                                   make_embed_transform(
                                       make_tuple(XDot, WTilde),
                                       make_tuple(-ConvDilationW / GcdStrideDilationW, I1)),
                                   make_pass_through_transform(K)),
                        make_tuple(Sequence<0>{},
                                   Sequence<1>{},
                                   Sequence<2>{},
                                   Sequence<3>{},
                                   Sequence<4>{}),
                        make_tuple(Sequence<0>{},
                                   Sequence<1, 2>{},
                                   Sequence<3, 4>{},
                                   Sequence<5, 6>{},
                                   Sequence<7>{}));

                const auto
                    out_n_zdotslice_dtildeslice_ydotslice_htildeslice_xdotslice_wtildeslice_k_grid_desc =
                        transform_tensor_descriptor(
                            out_n_zdot_dtilde_ydot_htilde_xdot_wtilde_k_grid_desc,
                            make_tuple(make_pass_through_transform(N),
                                       make_slice_transform(ZDot, I0, ZDotSlice),
                                       make_slice_transform(DTilde, IDTildeSliceBegin, DTildeSlice),
                                       make_slice_transform(YDot, I0, YDotSlice),
                                       make_slice_transform(HTilde, IHTildeSliceBegin, HTildeSlice),
                                       make_slice_transform(XDot, I0, XDotSlice),
                                       make_slice_transform(WTilde, IWTildeSliceBegin, WTildeSlice),
                                       make_pass_through_transform(K)),
                            make_tuple(Sequence<0>{},
                                       Sequence<1>{},
                                       Sequence<2>{},
                                       Sequence<3>{},
                                       Sequence<4>{},
                                       Sequence<5>{},
                                       Sequence<6>{},
                                       Sequence<7>{}),
                            make_tuple(Sequence<0>{},
                                       Sequence<1>{},
                                       Sequence<2>{},
                                       Sequence<3>{},
                                       Sequence<4>{},
                                       Sequence<5>{},
                                       Sequence<6>{},
                                       Sequence<7>{}));

                const auto out_gemmk_gemmmraw_grid_desc = transform_tensor_descriptor(
                    out_n_zdotslice_dtildeslice_ydotslice_htildeslice_xdotslice_wtildeslice_k_grid_desc,
                    make_tuple(
                        make_merge_transform(make_tuple(ZDotSlice, YDotSlice, XDotSlice, K)),
                        make_merge_transform(make_tuple(N, DTildeSlice, HTildeSlice, WTildeSlice))),
                    make_tuple(Sequence<1, 3, 5, 7>{}, Sequence<0, 2, 4, 6>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}));

                const auto out_gemmk_gemmm_padded_grid_desc =
                    ck::tensor_operation::device::PadTensorDescriptor(
                        out_gemmk_gemmmraw_grid_desc,
                        make_tuple(GemmKPerBlock, GemmMPerBlock),
                        Sequence<true, DoPadGemmM>{});

                const index_t AK0 = out_gemmk_gemmm_padded_grid_desc.GetLength(I0) / AK1;

                const auto out_gemmak0_gemmm_gemmak1_grid_desc = transform_tensor_descriptor(
                    out_gemmk_gemmm_padded_grid_desc,
                    make_tuple(make_unmerge_transform(make_tuple(AK0, AK1)),
                               make_pass_through_transform(
                                   out_gemmk_gemmm_padded_grid_desc.GetLength(I1))),
                    make_tuple(Sequence<0>{}, Sequence<1>{}),
                    make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

                return out_gemmak0_gemmm_gemmak1_grid_desc;
            }
            else
            {
                throw std::runtime_error("wrong! only implemented for 2D and 3D now");
            }
        }
    }

    template <typename BLayout,
              typename std::enable_if<(NDimSpatial == 2 || NDimSpatial == 3) &&
                                          (is_same_v<BLayout, tensor_layout::convolution::GKYXC> ||
                                           is_same_v<BLayout, tensor_layout::convolution::GKZYXC>),
                                      bool>::type = false>
    static auto MakeBDescriptor_BK0_N_BK1(
        const std::array<index_t, NDimSpatial + 3>& out_g_n_k_wos_lengths,
        const std::array<index_t, NDimSpatial + 3>& /* out_g_n_k_wos_strides */,
        const std::array<index_t, NDimSpatial + 3>& wei_g_k_c_xs_lengths,
        const std::array<index_t, NDimSpatial + 3>& /* wei_g_k_c_xs_strides */,
        const std::array<index_t, NDimSpatial + 3>& in_g_n_c_wis_lengths,
        const std::array<index_t, NDimSpatial + 3>& /* in_g_n_c_wis_strides */,
        const std::array<index_t, NDimSpatial>& conv_filter_strides,
        const std::array<index_t, NDimSpatial>& conv_filter_dilations,
        const std::array<index_t, NDimSpatial>& /* input_left_pads */,
        const std::array<index_t, NDimSpatial>& /* input_right_pads */,
        const std::array<index_t, NDimSpatial>& tildes)
    {
        index_t i_ztilde = tildes[ZIdx - NonSpatialDimsNum];
        index_t i_ytilde = tildes[YIdx - NonSpatialDimsNum];
        index_t i_xtilde = tildes[XIdx - NonSpatialDimsNum];

        const index_t N = in_g_n_c_wis_lengths[1];
        const index_t K = wei_g_k_c_xs_lengths[1];
        const index_t C = wei_g_k_c_xs_lengths[2];

        const index_t Do = NDimSpatial == 3 ? out_g_n_k_wos_lengths[DIdx] : 1;
        const index_t Ho = out_g_n_k_wos_lengths[HIdx];
        const index_t Wo = out_g_n_k_wos_lengths[WIdx];

        const index_t Z = NDimSpatial == 3 ? wei_g_k_c_xs_lengths[ZIdx] : 1;
        const index_t Y = wei_g_k_c_xs_lengths[YIdx];
        const index_t X = wei_g_k_c_xs_lengths[XIdx];

        const index_t ConvStrideD = conv_filter_strides[DIdx - NonSpatialDimsNum];
        const index_t ConvStrideH = conv_filter_strides[HIdx - NonSpatialDimsNum];
        const index_t ConvStrideW = conv_filter_strides[WIdx - NonSpatialDimsNum];

        const index_t ConvDilationD = conv_filter_dilations[DIdx - NonSpatialDimsNum];
        const index_t ConvDilationH = conv_filter_dilations[HIdx - NonSpatialDimsNum];
        const index_t ConvDilationW = conv_filter_dilations[WIdx - NonSpatialDimsNum];

        // assume packed
        // k_y_x_c for 2d or k_z_y_x_c for 3d
        const auto wei_grid_desc = make_wei_grid_desc<BLayout>(K, Z, Y, X, C);

        if constexpr(ConvBwdDataSpecialization ==
                     ck::tensor_operation::device::ConvolutionBackwardDataSpecialization::
                         Filter1x1Stride1Pad0)
        {
            const index_t BK0 = math::integer_divide_ceil(K, BK1);

            // B: weight tensor
            const auto wei_gemmbk0_gemmnraw_gemmbk1_grid_desc =
                transform_tensor_descriptor(make_naive_tensor_descriptor_packed(make_tuple(K, C)),
                                            make_tuple(make_unmerge_transform(make_tuple(BK0, BK1)),
                                                       make_pass_through_transform(C)),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}),
                                            make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
            make_naive_tensor_descriptor(make_tuple(N * Do * Ho * Wo, C), make_tuple(I0, I1));

            const auto wei_gemmbk0_gemmn_gemmbk1_grid_desc =
                ck::tensor_operation::device::PadTensorDescriptor(
                    wei_gemmbk0_gemmnraw_gemmbk1_grid_desc,
                    make_tuple(BK0, GemmNPerBlock, BK1),
                    Sequence<false, DoPadGemmN, false>{});

            return wei_gemmbk0_gemmn_gemmbk1_grid_desc;
        }
        else
        {
            const auto GcdStrideDilationD = math::gcd(ConvStrideD, ConvDilationD);
            const auto GcdStrideDilationH = math::gcd(ConvStrideH, ConvDilationH);
            const auto GcdStrideDilationW = math::gcd(ConvStrideW, ConvDilationW);

            const auto ZTilde = ConvStrideD / GcdStrideDilationD;
            const auto YTilde = ConvStrideH / GcdStrideDilationH;
            const auto XTilde = ConvStrideW / GcdStrideDilationW;

            const auto ZDot = math::integer_divide_ceil(Z, ZTilde);
            const auto YDot = math::integer_divide_ceil(Y, YTilde);
            const auto XDot = math::integer_divide_ceil(X, XTilde);

            // GemmK is different for each GEMM
            const auto ZDotSlice = math::integer_divide_ceil(Z - i_ztilde, ZTilde);
            const auto YDotSlice = math::integer_divide_ceil(Y - i_ytilde, YTilde);
            const auto XDotSlice = math::integer_divide_ceil(X - i_xtilde, XTilde);

            // B weight tensor
            if constexpr(NDimSpatial == 2)
            {
                const auto wei_k_ydot_ytilde_xdot_xtilde_c_grid_desc = transform_tensor_descriptor(
                    wei_grid_desc,
                    make_tuple(
                        make_pass_through_transform(K),
                        make_embed_transform(make_tuple(YDot, YTilde),
                                             make_tuple(ConvStrideH / GcdStrideDilationH, I1)),
                        make_embed_transform(make_tuple(XDot, XTilde),
                                             make_tuple(ConvStrideW / GcdStrideDilationW, I1)),
                        make_pass_through_transform(C)),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                    make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3, 4>{}, Sequence<5>{}));

                const auto wei_k_ydotslice_xdotslice_c_grid_desc = transform_tensor_descriptor(
                    wei_k_ydot_ytilde_xdot_xtilde_c_grid_desc,
                    make_tuple(make_pass_through_transform(K),
                               make_slice_transform(YDot, I0, YDotSlice),
                               make_slice_transform(XDot, I0, XDotSlice),
                               make_freeze_transform(i_ytilde),
                               make_freeze_transform(i_xtilde),
                               make_pass_through_transform(C)),
                    make_tuple(Sequence<0>{},
                               Sequence<1>{},
                               Sequence<3>{},
                               Sequence<2>{},
                               Sequence<4>{},
                               Sequence<5>{}),
                    make_tuple(Sequence<0>{},
                               Sequence<1>{},
                               Sequence<2>{},
                               Sequence<>{},
                               Sequence<>{},
                               Sequence<3>{}));

                const auto wei_gemmk_gemmnraw_grid_desc = transform_tensor_descriptor(
                    wei_k_ydotslice_xdotslice_c_grid_desc,
                    make_tuple(make_merge_transform(make_tuple(YDotSlice, XDotSlice, K)),
                               make_pass_through_transform(C)),
                    make_tuple(Sequence<1, 2, 0>{}, Sequence<3>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}));

                const auto wei_gemmk_gemmn_padded_grid_desc =
                    ck::tensor_operation::device::PadTensorDescriptor(
                        wei_gemmk_gemmnraw_grid_desc,
                        make_tuple(GemmKPerBlock, GemmNPerBlock),
                        Sequence<true, DoPadGemmN>{});

                const index_t BK0 = wei_gemmk_gemmn_padded_grid_desc.GetLength(I0) / BK1;

                const auto wei_gemmbk0_gemmn_gemmbk1_grid_desc = transform_tensor_descriptor(
                    wei_gemmk_gemmn_padded_grid_desc,
                    make_tuple(make_unmerge_transform(make_tuple(BK0, BK1)),
                               make_pass_through_transform(
                                   wei_gemmk_gemmn_padded_grid_desc.GetLength(I1))),
                    make_tuple(Sequence<0>{}, Sequence<1>{}),
                    make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

                return wei_gemmbk0_gemmn_gemmbk1_grid_desc;
            }
            else if constexpr(NDimSpatial == 3)
            {
                const auto wei_k_zdot_ztilde_ydot_ytilde_xdot_xtilde_c_grid_desc =
                    transform_tensor_descriptor(
                        wei_grid_desc,
                        make_tuple(
                            make_pass_through_transform(K),
                            make_embed_transform(make_tuple(ZDot, ZTilde),
                                                 make_tuple(ConvStrideD / GcdStrideDilationD, I1)),
                            make_embed_transform(make_tuple(YDot, YTilde),
                                                 make_tuple(ConvStrideH / GcdStrideDilationH, I1)),
                            make_embed_transform(make_tuple(XDot, XTilde),
                                                 make_tuple(ConvStrideW / GcdStrideDilationW, I1)),
                            make_pass_through_transform(C)),
                        make_tuple(Sequence<0>{},
                                   Sequence<1>{},
                                   Sequence<2>{},
                                   Sequence<3>{},
                                   Sequence<4>{}),
                        make_tuple(Sequence<0>{},
                                   Sequence<1, 2>{},
                                   Sequence<3, 4>{},
                                   Sequence<5, 6>{},
                                   Sequence<7>{}));

                const auto wei_gemmk_zdotslice_ydotslice_xdotslice_c_grid_desc =
                    transform_tensor_descriptor(
                        wei_k_zdot_ztilde_ydot_ytilde_xdot_xtilde_c_grid_desc,
                        make_tuple(make_pass_through_transform(K),
                                   make_slice_transform(ZDot, I0, ZDotSlice),
                                   make_slice_transform(YDot, I0, YDotSlice),
                                   make_slice_transform(XDot, I0, XDotSlice),
                                   make_freeze_transform(i_ztilde),
                                   make_freeze_transform(i_ytilde),
                                   make_freeze_transform(i_xtilde),
                                   make_pass_through_transform(C)),
                        make_tuple(Sequence<0>{},
                                   Sequence<1>{},
                                   Sequence<3>{},
                                   Sequence<5>{},
                                   Sequence<2>{},
                                   Sequence<4>{},
                                   Sequence<6>{},
                                   Sequence<7>{}),
                        make_tuple(Sequence<0>{},
                                   Sequence<1>{},
                                   Sequence<2>{},
                                   Sequence<3>{},
                                   Sequence<>{},
                                   Sequence<>{},
                                   Sequence<>{},
                                   Sequence<4>{}));

                const auto wei_gemmk_gemmnraw_grid_desc = transform_tensor_descriptor(
                    wei_gemmk_zdotslice_ydotslice_xdotslice_c_grid_desc,
                    make_tuple(make_merge_transform(make_tuple(ZDotSlice, YDotSlice, XDotSlice, K)),
                               make_pass_through_transform(C)),
                    make_tuple(Sequence<1, 2, 3, 0>{}, Sequence<4>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}));

                const auto wei_gemmk_gemmn_padded_grid_desc =
                    ck::tensor_operation::device::PadTensorDescriptor(
                        wei_gemmk_gemmnraw_grid_desc,
                        make_tuple(GemmKPerBlock, GemmNPerBlock),
                        Sequence<true, DoPadGemmN>{});

                const index_t BK0 = wei_gemmk_gemmn_padded_grid_desc.GetLength(I0) / BK1;

                const auto wei_gemmbk0_gemm_gemmbk1_grid_desc = transform_tensor_descriptor(
                    wei_gemmk_gemmn_padded_grid_desc,
                    make_tuple(make_unmerge_transform(make_tuple(BK0, BK1)),
                               make_pass_through_transform(
                                   wei_gemmk_gemmn_padded_grid_desc.GetLength(I1))),
                    make_tuple(Sequence<0>{}, Sequence<1>{}),
                    make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

                return wei_gemmbk0_gemm_gemmbk1_grid_desc;
            }
            else
            {
                throw std::runtime_error("wrong! only implemented for 2D and 3D now");
            }
        }
    }

    template <typename CLayout,
              typename std::enable_if<(NDimSpatial == 2 || NDimSpatial == 3) &&
                                          (is_same_v<CLayout, tensor_layout::convolution::GNHWC> ||
                                           is_same_v<CLayout, tensor_layout::convolution::GNDHWC> ||
                                           is_same_v<CLayout, tensor_layout::convolution::NHWGC> ||
                                           is_same_v<CLayout, tensor_layout::convolution::NDHWGC> ||
                                           is_same_v<CLayout, tensor_layout::convolution::G_NHW_C>),
                                      bool>::type = false>
    static auto
    MakeCDescriptor_M_N(const std::array<index_t, NDimSpatial + 3>& out_g_n_k_wos_lengths,
                        const std::array<index_t, NDimSpatial + 3>& /* out_g_n_k_wos_strides */,
                        const std::array<index_t, NDimSpatial + 3>& wei_g_k_c_xs_lengths,
                        const std::array<index_t, NDimSpatial + 3>& /* wei_g_k_c_xs_strides */,
                        const std::array<index_t, NDimSpatial + 3>& in_g_n_c_wis_lengths,
                        const std::array<index_t, NDimSpatial + 3>& in_g_n_c_wis_strides,
                        const std::array<index_t, NDimSpatial>& conv_filter_strides,
                        const std::array<index_t, NDimSpatial>& conv_filter_dilations,
                        const std::array<index_t, NDimSpatial>& input_left_pads,
                        const std::array<index_t, NDimSpatial>& input_right_pads,
                        const std::array<index_t, NDimSpatial>& tildes)
    {
        index_t i_ztilde = tildes[ZIdx - NonSpatialDimsNum];
        index_t i_ytilde = tildes[YIdx - NonSpatialDimsNum];
        index_t i_xtilde = tildes[XIdx - NonSpatialDimsNum];

        const index_t N = in_g_n_c_wis_lengths[1];
        const index_t C = wei_g_k_c_xs_lengths[2];

        const index_t Di = NDimSpatial == 3 ? in_g_n_c_wis_lengths[DIdx] : 1;
        const index_t Hi = in_g_n_c_wis_lengths[HIdx];
        const index_t Wi = in_g_n_c_wis_lengths[WIdx];

        const index_t Do = NDimSpatial == 3 ? out_g_n_k_wos_lengths[DIdx] : 1;
        const index_t Ho = out_g_n_k_wos_lengths[HIdx];
        const index_t Wo = out_g_n_k_wos_lengths[WIdx];

        const index_t Z = NDimSpatial == 3 ? wei_g_k_c_xs_lengths[ZIdx] : 1;
        const index_t Y = wei_g_k_c_xs_lengths[YIdx];
        const index_t X = wei_g_k_c_xs_lengths[XIdx];

        const index_t InLeftPadD = input_left_pads[DIdx - NonSpatialDimsNum];
        const index_t InLeftPadH = input_left_pads[HIdx - NonSpatialDimsNum];
        const index_t InLeftPadW = input_left_pads[WIdx - NonSpatialDimsNum];

        const index_t InRightPadD = input_right_pads[DIdx - NonSpatialDimsNum];
        const index_t InRightPadH = input_right_pads[HIdx - NonSpatialDimsNum];
        const index_t InRightPadW = input_right_pads[WIdx - NonSpatialDimsNum];

        const index_t ConvStrideD = conv_filter_strides[DIdx - NonSpatialDimsNum];
        const index_t ConvStrideH = conv_filter_strides[HIdx - NonSpatialDimsNum];
        const index_t ConvStrideW = conv_filter_strides[WIdx - NonSpatialDimsNum];

        const index_t ConvDilationD = conv_filter_dilations[DIdx - NonSpatialDimsNum];
        const index_t ConvDilationH = conv_filter_dilations[HIdx - NonSpatialDimsNum];
        const index_t ConvDilationW = conv_filter_dilations[WIdx - NonSpatialDimsNum];

        // assume strided
        // n_hi_wi_c for 2d n_di_hi_wi_c for 3d
        const auto in_grid_desc =
            make_in_grid_desc<NDimSpatial, CLayout>(N, Di, Hi, Wi, C, in_g_n_c_wis_strides);

        if constexpr(ConvBwdDataSpecialization ==
                     ck::tensor_operation::device::ConvolutionBackwardDataSpecialization::
                         Filter1x1Stride1Pad0)
        {
            // C: input tensor
            if constexpr(NDimSpatial == 2)
            {
                const auto in_n_y_ho_x_wo_c_grid_desc = transform_tensor_descriptor(
                    in_grid_desc,
                    make_tuple(
                        make_pass_through_transform(N),
                        make_embed_transform(make_tuple(I1, Ho), make_tuple(I1, ConvStrideH)),
                        make_embed_transform(make_tuple(I1, Wo), make_tuple(I1, ConvStrideW)),
                        make_pass_through_transform(C)),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                    make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3, 4>{}, Sequence<5>{}));

                const auto in_gemmmraw_gemmnraw_grid_desc = transform_tensor_descriptor(
                    in_n_y_ho_x_wo_c_grid_desc,
                    make_tuple(make_freeze_transform(I0),
                               make_freeze_transform(I0),
                               make_merge_transform(make_tuple(N, Ho, Wo)),
                               make_pass_through_transform(C)),
                    make_tuple(Sequence<1>{}, Sequence<3>{}, Sequence<0, 2, 4>{}, Sequence<5>{}),
                    make_tuple(Sequence<>{}, Sequence<>{}, Sequence<0>{}, Sequence<1>{}));

                const auto in_gemmm_gemmn_grid_desc =
                    ck::tensor_operation::device::PadTensorDescriptor(
                        in_gemmmraw_gemmnraw_grid_desc,
                        make_tuple(GemmMPerBlock, GemmNPerBlock),
                        Sequence<DoPadGemmM, DoPadGemmN>{});

                return in_gemmm_gemmn_grid_desc;
            }
            else if constexpr(NDimSpatial == 3)
            {

                // C: input tensor
                const auto in_n_x_do_y_ho_x_wo_c_grid_desc = transform_tensor_descriptor(
                    in_grid_desc,
                    make_tuple(
                        make_pass_through_transform(N),
                        make_embed_transform(make_tuple(I1, Do), make_tuple(I1, ConvStrideD)),
                        make_embed_transform(make_tuple(I1, Ho), make_tuple(I1, ConvStrideH)),
                        make_embed_transform(make_tuple(I1, Wo), make_tuple(I1, ConvStrideW)),
                        make_pass_through_transform(C)),
                    make_tuple(
                        Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
                    make_tuple(Sequence<0>{},
                               Sequence<1, 2>{},
                               Sequence<3, 4>{},
                               Sequence<5, 6>{},
                               Sequence<7>{}));

                const auto in_gemmmraw_gemmnraw_grid_desc = transform_tensor_descriptor(
                    in_n_x_do_y_ho_x_wo_c_grid_desc,
                    make_tuple(make_freeze_transform(I0),
                               make_freeze_transform(I0),
                               make_freeze_transform(I0),
                               make_merge_transform(make_tuple(N, Do, Ho, Wo)),
                               make_pass_through_transform(C)),
                    make_tuple(Sequence<1>{},
                               Sequence<3>{},
                               Sequence<5>{},
                               Sequence<0, 2, 4, 6>{},
                               Sequence<7>{}),
                    make_tuple(
                        Sequence<>{}, Sequence<>{}, Sequence<>{}, Sequence<0>{}, Sequence<1>{}));

                const auto in_gemmm_gemmn_grid_desc =
                    ck::tensor_operation::device::PadTensorDescriptor(
                        in_gemmmraw_gemmnraw_grid_desc,
                        make_tuple(GemmMPerBlock, GemmNPerBlock),
                        Sequence<DoPadGemmM, DoPadGemmN>{});

                return in_gemmm_gemmn_grid_desc;
            }
            else
            {
                throw std::runtime_error("wrong! only implemented for 2D and 3D now");
            }
        }
        else
        {
            const auto GcdStrideDilationD = math::gcd(ConvStrideD, ConvDilationD);
            const auto GcdStrideDilationH = math::gcd(ConvStrideH, ConvDilationH);
            const auto GcdStrideDilationW = math::gcd(ConvStrideW, ConvDilationW);

            const auto ZTilde = ConvStrideD / GcdStrideDilationD;
            const auto YTilde = ConvStrideH / GcdStrideDilationH;
            const auto XTilde = ConvStrideW / GcdStrideDilationW;

            const auto DTilde =
                Do + math::integer_divide_ceil(ConvDilationD * (Z - I1), ConvStrideD);
            const auto HTilde =
                Ho + math::integer_divide_ceil(ConvDilationH * (Y - I1), ConvStrideH);
            const auto WTilde =
                Wo + math::integer_divide_ceil(ConvDilationW * (X - I1), ConvStrideW);

            // only work on DTilde, HTilde and WTilde that contribute to
            // non-padding area of input tensor
            const auto IDTildeSliceBegin = math::integer_divide_floor(
                math::max(I0, InLeftPadD - ConvDilationD * (ZTilde - I1)), ConvStrideD);
            const auto IHTildeSliceBegin = math::integer_divide_floor(
                math::max(I0, InLeftPadH - ConvDilationH * (YTilde - I1)), ConvStrideH);
            const auto IWTildeSliceBegin = math::integer_divide_floor(
                math::max(I0, InLeftPadW - ConvDilationW * (XTilde - I1)), ConvStrideW);

            const auto IDTildeSliceEnd = math::min(
                DTilde, math::integer_divide_ceil(InLeftPadD + Di - I1, ConvStrideD) + I1);
            const auto IHTildeSliceEnd = math::min(
                HTilde, math::integer_divide_ceil(InLeftPadH + Hi - I1, ConvStrideH) + I1);
            const auto IWTildeSliceEnd = math::min(
                WTilde, math::integer_divide_ceil(InLeftPadW + Wi - I1, ConvStrideW) + I1);

            const auto DTildeSlice = IDTildeSliceEnd - IDTildeSliceBegin;
            const auto HTildeSlice = IHTildeSliceEnd - IHTildeSliceBegin;
            const auto WTildeSlice = IWTildeSliceEnd - IWTildeSliceBegin;

            // C: input tensor
            if constexpr(NDimSpatial == 2)
            {
                const auto in_n_hip_wip_c_grid_desc = transform_tensor_descriptor(
                    in_grid_desc,
                    make_tuple(make_pass_through_transform(N),
                               make_pad_transform(Hi, InLeftPadH, InRightPadH),
                               make_pad_transform(Wi, InLeftPadW, InRightPadW),
                               make_pass_through_transform(C)),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

                const auto in_n_ytilde_htilde_xtilde_wtilde_c_grid_desc =
                    transform_tensor_descriptor(
                        in_n_hip_wip_c_grid_desc,
                        make_tuple(make_pass_through_transform(N),
                                   make_embed_transform(make_tuple(YTilde, HTilde),
                                                        make_tuple(ConvDilationH, ConvStrideH)),
                                   make_embed_transform(make_tuple(XTilde, WTilde),
                                                        make_tuple(ConvDilationW, ConvStrideW)),
                                   make_pass_through_transform(C)),
                        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                        make_tuple(
                            Sequence<0>{}, Sequence<1, 2>{}, Sequence<3, 4>{}, Sequence<5>{}));

                const auto in_n_htildeslice_wtildeslice_c_grid_desc = transform_tensor_descriptor(
                    in_n_ytilde_htilde_xtilde_wtilde_c_grid_desc,
                    make_tuple(make_pass_through_transform(N),
                               make_freeze_transform(i_ytilde),
                               make_slice_transform(HTilde, IHTildeSliceBegin, HTildeSlice),
                               make_freeze_transform(i_xtilde),
                               make_slice_transform(WTilde, IWTildeSliceBegin, WTildeSlice),
                               make_pass_through_transform(C)),
                    make_tuple(Sequence<0>{},
                               Sequence<1>{},
                               Sequence<2>{},
                               Sequence<3>{},
                               Sequence<4>{},
                               Sequence<5>{}),
                    make_tuple(Sequence<0>{},
                               Sequence<>{},
                               Sequence<1>{},
                               Sequence<>{},
                               Sequence<2>{},
                               Sequence<3>{}));

                const auto in_gemmmraw_gemmnraw_grid_desc = transform_tensor_descriptor(
                    in_n_htildeslice_wtildeslice_c_grid_desc,
                    make_tuple(make_merge_transform(make_tuple(N, HTildeSlice, WTildeSlice)),
                               make_pass_through_transform(C)),
                    make_tuple(Sequence<0, 1, 2>{}, Sequence<3>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}));

                const auto in_gemmm_gemmn_grid_desc =
                    ck::tensor_operation::device::PadTensorDescriptor(
                        in_gemmmraw_gemmnraw_grid_desc,
                        make_tuple(GemmMPerBlock, GemmNPerBlock),
                        Sequence<DoPadGemmM, DoPadGemmN>{});

                return in_gemmm_gemmn_grid_desc;
            }
            else if(NDimSpatial == 3)
            {
                const auto in_n_dip_hip_wip_c_grid_desc = transform_tensor_descriptor(
                    in_grid_desc,
                    make_tuple(make_pass_through_transform(N),
                               make_pad_transform(Di, InLeftPadD, InRightPadD),
                               make_pad_transform(Hi, InLeftPadH, InRightPadH),
                               make_pad_transform(Wi, InLeftPadW, InRightPadW),
                               make_pass_through_transform(C)),
                    make_tuple(
                        Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
                    make_tuple(
                        Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}));

                const auto in_n_ztilde_dtilde_ytilde_htilde_xtilde_wtilde_c_grid_desc =
                    transform_tensor_descriptor(
                        in_n_dip_hip_wip_c_grid_desc,
                        make_tuple(make_pass_through_transform(N),
                                   make_embed_transform(make_tuple(ZTilde, DTilde),
                                                        make_tuple(ConvDilationD, ConvStrideD)),
                                   make_embed_transform(make_tuple(YTilde, HTilde),
                                                        make_tuple(ConvDilationH, ConvStrideH)),
                                   make_embed_transform(make_tuple(XTilde, WTilde),
                                                        make_tuple(ConvDilationW, ConvStrideW)),
                                   make_pass_through_transform(C)),
                        make_tuple(Sequence<0>{},
                                   Sequence<1>{},
                                   Sequence<2>{},
                                   Sequence<3>{},
                                   Sequence<4>{}),
                        make_tuple(Sequence<0>{},
                                   Sequence<1, 2>{},
                                   Sequence<3, 4>{},
                                   Sequence<5, 6>{},
                                   Sequence<7>{}));

                const auto in_n_dtildeslice_htildeslice_wtildeslice_c_grid_desc =
                    transform_tensor_descriptor(
                        in_n_ztilde_dtilde_ytilde_htilde_xtilde_wtilde_c_grid_desc,
                        make_tuple(make_pass_through_transform(N),
                                   make_freeze_transform(i_ztilde),
                                   make_slice_transform(DTilde, IDTildeSliceBegin, DTildeSlice),
                                   make_freeze_transform(i_ytilde),
                                   make_slice_transform(HTilde, IHTildeSliceBegin, HTildeSlice),
                                   make_freeze_transform(i_xtilde),
                                   make_slice_transform(WTilde, IWTildeSliceBegin, WTildeSlice),
                                   make_pass_through_transform(C)),
                        make_tuple(Sequence<0>{},
                                   Sequence<1>{},
                                   Sequence<2>{},
                                   Sequence<3>{},
                                   Sequence<4>{},
                                   Sequence<5>{},
                                   Sequence<6>{},
                                   Sequence<7>{}),
                        make_tuple(Sequence<0>{},
                                   Sequence<>{},
                                   Sequence<1>{},
                                   Sequence<>{},
                                   Sequence<2>{},
                                   Sequence<>{},
                                   Sequence<3>{},
                                   Sequence<4>{}));

                const auto in_gemmmraw_gemmnraw_grid_desc = transform_tensor_descriptor(
                    in_n_dtildeslice_htildeslice_wtildeslice_c_grid_desc,
                    make_tuple(
                        make_merge_transform(make_tuple(N, DTildeSlice, HTildeSlice, WTildeSlice)),
                        make_pass_through_transform(C)),
                    make_tuple(Sequence<0, 1, 2, 3>{}, Sequence<4>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}));

                const auto in_gemmm_gemmn_grid_desc =
                    ck::tensor_operation::device::PadTensorDescriptor(
                        in_gemmmraw_gemmnraw_grid_desc,
                        make_tuple(GemmMPerBlock, GemmNPerBlock),
                        Sequence<DoPadGemmM, DoPadGemmN>{});
                return in_gemmm_gemmn_grid_desc;
            }
            else
            {
                throw std::runtime_error("wrong! only implemented for 2D and 3D now");
            }
        }
    }

    // for input bias
    template <typename CLayout,
              typename std::enable_if<NDimSpatial == 2 &&
                                          (is_same_v<CLayout, tensor_layout::convolution::GC> ||
                                           is_same_v<CLayout, tensor_layout::convolution::G_C>),
                                      bool>::type = false>
    static auto
    MakeCDescriptor_M_N(const std::array<index_t, NDimSpatial + 3>& out_g_n_k_wos_lengths,
                        const std::array<index_t, NDimSpatial + 3>& /* out_g_n_k_wos_strides */,
                        const std::array<index_t, NDimSpatial + 3>& wei_g_k_c_xs_lengths,
                        const std::array<index_t, NDimSpatial + 3>& /* wei_g_k_c_xs_strides */,
                        const std::array<index_t, NDimSpatial + 3>& in_g_n_c_wis_lengths,
                        const std::array<index_t, NDimSpatial + 3>& /* in_g_n_c_wis_strides */,
                        const std::array<index_t, NDimSpatial>& conv_filter_strides,
                        const std::array<index_t, NDimSpatial>& conv_filter_dilations,
                        const std::array<index_t, NDimSpatial>& input_left_pads,
                        const std::array<index_t, NDimSpatial>& /* input_right_pads */,
                        const std::array<index_t, NDimSpatial>& /* tildes */)
    {
        const index_t N = in_g_n_c_wis_lengths[1];
        const index_t C = wei_g_k_c_xs_lengths[2];

        const index_t Hi = in_g_n_c_wis_lengths[3];
        const index_t Wi = in_g_n_c_wis_lengths[4];

        const index_t Ho = out_g_n_k_wos_lengths[3];
        const index_t Wo = out_g_n_k_wos_lengths[4];

        const index_t Y = wei_g_k_c_xs_lengths[3];
        const index_t X = wei_g_k_c_xs_lengths[4];

        const index_t InLeftPadH = input_left_pads[0];
        const index_t InLeftPadW = input_left_pads[1];

        const index_t ConvStrideH = conv_filter_strides[0];
        const index_t ConvStrideW = conv_filter_strides[1];

        const index_t ConvDilationH = conv_filter_dilations[0];
        const index_t ConvDilationW = conv_filter_dilations[1];

        if constexpr(ConvBwdDataSpecialization ==
                     ck::tensor_operation::device::ConvolutionBackwardDataSpecialization::
                         Filter1x1Stride1Pad0)
        {
            const auto in_gemmm_gemmn_grid_desc =
                make_naive_tensor_descriptor(make_tuple(N * Ho * Wo, C), make_tuple(I0, I1));

            return in_gemmm_gemmn_grid_desc;
        }
        else
        {
            const auto GcdStrideDilationH = math::gcd(ConvStrideH, ConvDilationH);
            const auto GcdStrideDilationW = math::gcd(ConvStrideW, ConvDilationW);

            const auto YTilde = ConvStrideH / GcdStrideDilationH;
            const auto XTilde = ConvStrideW / GcdStrideDilationW;

            const auto HTilde =
                Ho + math::integer_divide_ceil(ConvDilationH * (Y - I1), ConvStrideH);
            const auto WTilde =
                Wo + math::integer_divide_ceil(ConvDilationW * (X - I1), ConvStrideW);

            // only work on HTilde and WTilde that contribute to non-padding area of input tensor
            const auto IHTildeSliceBegin = math::integer_divide_floor(
                math::max(I0, InLeftPadH - ConvDilationH * (YTilde - I1)), ConvStrideH);
            const auto IWTildeSliceBegin = math::integer_divide_floor(
                math::max(I0, InLeftPadW - ConvDilationW * (XTilde - I1)), ConvStrideW);

            const auto IHTildeSliceEnd = math::min(
                HTilde, math::integer_divide_ceil(InLeftPadH + Hi - I1, ConvStrideH) + I1);
            const auto IWTildeSliceEnd = math::min(
                WTilde, math::integer_divide_ceil(InLeftPadW + Wi - I1, ConvStrideW) + I1);

            const auto HTildeSlice = IHTildeSliceEnd - IHTildeSliceBegin;
            const auto WTildeSlice = IWTildeSliceEnd - IWTildeSliceBegin;

            // bias tensor
            const auto in_gemmmraw_gemmnraw_grid_desc = make_naive_tensor_descriptor(
                make_tuple(N * HTildeSlice * WTildeSlice, C), make_tuple(I0, I1));

            const auto in_gemmm_gemmn_grid_desc = ck::tensor_operation::device::PadTensorDescriptor(
                in_gemmmraw_gemmnraw_grid_desc,
                make_tuple(GemmMPerBlock, GemmNPerBlock),
                Sequence<DoPadGemmM, DoPadGemmN>{});

            return in_gemmm_gemmn_grid_desc;
        }
    }
};

} // namespace tensor_operation
} // namespace ck
