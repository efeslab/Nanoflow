// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/matrix_padder.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_utils.hpp"

namespace ck {
namespace tensor_operation {

template <typename ALayout,
          typename BLayout,
          typename ELayout,
          index_t NDimSpatial,
          index_t MPerThread,
          index_t NPerThread>
struct TransformConvNGCHWToNHWGC
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};
    static constexpr auto I5 = Number<5>{};

    template <ck::index_t NDim, typename ck::enable_if<NDim == 1, bool>::type = false>
    static auto MakeNGCHWTransposeDesc(std::array<ck::index_t, NDimSpatial + 3> g_n_c_wis_lengths,
                                       std::array<ck::index_t, NDimSpatial + 3> g_n_c_wis_strides)
    {
        const index_t& G  = g_n_c_wis_lengths[I0];
        const index_t& N  = g_n_c_wis_lengths[I1];
        const index_t& C  = g_n_c_wis_lengths[I2];
        const index_t& Wi = g_n_c_wis_lengths[I3];

        const index_t& GStride  = g_n_c_wis_strides[I0];
        const index_t& NStride  = g_n_c_wis_strides[I1];
        const index_t& CStride  = g_n_c_wis_strides[I2];
        const index_t& WiStride = g_n_c_wis_strides[I3];

        const auto desc = make_naive_tensor_descriptor(
            make_tuple(N, G, C, Wi), make_tuple(NStride, GStride, CStride, WiStride));
        const auto merged_desc =
            transform_tensor_descriptor(desc,
                                        make_tuple(make_merge_transform(make_tuple(N, G, C)),
                                                   make_merge_transform(make_tuple(Wi))),
                                        make_tuple(Sequence<0, 1, 2>{}, Sequence<3>{}),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}));
        return device::PadTensorDescriptor(
            merged_desc, make_tuple(MPerThread, NPerThread), Sequence<true, true>{});
    }

    template <ck::index_t NDim, typename ck::enable_if<NDim == 1, bool>::type = false>
    static auto MakeNHWGCTransposeDesc(std::array<ck::index_t, NDimSpatial + 3> g_n_c_wis_lengths,
                                       std::array<ck::index_t, NDimSpatial + 3> g_n_c_wis_strides)
    {
        const index_t& G  = g_n_c_wis_lengths[I0];
        const index_t& N  = g_n_c_wis_lengths[I1];
        const index_t& C  = g_n_c_wis_lengths[I2];
        const index_t& Wi = g_n_c_wis_lengths[I3];

        const index_t& NStride = g_n_c_wis_strides[I1];
        const index_t WiStride = G * C;
        const index_t GStride  = C;
        const index_t CStride  = 1;

        const auto desc = make_naive_tensor_descriptor(
            make_tuple(N, G, C, Wi), make_tuple(NStride, GStride, CStride, WiStride));
        const auto merged_desc =
            transform_tensor_descriptor(desc,
                                        make_tuple(make_merge_transform(make_tuple(N, G, C)),
                                                   make_merge_transform(make_tuple(Wi))),
                                        make_tuple(Sequence<0, 1, 2>{}, Sequence<3>{}),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}));
        return device::PadTensorDescriptor(
            merged_desc, make_tuple(MPerThread, NPerThread), Sequence<true, true>{});
    }

    template <ck::index_t NDim, typename ck::enable_if<NDim == 2, bool>::type = false>
    static auto MakeNGCHWTransposeDesc(std::array<ck::index_t, NDimSpatial + 3> g_n_c_wis_lengths,
                                       std::array<ck::index_t, NDimSpatial + 3> g_n_c_wis_strides)
    {
        const index_t& G  = g_n_c_wis_lengths[I0];
        const index_t& N  = g_n_c_wis_lengths[I1];
        const index_t& C  = g_n_c_wis_lengths[I2];
        const index_t& Hi = g_n_c_wis_lengths[I3];
        const index_t& Wi = g_n_c_wis_lengths[I4];

        const index_t& GStride  = g_n_c_wis_strides[I0];
        const index_t& NStride  = g_n_c_wis_strides[I1];
        const index_t& CStride  = g_n_c_wis_strides[I2];
        const index_t& HiStride = g_n_c_wis_strides[I3];
        const index_t& WiStride = g_n_c_wis_strides[I4];

        const auto desc = make_naive_tensor_descriptor(
            make_tuple(N, G, C, Hi, Wi), make_tuple(NStride, GStride, CStride, HiStride, WiStride));
        const auto merged_desc =
            transform_tensor_descriptor(desc,
                                        make_tuple(make_merge_transform(make_tuple(N, G, C)),
                                                   make_merge_transform(make_tuple(Hi, Wi))),
                                        make_tuple(Sequence<0, 1, 2>{}, Sequence<3, 4>{}),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}));
        return device::PadTensorDescriptor(
            merged_desc, make_tuple(MPerThread, NPerThread), Sequence<true, true>{});
    }

    template <ck::index_t NDim, typename ck::enable_if<NDim == 2, bool>::type = false>
    static auto MakeNHWGCTransposeDesc(std::array<ck::index_t, NDimSpatial + 3> g_n_c_wis_lengths,
                                       std::array<ck::index_t, NDimSpatial + 3> g_n_c_wis_strides)
    {
        const index_t& G  = g_n_c_wis_lengths[I0];
        const index_t& N  = g_n_c_wis_lengths[I1];
        const index_t& C  = g_n_c_wis_lengths[I2];
        const index_t& Hi = g_n_c_wis_lengths[I3];
        const index_t& Wi = g_n_c_wis_lengths[I4];

        const index_t& NStride = g_n_c_wis_strides[I1];
        const index_t HiStride = Wi * G * C;
        const index_t WiStride = G * C;
        const index_t GStride  = C;
        const index_t CStride  = 1;

        const auto desc = make_naive_tensor_descriptor(
            make_tuple(N, G, C, Hi, Wi), make_tuple(NStride, GStride, CStride, HiStride, WiStride));
        const auto merged_desc =
            transform_tensor_descriptor(desc,
                                        make_tuple(make_merge_transform(make_tuple(N, G, C)),
                                                   make_merge_transform(make_tuple(Hi, Wi))),
                                        make_tuple(Sequence<0, 1, 2>{}, Sequence<3, 4>{}),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}));
        return device::PadTensorDescriptor(
            merged_desc, make_tuple(MPerThread, NPerThread), Sequence<true, true>{});
    }

    template <ck::index_t NDim, typename ck::enable_if<NDim == 3, bool>::type = false>
    static auto MakeNGCHWTransposeDesc(std::array<ck::index_t, NDimSpatial + 3> g_n_c_wis_lengths,
                                       std::array<ck::index_t, NDimSpatial + 3> g_n_c_wis_strides)
    {
        const index_t& G  = g_n_c_wis_lengths[I0];
        const index_t& N  = g_n_c_wis_lengths[I1];
        const index_t& C  = g_n_c_wis_lengths[I2];
        const index_t& Di = g_n_c_wis_lengths[I3];
        const index_t& Hi = g_n_c_wis_lengths[I4];
        const index_t& Wi = g_n_c_wis_lengths[I5];

        const index_t& GStride  = g_n_c_wis_strides[I0];
        const index_t& NStride  = g_n_c_wis_strides[I1];
        const index_t& CStride  = g_n_c_wis_strides[I2];
        const index_t& DiStride = g_n_c_wis_strides[I3];
        const index_t& HiStride = g_n_c_wis_strides[I4];
        const index_t& WiStride = g_n_c_wis_strides[I5];

        const auto desc = make_naive_tensor_descriptor(
            make_tuple(N, G, C, Di, Hi, Wi),
            make_tuple(NStride, GStride, CStride, DiStride, HiStride, WiStride));
        const auto merged_desc =
            transform_tensor_descriptor(desc,
                                        make_tuple(make_merge_transform(make_tuple(N, G, C)),
                                                   make_merge_transform(make_tuple(Di, Hi, Wi))),
                                        make_tuple(Sequence<0, 1, 2>{}, Sequence<3, 4, 5>{}),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}));
        return device::PadTensorDescriptor(
            merged_desc, make_tuple(MPerThread, NPerThread), Sequence<true, true>{});
    }

    template <ck::index_t NDim, typename ck::enable_if<NDim == 3, bool>::type = false>
    static auto MakeNHWGCTransposeDesc(std::array<ck::index_t, NDimSpatial + 3> g_n_c_wis_lengths,
                                       std::array<ck::index_t, NDimSpatial + 3> g_n_c_wis_strides)
    {
        const index_t& G  = g_n_c_wis_lengths[I0];
        const index_t& N  = g_n_c_wis_lengths[I1];
        const index_t& C  = g_n_c_wis_lengths[I2];
        const index_t& Di = g_n_c_wis_lengths[I3];
        const index_t& Hi = g_n_c_wis_lengths[I4];
        const index_t& Wi = g_n_c_wis_lengths[I5];

        const index_t& NStride = g_n_c_wis_strides[I1];
        const index_t DiStride = Hi * Wi * G * C;
        const index_t HiStride = Wi * G * C;
        const index_t WiStride = G * C;
        const index_t GStride  = C;
        const index_t CStride  = 1;

        const auto desc = make_naive_tensor_descriptor(
            make_tuple(N, G, C, Di, Hi, Wi),
            make_tuple(NStride, GStride, CStride, DiStride, HiStride, WiStride));
        const auto merged_desc =
            transform_tensor_descriptor(desc,
                                        make_tuple(make_merge_transform(make_tuple(N, G, C)),
                                                   make_merge_transform(make_tuple(Di, Hi, Wi))),
                                        make_tuple(Sequence<0, 1, 2>{}, Sequence<3, 4, 5>{}),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}));
        return device::PadTensorDescriptor(
            merged_desc, make_tuple(MPerThread, NPerThread), Sequence<true, true>{});
    }

    static auto TransposeStrides(const std::array<index_t, NDimSpatial + 3>& g_n_c_wis_lengths,
                                 const std::array<index_t, NDimSpatial + 3>& g_n_c_wis_strides)
    {
        if constexpr(device::is_NGCHW_GKYXC_NGKHW<ALayout, BLayout, ELayout>() ||
                     device::is_NGCDHW_GKZYXC_NGKDHW<ALayout, BLayout, ELayout>())
        {
            std::array<index_t, NDimSpatial + 3> g_n_c_wis_strides_transposed;
            const auto G = g_n_c_wis_lengths[I0];
            const auto C = g_n_c_wis_lengths[I2];

            g_n_c_wis_strides_transposed[I0] = C;
            g_n_c_wis_strides_transposed[I1] = g_n_c_wis_strides[I1];
            g_n_c_wis_strides_transposed[I2] = I1;
            if constexpr(NDimSpatial == 2)
            {
                g_n_c_wis_strides_transposed[I3] = g_n_c_wis_lengths[I4] * G * C;
                g_n_c_wis_strides_transposed[I4] = G * C;
            }
            else if constexpr(NDimSpatial == 3)
            {
                g_n_c_wis_strides_transposed[I3] =
                    g_n_c_wis_lengths[I4] * g_n_c_wis_lengths[I5] * G * C;
                g_n_c_wis_strides_transposed[I4] = g_n_c_wis_lengths[I5] * G * C;
                g_n_c_wis_strides_transposed[I5] = G * C;
            }
            return g_n_c_wis_strides_transposed;
        }
        else
        {
            // transpose not needed
            return g_n_c_wis_strides;
        }
    }
};

} // namespace tensor_operation
} // namespace ck
