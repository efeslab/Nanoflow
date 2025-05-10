// SPDX-License-Identifier: MIT
// Copyright (c) 2023-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

// 1d
template <typename InLayout, typename WeiLayout, typename OutLayout>
constexpr bool is_NWGK_GKXC_NWGC()
{
    return is_same_v<InLayout, tensor_layout::convolution::NWGC> &&
           is_same_v<WeiLayout, tensor_layout::convolution::GKXC> &&
           is_same_v<OutLayout, tensor_layout::convolution::NWGK>;
}

template <typename InLayout, typename WeiLayout, typename OutLayout>
constexpr bool is_GNWK_GKXC_GNWC()
{
    return is_same_v<InLayout, tensor_layout::convolution::GNWC> &&
           is_same_v<WeiLayout, tensor_layout::convolution::GKXC> &&
           is_same_v<OutLayout, tensor_layout::convolution::GNWK>;
}
// 2d
template <typename InLayout, typename WeiLayout, typename OutLayout>
constexpr bool is_NHWGK_GKYXC_NHWGC()
{
    return is_same_v<InLayout, tensor_layout::convolution::NHWGC> &&
           is_same_v<WeiLayout, tensor_layout::convolution::GKYXC> &&
           is_same_v<OutLayout, tensor_layout::convolution::NHWGK>;
}

template <typename InLayout, typename WeiLayout, typename OutLayout>
constexpr bool is_GNHWK_GKYXC_GNHWC()
{
    return is_same_v<InLayout, tensor_layout::convolution::GNHWC> &&
           is_same_v<WeiLayout, tensor_layout::convolution::GKYXC> &&
           is_same_v<OutLayout, tensor_layout::convolution::GNHWK>;
}
// 3d
template <typename InLayout, typename WeiLayout, typename OutLayout>
constexpr bool is_NDHWGK_GKZYXC_NDHWGC()
{
    return is_same_v<InLayout, tensor_layout::convolution::NDHWGC> &&
           is_same_v<WeiLayout, tensor_layout::convolution::GKZYXC> &&
           is_same_v<OutLayout, tensor_layout::convolution::NDHWGK>;
}

template <typename InLayout, typename WeiLayout, typename OutLayout>
constexpr bool is_GNDHWK_GKZYXC_GNDHWC()
{
    return is_same_v<InLayout, tensor_layout::convolution::GNDHWC> &&
           is_same_v<WeiLayout, tensor_layout::convolution::GKZYXC> &&
           is_same_v<OutLayout, tensor_layout::convolution::GNDHWK>;
}

template <index_t NumATensor = 1, index_t NumBTensor = 1, index_t NumDTensor = 0, typename = void>
struct ComputePtrOffsetOfStridedBatch
{
};

template <index_t NumATensor, index_t NumBTensor, index_t NumDTensor>
struct ComputePtrOffsetOfStridedBatch<NumATensor,
                                      NumBTensor,
                                      NumDTensor,
                                      ck::enable_if_t<(NumATensor > 1 || NumBTensor > 1)>>
{
    ComputePtrOffsetOfStridedBatch() = default;

    ComputePtrOffsetOfStridedBatch(Array<ck::index_t, NumATensor>& BatchStrideAs,
                                   Array<ck::index_t, NumBTensor>& BatchStrideBs,
                                   Array<ck::index_t, NumDTensor>& BatchStrideDs,
                                   index_t BatchStrideE)
        : BatchStrideA_(BatchStrideAs),
          BatchStrideB_(BatchStrideBs),
          BatchStrideDs_(BatchStrideDs),
          BatchStrideE_(BatchStrideE)
    {
    }

    __host__ __device__ constexpr auto GetAsPtrOffset(index_t g_idx) const
    {
        Array<long_index_t, NumATensor> as_offset;
        static_for<0, NumATensor, 1>{}(
            [&](auto i) { as_offset(i) = g_idx * static_cast<long_index_t>(BatchStrideA_[i]); });
        return as_offset;
    }

    __host__ __device__ constexpr auto GetBsPtrOffset(index_t g_idx) const
    {
        Array<long_index_t, NumBTensor> bs_offset;
        static_for<0, NumBTensor, 1>{}(
            [&](auto i) { bs_offset(i) = g_idx * static_cast<long_index_t>(BatchStrideB_[i]); });
        return bs_offset;
    }

    __host__ __device__ constexpr auto GetDsPtrOffset(index_t g_idx) const
    {
        Array<long_index_t, NumDTensor> ds_offset;
        static_for<0, NumDTensor, 1>{}(
            [&](auto i) { ds_offset(i) = g_idx * static_cast<long_index_t>(BatchStrideDs_[i]); });
        return ds_offset;
    }

    [[maybe_unused]] __host__ __device__ constexpr long_index_t GetEPtrOffset(index_t g_idx) const
    {
        return g_idx * static_cast<long_index_t>(BatchStrideE_);
    }

    // alias for kernels without multiple D
    [[maybe_unused]] __host__ __device__ constexpr long_index_t GetCPtrOffset(index_t g_idx) const
    {
        return g_idx * static_cast<long_index_t>(BatchStrideE_);
    }

    Array<ck::index_t, NumATensor> BatchStrideA_;
    Array<ck::index_t, NumBTensor> BatchStrideB_;
    Array<ck::index_t, NumDTensor> BatchStrideDs_;
    index_t BatchStrideE_;
    index_t& BatchStrideC_ = BatchStrideE_; // alias for kernels without multiple D
};

template <index_t NumATensor, index_t NumBTensor, index_t NumDTensor>
struct ComputePtrOffsetOfStridedBatch<NumATensor,
                                      NumBTensor,
                                      NumDTensor,
                                      ck::enable_if_t<(NumATensor == 1 && NumBTensor == 1)>>
{
    ComputePtrOffsetOfStridedBatch() = default;

    ComputePtrOffsetOfStridedBatch(index_t BatchStrideA,
                                   index_t BatchStrideB,
                                   Array<ck::index_t, NumDTensor> BatchStrideDs,
                                   index_t BatchStrideE)
        : BatchStrideA_(BatchStrideA),
          BatchStrideB_(BatchStrideB),
          BatchStrideDs_(BatchStrideDs),
          BatchStrideE_(BatchStrideE)
    {
    }

    __host__ __device__ constexpr long_index_t GetAPtrOffset(index_t g_idx) const
    {
        return g_idx * static_cast<long_index_t>(BatchStrideA_);
    }

    __host__ __device__ constexpr long_index_t GetBPtrOffset(index_t g_idx) const
    {
        return g_idx * static_cast<long_index_t>(BatchStrideB_);
    }

    __host__ __device__ constexpr auto GetDsPtrOffset(index_t g_idx) const
    {
        Array<long_index_t, NumDTensor> ds_offset;
        static_for<0, NumDTensor, 1>{}(
            [&](auto i) { ds_offset(i) = g_idx * static_cast<long_index_t>(BatchStrideDs_[i]); });
        return ds_offset;
    }

    [[maybe_unused]] __host__ __device__ constexpr long_index_t GetEPtrOffset(index_t g_idx) const
    {
        return g_idx * static_cast<long_index_t>(BatchStrideE_);
    }

    // alias for kernels without multiple D
    [[maybe_unused]] __host__ __device__ constexpr long_index_t GetCPtrOffset(index_t g_idx) const
    {
        return g_idx * static_cast<long_index_t>(BatchStrideE_);
    }

    ck::index_t BatchStrideA_;
    ck::index_t BatchStrideB_;
    Array<ck::index_t, NumDTensor> BatchStrideDs_;
    index_t BatchStrideE_;
    index_t& BatchStrideC_ = BatchStrideE_; // alias for kernels without multiple D
};

template <bool isTuple, typename Tensors>
constexpr static auto GetNumABTensors()
{
    if constexpr(isTuple)
    {
        return Number<Tensors::Size()>{};
    }
    else
    {
        return Number<1>{};
    }
}

template <bool isTuple, typename GridwiseGemm, typename DataType>
constexpr static auto GetAGridPointer()
{
    if constexpr(isTuple)
    {
        return typename GridwiseGemm::AsGridPointer{};
    }
    else
    {
        return Tuple<const DataType*>{};
    }
}

template <bool isTuple, typename GridwiseGemm, typename DataType>
constexpr static auto GetBGridPointer()
{
    if constexpr(isTuple)
    {
        return typename GridwiseGemm::BsGridPointer{};
    }
    else
    {
        return Tuple<const DataType*>{};
    }
}

template <bool isTuple, typename Id, typename Type>
constexpr static auto UnpackDataType()
{
    if constexpr(isTuple)
    {
        // unpack if tuple
        return tuple_element_t<Id{}, Type>{};
    }
    else
    {
        // if no, return Type
        return Type{};
    }
}

} // namespace device
} // namespace tensor_operation
} // namespace ck
