// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
namespace ck_tile {

template <typename WarpGemmAttribute_>
struct WarpGemmImpl
{
    using WarpGemmAttribute = remove_cvref_t<WarpGemmAttribute_>;

    static constexpr index_t kM = WarpGemmAttribute::kM;
    static constexpr index_t kN = WarpGemmAttribute::kN;
    static constexpr index_t kK = WarpGemmAttribute::kK;

    using ADataType = typename WarpGemmAttribute::ADataType;
    using BDataType = typename WarpGemmAttribute::BDataType;
    using CDataType = typename WarpGemmAttribute::CDataType;

    using AWarpDstrEncoding = typename WarpGemmAttribute::AWarpDstrEncoding;
    using BWarpDstrEncoding = typename WarpGemmAttribute::BWarpDstrEncoding;
    using CWarpDstrEncoding = typename WarpGemmAttribute::CWarpDstrEncoding;

    using AWarpDstr = remove_cvref_t<decltype(make_static_tile_distribution(AWarpDstrEncoding{}))>;
    using BWarpDstr = remove_cvref_t<decltype(make_static_tile_distribution(BWarpDstrEncoding{}))>;
    using CWarpDstr = remove_cvref_t<decltype(make_static_tile_distribution(CWarpDstrEncoding{}))>;

    using AWarpTensor = static_distributed_tensor<ADataType, AWarpDstr>;
    using BWarpTensor = static_distributed_tensor<BDataType, BWarpDstr>;
    using CWarpTensor = static_distributed_tensor<CDataType, CWarpDstr>;

    CK_TILE_DEVICE void operator()(CWarpTensor& c, const AWarpTensor& a, const BWarpTensor& b) const
    {
        using AVec = ext_vector_t<ADataType, AWarpTensor::get_thread_buffer_size()>;
        using BVec = ext_vector_t<BDataType, BWarpTensor::get_thread_buffer_size()>;
        using CVec = ext_vector_t<CDataType, CWarpTensor::get_thread_buffer_size()>;

        constexpr auto I0 = number<0>{};

        const auto a_vec = a.get_thread_buffer().template get_as<AVec>()[I0];
        const auto b_vec = b.get_thread_buffer().template get_as<BVec>()[I0];
        auto c_vec       = c.get_thread_buffer().template get_as<CVec>()[I0];

        // c_vec += a_vec * b_vec
        WarpGemmAttribute{}(c_vec, a_vec, b_vec);

        c.get_thread_buffer().template set_as<CVec>(I0, c_vec);
    }

    CK_TILE_DEVICE auto operator()(const AWarpTensor& a, const BWarpTensor& b) const
    {
        CWarpTensor c;

        using AVec = ext_vector_t<ADataType, AWarpTensor::get_thread_buffer_size()>;
        using BVec = ext_vector_t<BDataType, BWarpTensor::get_thread_buffer_size()>;
        using CVec = ext_vector_t<CDataType, CWarpTensor::get_thread_buffer_size()>;

        constexpr auto I0 = number<0>{};

        const auto a_vec = a.get_thread_buffer().template get_as<AVec>()[I0];
        const auto b_vec = b.get_thread_buffer().template get_as<BVec>()[I0];

        // c_vec = a_vec * b_vec
        auto c_vec = WarpGemmAttribute{}(a_vec, b_vec);

        c.get_thread_buffer().template set_as<CVec>(I0, c_vec);

        return c;
    }
};

} // namespace ck_tile
