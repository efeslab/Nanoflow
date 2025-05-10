// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_attribute_mfma_impl.hpp"

namespace ck_tile {

template <typename WarpGemmAttributeMfmaImpl_>
struct WarpGemmAtrributeMfma
{
    using Impl = remove_cvref_t<WarpGemmAttributeMfmaImpl_>;

    using ADataType = typename Impl::ADataType;
    using BDataType = typename Impl::BDataType;
    using CDataType = typename Impl::CDataType;

    using AVecType = typename Impl::AVecType;
    using BVecType = typename Impl::BVecType;
    using CVecType = typename Impl::CVecType;

    static constexpr index_t kM = Impl::kM;
    static constexpr index_t kN = Impl::kN;
    static constexpr index_t kK = Impl::kK;

    using AWarpDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<Impl::kAMLane>, sequence<Impl::kABKLane, Impl::kABKPerLane>>,
        tuple<sequence<2, 1>>,
        tuple<sequence<0, 0>>,
        sequence<2>,
        sequence<1>>;

    using BWarpDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<Impl::kBNLane>, sequence<Impl::kABKLane, Impl::kABKPerLane>>,
        tuple<sequence<2, 1>>,
        tuple<sequence<0, 0>>,
        sequence<2>,
        sequence<1>>;

    using CWarpDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<Impl::kCM0PerLane, Impl::kCMLane, Impl::kCM1PerLane>,
              sequence<Impl::kCNLane>>,
        tuple<sequence<1, 2>>,
        tuple<sequence<1, 0>>,
        sequence<1, 1>,
        sequence<0, 2>>;

    // c_vec += a_vec * b_vec
    CK_TILE_DEVICE void
    operator()(CVecType& c_vec, const AVecType& a_vec, const BVecType& b_vec) const
    {
        Impl{}(c_vec, a_vec, b_vec);
    }

    // c_vec = a_vec * b_vec
    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
        return Impl{}(a_vec, b_vec);
    }
};

template <typename WarpGemmAttributeMfmaImpl_, index_t kKIter>
struct WarpGemmAtrributeMfmaIterateK
{
    static_assert(kKIter > 0, "wrong!");

    using Impl = remove_cvref_t<WarpGemmAttributeMfmaImpl_>;

    using ADataType = typename Impl::ADataType;
    using BDataType = typename Impl::BDataType;
    using CDataType = typename Impl::CDataType;

    using AVecType =
        ext_vector_t<ADataType, vector_traits<typename Impl::AVecType>::vector_size * kKIter>;
    using BVecType =
        ext_vector_t<BDataType, vector_traits<typename Impl::BVecType>::vector_size * kKIter>;
    using CVecType = typename Impl::CVecType;

    static constexpr index_t kM = Impl::kM;
    static constexpr index_t kN = Impl::kN;
    static constexpr index_t kK = Impl::kK * kKIter;

    using AWarpDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<Impl::kAMLane>, sequence<Impl::kABKLane, Impl::kABKPerLane * kKIter>>,
        tuple<sequence<2, 1>>,
        tuple<sequence<0, 0>>,
        sequence<2>,
        sequence<1>>;

    using BWarpDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<Impl::kBNLane>, sequence<Impl::kABKLane, Impl::kABKPerLane * kKIter>>,
        tuple<sequence<2, 1>>,
        tuple<sequence<0, 0>>,
        sequence<2>,
        sequence<1>>;

    using CWarpDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<Impl::kCM0PerLane, Impl::kCMLane, Impl::kCM1PerLane>,
              sequence<Impl::kCNLane>>,
        tuple<sequence<1, 2>>,
        tuple<sequence<1, 0>>,
        sequence<1, 1>,
        sequence<0, 2>>;

    // c_vec += a_vec * b_vec
    CK_TILE_DEVICE void
    operator()(CVecType& c_vec, const AVecType& a_vec, const BVecType& b_vec) const
    {
        using buf_a = thread_buffer<typename Impl::AVecType, kKIter>;
        using buf_b = thread_buffer<typename Impl::BVecType, kKIter>;

        static_for<0, kKIter, 1>{}([&](auto iKIter) {
            Impl{}(c_vec,
                   reinterpret_cast<const buf_a>(a_vec)
                       .template get_as<typename Impl::AVecType>()[iKIter],
                   reinterpret_cast<const buf_b>(b_vec)
                       .template get_as<typename Impl::BVecType>()[iKIter]);
        });
    }

    // c_vec = a_vec * b_vec
    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
        constexpr auto I0 = number<0>{};
        using buf_a       = thread_buffer<typename Impl::AVecType, kKIter>;
        using buf_b       = thread_buffer<typename Impl::BVecType, kKIter>;

        // c = a * b
        auto c_vec = Impl{}(
            reinterpret_cast<const buf_a>(a_vec).template get_as<typename Impl::AVecType>()[I0],
            reinterpret_cast<const buf_b>(b_vec).template get_as<typename Impl::BVecType>()[I0]);

        // c += a * b
        static_for<1, kKIter, 1>{}([&](auto iKIter) {
            Impl{}(c_vec,
                   reinterpret_cast<const buf_a>(a_vec)
                       .template get_as<typename Impl::AVecType>()[iKIter],
                   reinterpret_cast<const buf_b>(b_vec)
                       .template get_as<typename Impl::BVecType>()[iKIter]);
        });

        return c_vec;
    }
};

template <typename WarpGemmAttributeMfmaImpl_>
struct WarpGemmAtrributeMfmaTransposedCDistribution
{
    using Impl = remove_cvref_t<WarpGemmAttributeMfmaImpl_>;

    using ADataType = typename Impl::BDataType;
    using BDataType = typename Impl::ADataType;
    using CDataType = typename Impl::CDataType;

    using AVecType = typename Impl::BVecType;
    using BVecType = typename Impl::AVecType;
    using CVecType = typename Impl::CVecType;

    static constexpr index_t kM = Impl::kN;
    static constexpr index_t kN = Impl::kM;
    static constexpr index_t kK = Impl::kK;

    using AWarpDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<Impl::kBNLane>, sequence<Impl::kABKLane, Impl::kABKPerLane>>,
        tuple<sequence<2, 1>>,
        tuple<sequence<0, 0>>,
        sequence<2>,
        sequence<1>>;

    using BWarpDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<Impl::kAMLane>, sequence<Impl::kABKLane, Impl::kABKPerLane>>,
        tuple<sequence<2, 1>>,
        tuple<sequence<0, 0>>,
        sequence<2>,
        sequence<1>>;

    using CWarpDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<Impl::kCNLane>,
              sequence<Impl::kCM0PerLane, Impl::kCMLane, Impl::kCM1PerLane>>,
        tuple<sequence<2, 1>>,
        tuple<sequence<1, 0>>,
        sequence<2, 2>,
        sequence<0, 2>>;

    // c_vec += a_vec * b_vec
    CK_TILE_DEVICE void
    operator()(CVecType& c_vec, const AVecType& a_vec, const BVecType& b_vec) const
    {
        // swap A and B
        Impl{}(c_vec, b_vec, a_vec);
    }

    // c_vec = a_vec * b_vec
    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
        // swap A and B
        return Impl{}(b_vec, a_vec);
    }
};

template <typename WarpGemmAttributeMfmaImpl_>
struct WarpGemmAtrributeMfmaTransposedCDistribution_SwizzleB
{
    using Impl = remove_cvref_t<WarpGemmAttributeMfmaImpl_>;

    using ADataType = typename Impl::BDataType;
    using BDataType = typename Impl::ADataType;
    using CDataType = typename Impl::CDataType;

    using AVecType = typename Impl::BVecType;
    using BVecType = typename Impl::AVecType;
    using CVecType = typename Impl::CVecType;

    static constexpr index_t kM = Impl::kN;
    static constexpr index_t kN = Impl::kM;
    static constexpr index_t kK = Impl::kK;

    using AWarpDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<Impl::kBNLane>, sequence<Impl::kABKLane, Impl::kABKPerLane>>,
        tuple<sequence<2, 1>>,
        tuple<sequence<0, 0>>,
        sequence<2>,
        sequence<1>>;

    using BWarpDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<Impl::kAMLane / (Impl::kABKPerLane * Impl::kABKLane * 2),
                       Impl::kABKLane,
                       2,
                       Impl::kABKPerLane>,
              sequence<Impl::kABKLane, Impl::kABKPerLane>>,
        tuple<sequence<2, 1, 1, 1, 1>>,
        tuple<sequence<0, 0, 2, 1, 3>>,
        sequence<2>,
        sequence<1>>;

    using CWarpDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<Impl::kCNLane>,
              sequence<Impl::kCM0PerLane / 2, Impl::kCMLane, Impl::kCM1PerLane * 2>>,
        tuple<sequence<2, 1>>,
        tuple<sequence<1, 0>>,
        sequence<2, 2>,
        sequence<0, 2>>;

    // c_vec += a_vec * b_vec
    CK_TILE_DEVICE void
    operator()(CVecType& c_vec, const AVecType& a_vec, const BVecType& b_vec) const
    {
        // swap A and B
        Impl{}(c_vec, b_vec, a_vec);
    }

    // c_vec = a_vec * b_vec
    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
        // swap A and B
        return Impl{}(b_vec, a_vec);
    }
};

template <typename WarpGemmAttributeMfmaImpl_, index_t kKIter>
struct WarpGemmAtrributeMfmaIterateKAndTransposedCDistribution
{
    using Impl = remove_cvref_t<WarpGemmAttributeMfmaImpl_>;

    // swap A and B
    using ADataType = typename Impl::BDataType;
    using BDataType = typename Impl::ADataType;
    using CDataType = typename Impl::CDataType;

    using AVecType =
        ext_vector_t<ADataType, vector_traits<typename Impl::AVecType>::vector_size * kKIter>;
    using BVecType =
        ext_vector_t<BDataType, vector_traits<typename Impl::BVecType>::vector_size * kKIter>;
    using CVecType = typename Impl::CVecType;

    static constexpr index_t kM = Impl::kN;
    static constexpr index_t kN = Impl::kM;
    static constexpr index_t kK = Impl::kK * kKIter;

    using AWarpDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<Impl::kBNLane>, sequence<Impl::kABKLane, Impl::kABKPerLane * kKIter>>,
        tuple<sequence<2, 1>>,
        tuple<sequence<0, 0>>,
        sequence<2>,
        sequence<1>>;

    using BWarpDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<Impl::kAMLane>, sequence<Impl::kABKLane, Impl::kABKPerLane * kKIter>>,
        tuple<sequence<2, 1>>,
        tuple<sequence<0, 0>>,
        sequence<2>,
        sequence<1>>;

    using CWarpDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<Impl::kCNLane>,
              sequence<Impl::kCM0PerLane, Impl::kCMLane, Impl::kCM1PerLane>>,
        tuple<sequence<2, 1>>,
        tuple<sequence<1, 0>>,
        sequence<2, 2>,
        sequence<0, 2>>;

    // c_vec += a_vec * b_vec
    CK_TILE_DEVICE void
    operator()(CVecType& c_vec, const AVecType& a_vec, const BVecType& b_vec) const
    {
        using buf_a = thread_buffer<typename Impl::AVecType, kKIter>;
        using buf_b = thread_buffer<typename Impl::BVecType, kKIter>;
        // swap A and B, value and type
        static_for<0, kKIter, 1>{}([&](auto iKIter) {
            Impl{}(c_vec,
                   reinterpret_cast<const buf_b&>(b_vec)
                       .template get_as<typename Impl::BVecType>()[iKIter],
                   reinterpret_cast<const buf_a&>(a_vec)
                       .template get_as<typename Impl::AVecType>()[iKIter]);
        });
    }

    // c_vec = a_vec * b_vec
    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
        constexpr auto I0 = number<0>{};
        using buf_a       = thread_buffer<typename Impl::AVecType, kKIter>;
        using buf_b       = thread_buffer<typename Impl::BVecType, kKIter>;

        // swap A and B, value and type
        auto c_vec = Impl{}(
            reinterpret_cast<const buf_b&>(b_vec).template get_as<typename Impl::BVecType>()[I0],
            reinterpret_cast<const buf_a&>(a_vec).template get_as<typename Impl::AVecType>()[I0]);

        static_for<1, kKIter, 1>{}([&](auto iKIter) {
            Impl{}(c_vec,
                   reinterpret_cast<const buf_b&>(b_vec)
                       .template get_as<typename Impl::BVecType>()[iKIter],
                   reinterpret_cast<const buf_a&>(a_vec)
                       .template get_as<typename Impl::AVecType>()[iKIter]);
        });

        return c_vec;
    }
};

template <typename WarpGemmAttributeMfmaImpl_, index_t kKIter, index_t SFactor_ = 2>
struct WarpGemmAtrributeMfmaIterateKAndTransposedCDistribution_SwizzleB
{
    using Impl = remove_cvref_t<WarpGemmAttributeMfmaImpl_>;

    // swap A and B
    using ADataType = typename Impl::BDataType;
    using BDataType = typename Impl::ADataType;
    using CDataType = typename Impl::CDataType;

    using AVecType =
        ext_vector_t<ADataType, vector_traits<typename Impl::AVecType>::vector_size * kKIter>;
    using BVecType =
        ext_vector_t<BDataType, vector_traits<typename Impl::BVecType>::vector_size * kKIter>;
    using CVecType = typename Impl::CVecType;

    static constexpr index_t kM      = Impl::kN;
    static constexpr index_t kN      = Impl::kM;
    static constexpr index_t kK      = Impl::kK * kKIter;
    static constexpr index_t SFactor = SFactor_; // group how many CM1 together

    using AWarpDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<Impl::kBNLane>, sequence<Impl::kABKLane, Impl::kABKPerLane * kKIter>>,
        tuple<sequence<2, 1>>,
        tuple<sequence<0, 0>>,
        sequence<2>,
        sequence<1>>;
#if 0
    using BWarpDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<Impl::kAMLane / (Impl::kABKPerLane * Impl::kABKLane * 2),
                       Impl::kABKLane,
                       2,
                       Impl::kABKPerLane>,
              sequence<Impl::kABKLane, Impl::kABKPerLane * kKIter>>,
        tuple<sequence<2, 1, 1, 1, 1>>,
        tuple<sequence<0, 0, 2, 1, 3>>,
        sequence<2>,
        sequence<1>>;

    using CWarpDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<Impl::kCNLane>,
              sequence<Impl::kCM0PerLane / 2, Impl::kCMLane, Impl::kCM1PerLane * 2>>,
        tuple<sequence<2, 1>>,
        tuple<sequence<1, 0>>,
        sequence<2, 2>,
        sequence<0, 2>>;
#else
    // TODO: more test not only 32x32
    using BWarpDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<Impl::kAMLane / (Impl::kCMLane * SFactor * Impl::kCM1PerLane),
                       Impl::kCMLane,
                       SFactor,
                       Impl::kCM1PerLane>,
              sequence<Impl::kABKLane, Impl::kABKPerLane * kKIter>>,
        tuple<sequence<2, 1, 1, 1, 1>>,
        tuple<sequence<0, 0, 2, 1, 3>>,
        sequence<2>,
        sequence<1>>;

    using CWarpDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<Impl::kCNLane>,
              sequence<Impl::kCM0PerLane / SFactor, Impl::kCMLane, Impl::kCM1PerLane * SFactor>>,
        tuple<sequence<2, 1>>,
        tuple<sequence<1, 0>>,
        sequence<2, 2>,
        sequence<0, 2>>;
#endif
    // c_vec += a_vec * b_vec
    CK_TILE_DEVICE void
    operator()(CVecType& c_vec, const AVecType& a_vec, const BVecType& b_vec) const
    {
        using buf_a = thread_buffer<typename Impl::AVecType, kKIter>;
        using buf_b = thread_buffer<typename Impl::BVecType, kKIter>;
        // swap A and B, value and type
        static_for<0, kKIter, 1>{}([&](auto iKIter) {
            Impl{}(c_vec,
                   reinterpret_cast<const buf_b&>(b_vec)
                       .template get_as<typename Impl::BVecType>()[iKIter],
                   reinterpret_cast<const buf_a&>(a_vec)
                       .template get_as<typename Impl::AVecType>()[iKIter]);
        });
    }

    // c_vec = a_vec * b_vec
    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
        using buf_a       = thread_buffer<typename Impl::AVecType, kKIter>;
        using buf_b       = thread_buffer<typename Impl::BVecType, kKIter>;
        constexpr auto I0 = number<0>{};

        // swap A and B, value and type
        auto c_vec = Impl{}(
            reinterpret_cast<const buf_b&>(b_vec).template get_as<typename Impl::BVecType>()[I0],
            reinterpret_cast<const buf_a&>(a_vec).template get_as<typename Impl::AVecType>()[I0]);

        static_for<1, kKIter, 1>{}([&](auto iKIter) {
            Impl{}(c_vec,
                   reinterpret_cast<const buf_b&>(b_vec)
                       .template get_as<typename Impl::BVecType>()[iKIter],
                   reinterpret_cast<const buf_a&>(a_vec)
                       .template get_as<typename Impl::AVecType>()[iKIter]);
        });

        return c_vec;
    }
};

} // namespace ck_tile
