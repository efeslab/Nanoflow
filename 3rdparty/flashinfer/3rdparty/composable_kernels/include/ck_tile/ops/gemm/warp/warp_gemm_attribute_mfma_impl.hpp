// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

// FP16
struct WarpGemmAttributeMfmaImplF16F16F32M32N32K8
{
    using ADataType = fp16_t;
    using BDataType = fp16_t;
    using CDataType = float;

    using AVecType = ext_vector_t<fp16_t, 4>;
    using BVecType = ext_vector_t<fp16_t, 4>;
    using CVecType = ext_vector_t<float, 16>;

    static constexpr index_t kM = 32;
    static constexpr index_t kN = 32;
    static constexpr index_t kK = 8;

    static constexpr index_t kAMLane     = 32;
    static constexpr index_t kBNLane     = 32;
    static constexpr index_t kABKLane    = 2;
    static constexpr index_t kABKPerLane = 4;

    static constexpr index_t kCMLane     = 2;
    static constexpr index_t kCNLane     = 32;
    static constexpr index_t kCM0PerLane = 4;
    static constexpr index_t kCM1PerLane = 4;

    // c_vec += a_vec * b_vec
    CK_TILE_DEVICE void
    operator()(CVecType& c_vec, const AVecType& a_vec, const BVecType& b_vec) const
    {
#if defined(__gfx9__)
        c_vec = __builtin_amdgcn_mfma_f32_32x32x8f16(a_vec, b_vec, c_vec, 0, 0, 0);
#else
        ck_tile::ignore = c_vec;
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
#endif
    }

    // c_vec = a_vec * b_vec
    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
#if defined(__gfx9__)
        return bit_cast<CVecType>(
            __builtin_amdgcn_mfma_f32_32x32x8f16(a_vec, b_vec, fp32x16_t{0.f}, 0, 0, 0));
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        return CVecType{0.f};
#endif
    }
};

struct WarpGemmAttributeMfmaImplF16F16F32M16N16K16
{
    using ADataType = fp16_t;
    using BDataType = fp16_t;
    using CDataType = float;

    using AVecType = ext_vector_t<fp16_t, 4>;
    using BVecType = ext_vector_t<fp16_t, 4>;
    using CVecType = ext_vector_t<float, 4>;

    static constexpr index_t kM = 16;
    static constexpr index_t kN = 16;
    static constexpr index_t kK = 16;

    static constexpr index_t kAMLane     = 16;
    static constexpr index_t kBNLane     = 16;
    static constexpr index_t kABKLane    = 4;
    static constexpr index_t kABKPerLane = 4;

    static constexpr index_t kCMLane     = 4;
    static constexpr index_t kCNLane     = 16;
    static constexpr index_t kCM0PerLane = 1;
    static constexpr index_t kCM1PerLane = 4;

    // c_vec += a_vec * b_vec
    CK_TILE_DEVICE void
    operator()(CVecType& c_vec, const AVecType& a_vec, const BVecType& b_vec) const
    {
#if defined(__gfx9__)
        c_vec = __builtin_amdgcn_mfma_f32_16x16x16f16(a_vec, b_vec, c_vec, 0, 0, 0);
#else
        ck_tile::ignore = c_vec;
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
#endif
    }

    // c_vec = a_vec * b_vec
    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
#if defined(__gfx9__)
        return bit_cast<CVecType>(
            __builtin_amdgcn_mfma_f32_16x16x16f16(a_vec, b_vec, fp32x4_t{0.f}, 0, 0, 0));
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        return CVecType{0.f};
#endif
    }
};

// Bf16
struct WarpGemmAttributeMfmaImplBf16Bf16F32M32N32K8
{
    using ADataType = bf16_t;
    using BDataType = bf16_t;
    using CDataType = float;

    using AVecType = ext_vector_t<bf16_t, 4>;
    using BVecType = ext_vector_t<bf16_t, 4>;
    using CVecType = ext_vector_t<float, 16>;

    static constexpr index_t kM = 32;
    static constexpr index_t kN = 32;
    static constexpr index_t kK = 8;

    static constexpr index_t kAMLane     = 32;
    static constexpr index_t kBNLane     = 32;
    static constexpr index_t kABKLane    = 2;
    static constexpr index_t kABKPerLane = 4;

    static constexpr index_t kCMLane     = 2;
    static constexpr index_t kCNLane     = 32;
    static constexpr index_t kCM0PerLane = 4;
    static constexpr index_t kCM1PerLane = 4;

    // c_vec += a_vec * b_vec
    CK_TILE_DEVICE void
    operator()(CVecType& c_vec, const AVecType& a_vec, const BVecType& b_vec) const
    {
#if defined(__gfx90a__) || defined(__gfx94__)
        c_vec = __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(a_vec, b_vec, c_vec, 0, 0, 0);
#elif defined(__gfx908__)
        static_for<0, 2, 1>{}([&](auto k) {
            c_vec = __builtin_amdgcn_mfma_f32_32x32x4bf16(
                reinterpret_cast<const thread_buffer<ADataType, 4>&>(a_vec)
                    .template get_as<ext_vector_t<bf16_t, 2>>()[number<k>{}],
                reinterpret_cast<const thread_buffer<BDataType, 4>&>(b_vec)
                    .template get_as<ext_vector_t<bf16_t, 2>>()[number<k>{}],
                c_vec,
                0,
                0,
                0);
        });
#else
        ck_tile::ignore = c_vec;
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
#endif
    }

    // c_vec = a_vec * b_vec
    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
#if defined(__gfx90a__) || defined(__gfx94__)
        return bit_cast<CVecType>(
            __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(a_vec, b_vec, fp32x16_t{0.f}, 0, 0, 0));
#elif defined(__gfx908__)
        CVecType c_vec{0.f};
        static_for<0, 2, 1>{}([&](auto k) {
            c_vec = __builtin_amdgcn_mfma_f32_32x32x4bf16(
                reinterpret_cast<const thread_buffer<ADataType, 4>&>(a_vec)
                    .template get_as<ext_vector_t<bf16_t, 2>>()[number<k>{}],
                reinterpret_cast<const thread_buffer<BDataType, 4>&>(b_vec)
                    .template get_as<ext_vector_t<bf16_t, 2>>()[number<k>{}],
                c_vec,
                0,
                0,
                0);
        });
        return c_vec;
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        return CVecType{0.f};
#endif
    }
};

struct WarpGemmAttributeMfmaImplBf16Bf16F32M16N16K16
{
    using ADataType = bf16_t;
    using BDataType = bf16_t;
    using CDataType = float;

    using AVecType = ext_vector_t<bf16_t, 4>;
    using BVecType = ext_vector_t<bf16_t, 4>;
    using CVecType = ext_vector_t<float, 4>;

    static constexpr index_t kM = 16;
    static constexpr index_t kN = 16;
    static constexpr index_t kK = 16;

    static constexpr index_t kAMLane     = 16;
    static constexpr index_t kBNLane     = 16;
    static constexpr index_t kABKLane    = 4;
    static constexpr index_t kABKPerLane = 4;

    static constexpr index_t kCMLane     = 4;
    static constexpr index_t kCNLane     = 16;
    static constexpr index_t kCM0PerLane = 1;
    static constexpr index_t kCM1PerLane = 4;

    // c_vec += a_vec * b_vec
    CK_TILE_DEVICE void
    operator()(CVecType& c_vec, const AVecType& a_vec, const BVecType& b_vec) const
    {
#if defined(__gfx90a__) || defined(__gfx94__)
        c_vec = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a_vec, b_vec, c_vec, 0, 0, 0);
#elif defined(__gfx908__)
        static_for<0, 2, 1>{}([&](auto k) {
            c_vec = __builtin_amdgcn_mfma_f32_16x16x8bf16(
                reinterpret_cast<const thread_buffer<ADataType, 4>&>(a_vec)
                    .template get_as<ext_vector_t<bf16_t, 2>>()[number<k>{}],
                reinterpret_cast<const thread_buffer<BDataType, 4>&>(b_vec)
                    .template get_as<ext_vector_t<bf16_t, 2>>()[number<k>{}],
                c_vec,
                0,
                0,
                0);
        });
#else
        ck_tile::ignore = c_vec;
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
#endif
    }

    // c_vec = a_vec * b_vec
    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
#if defined(__gfx90a__) || defined(__gfx94__)
        return bit_cast<CVecType>(
            __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a_vec, b_vec, fp32x4_t{0.f}, 0, 0, 0));
#elif defined(__gfx908__)
        CVecType c_vec{0.f};
        static_for<0, 2, 1>{}([&](auto k) {
            c_vec = __builtin_amdgcn_mfma_f32_16x16x8bf16(
                reinterpret_cast<const thread_buffer<ADataType, 4>&>(a_vec)
                    .template get_as<ext_vector_t<bf16_t, 2>>()[number<k>{}],
                reinterpret_cast<const thread_buffer<BDataType, 4>&>(b_vec)
                    .template get_as<ext_vector_t<bf16_t, 2>>()[number<k>{}],
                c_vec,
                0,
                0,
                0);
        });
        return c_vec;
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        return CVecType{0.f};
#endif
    }
};

// FP8
template <typename AType_, typename BType_>
struct WarpGemmAttributeMfmaImpl_f32_32x32x16_f8_base
{
    using ADataType = AType_;
    using BDataType = BType_;
    using CDataType = float;

    using AVecType = ext_vector_t<ADataType, 8>;
    using BVecType = ext_vector_t<BDataType, 8>;
    using CVecType = ext_vector_t<CDataType, 16>;

    static constexpr index_t kM = 32;
    static constexpr index_t kN = 32;
    static constexpr index_t kK = 16;

    static constexpr index_t kAMLane     = 32;
    static constexpr index_t kBNLane     = 32;
    static constexpr index_t kABKLane    = 2;
    static constexpr index_t kABKPerLane = 8;

    static constexpr index_t kCMLane     = 2;
    static constexpr index_t kCNLane     = 32;
    static constexpr index_t kCM0PerLane = 4;
    static constexpr index_t kCM1PerLane = 4;

    // c_vec += a_vec * b_vec
    CK_TILE_DEVICE void
    operator()(CVecType& c_vec, const AVecType& a_vec, const BVecType& b_vec) const
    {
#if defined(__gfx94__)
        if constexpr(std::is_same_v<ADataType, fp8_t> && std::is_same_v<BDataType, fp8_t>)
            c_vec = __builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(
                bit_cast<long>(a_vec), bit_cast<long>(b_vec), c_vec, 0, 0, 0);
        else if constexpr(std::is_same_v<ADataType, fp8_t> && std::is_same_v<BDataType, bf8_t>)
            c_vec = __builtin_amdgcn_mfma_f32_32x32x16_fp8_bf8(
                bit_cast<long>(a_vec), bit_cast<long>(b_vec), c_vec, 0, 0, 0);
        else if constexpr(std::is_same_v<ADataType, bf8_t> && std::is_same_v<BDataType, fp8_t>)
            c_vec = __builtin_amdgcn_mfma_f32_32x32x16_bf8_fp8(
                bit_cast<long>(a_vec), bit_cast<long>(b_vec), c_vec, 0, 0, 0);
        else if constexpr(std::is_same_v<ADataType, bf8_t> && std::is_same_v<BDataType, bf8_t>)
            c_vec = __builtin_amdgcn_mfma_f32_32x32x16_bf8_bf8(
                bit_cast<long>(a_vec), bit_cast<long>(b_vec), c_vec, 0, 0, 0);
#elif defined(__gfx908__) || defined(__gfx90a__)
        static_for<0, 8, 1>{}([&](auto k) {
            float a_f32 =
                type_convert<float>(reinterpret_cast<const thread_buffer<ADataType, 8>&>(a_vec)
                                        .template get_as<ADataType>()[number<k>{}]);
            float b_f32 =
                type_convert<float>(reinterpret_cast<const thread_buffer<BDataType, 8>&>(b_vec)
                                        .template get_as<BDataType>()[number<k>{}]);

            c_vec = __builtin_amdgcn_mfma_f32_32x32x2f32(a_f32, b_f32, c_vec, 0, 0, 0);
        });
#else
        ck_tile::ignore = c_vec;
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
#endif
    }

    // c_vec = a_vec * b_vec
    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
#if defined(__gfx94__)
        if constexpr(std::is_same_v<ADataType, fp8_t> && std::is_same_v<BDataType, fp8_t>)
            return bit_cast<CVecType>(__builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(
                bit_cast<long>(a_vec), bit_cast<long>(b_vec), CVecType{0.f}, 0, 0, 0));
        else if constexpr(std::is_same_v<ADataType, fp8_t> && std::is_same_v<BDataType, bf8_t>)
            return bit_cast<CVecType>(__builtin_amdgcn_mfma_f32_32x32x16_fp8_bf8(
                bit_cast<long>(a_vec), bit_cast<long>(b_vec), CVecType{0.f}, 0, 0, 0));
        else if constexpr(std::is_same_v<ADataType, bf8_t> && std::is_same_v<BDataType, fp8_t>)
            return bit_cast<CVecType>(__builtin_amdgcn_mfma_f32_32x32x16_bf8_fp8(
                bit_cast<long>(a_vec), bit_cast<long>(b_vec), CVecType{0.f}, 0, 0, 0));
        else if constexpr(std::is_same_v<ADataType, bf8_t> && std::is_same_v<BDataType, bf8_t>)
            return bit_cast<CVecType>(__builtin_amdgcn_mfma_f32_32x32x16_bf8_bf8(
                bit_cast<long>(a_vec), bit_cast<long>(b_vec), CVecType{0.f}, 0, 0, 0));
#elif defined(__gfx908__) || defined(__gfx90a__)
        CVecType c_vec{0.f};
        static_for<0, 8, 1>{}([&](auto k) {
            float a_f32 =
                type_convert<float>(reinterpret_cast<const thread_buffer<ADataType, 8>&>(a_vec)
                                        .template get_as<ADataType>()[number<k>{}]);
            float b_f32 =
                type_convert<float>(reinterpret_cast<const thread_buffer<BDataType, 8>&>(b_vec)
                                        .template get_as<BDataType>()[number<k>{}]);

            c_vec = __builtin_amdgcn_mfma_f32_32x32x2f32(a_f32, b_f32, c_vec, 0, 0, 0);
        });
        return c_vec;
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        return CVecType{0.f};
#endif
    }
};

using WarpGemmAttributeMfmaImpl_f32_32x32x16_fp8_fp8 =
    WarpGemmAttributeMfmaImpl_f32_32x32x16_f8_base<fp8_t, fp8_t>;
using WarpGemmAttributeMfmaImpl_f32_32x32x16_fp8_bf8 =
    WarpGemmAttributeMfmaImpl_f32_32x32x16_f8_base<fp8_t, bf8_t>;
using WarpGemmAttributeMfmaImpl_f32_32x32x16_bf8_fp8 =
    WarpGemmAttributeMfmaImpl_f32_32x32x16_f8_base<bf8_t, fp8_t>;
using WarpGemmAttributeMfmaImpl_f32_32x32x16_bf8_bf8 =
    WarpGemmAttributeMfmaImpl_f32_32x32x16_f8_base<bf8_t, bf8_t>;

} // namespace ck_tile
