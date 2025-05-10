// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm.hpp"

namespace ck_tile {

namespace impl {
template <typename AType,
          typename BType,
          typename CType,
          index_t MPerWave,
          index_t NPerWave,
          index_t KPerWave,
          bool TransposeC>
struct WarpGemmMfmaDispatcher;

// clang-format off
// fp16
template<> struct WarpGemmMfmaDispatcher<ck_tile::half_t, ck_tile::half_t, float, 32, 32,  8, false> { using Type = WarpGemmMfmaF16F16F32M32N32K8; };
template<> struct WarpGemmMfmaDispatcher<ck_tile::half_t, ck_tile::half_t, float, 32, 32,  8, true> { using Type = WarpGemmMfmaF16F16F32M32N32K8TransposedCDistribution; };
template<> struct WarpGemmMfmaDispatcher<ck_tile::half_t, ck_tile::half_t, float, 32, 32, 16, false> { using Type = WarpGemmMfmaF16F16F32M32N32K16; };
template<> struct WarpGemmMfmaDispatcher<ck_tile::half_t, ck_tile::half_t, float, 32, 32, 16, true> { using Type = WarpGemmMfmaF16F16F32M32N32K16TransposedCDistribution; };
template<> struct WarpGemmMfmaDispatcher<ck_tile::half_t, ck_tile::half_t, float, 16, 16, 16, false> { using Type = WarpGemmMfmaF16F16F32M16N16K16; };
template<> struct WarpGemmMfmaDispatcher<ck_tile::half_t, ck_tile::half_t, float, 16, 16, 16, true> { using Type = WarpGemmMfmaF16F16F32M16N16K16TransposedCDistribution; };
template<> struct WarpGemmMfmaDispatcher<ck_tile::half_t, ck_tile::half_t, float, 16, 16, 32, false> { using Type = WarpGemmMfmaF16F16F32M16N16K32; };
template<> struct WarpGemmMfmaDispatcher<ck_tile::half_t, ck_tile::half_t, float, 16, 16, 32, true> { using Type = WarpGemmMfmaF16F16F32M16N16K32TransposedCDistribution; };

// bf16
template<> struct WarpGemmMfmaDispatcher<ck_tile::bf16_t, ck_tile::bf16_t, float, 32, 32,  8, false> { using Type = WarpGemmMfmaBf16Bf16F32M32N32K8; };
template<> struct WarpGemmMfmaDispatcher<ck_tile::bf16_t, ck_tile::bf16_t, float, 32, 32,  8, true> { using Type = WarpGemmMfmaBf16Bf16F32M32N32K8TransposedCDistribution; };
template<> struct WarpGemmMfmaDispatcher<ck_tile::bf16_t, ck_tile::bf16_t, float, 32, 32, 16, false> { using Type = WarpGemmMfmaBf16Bf16F32M32N32K16; };
template<> struct WarpGemmMfmaDispatcher<ck_tile::bf16_t, ck_tile::bf16_t, float, 32, 32, 16, true> { using Type = WarpGemmMfmaBf16Bf16F32M32N32K16TransposedCDistribution; };
template<> struct WarpGemmMfmaDispatcher<ck_tile::bf16_t, ck_tile::bf16_t, float, 16, 16, 16, false> { using Type = WarpGemmMfmaBf16Bf16F32M16N16K16; };
template<> struct WarpGemmMfmaDispatcher<ck_tile::bf16_t, ck_tile::bf16_t, float, 16, 16, 16, true> { using Type = WarpGemmMfmaBf16Bf16F32M16N16K16TransposedCDistribution; };
template<> struct WarpGemmMfmaDispatcher<ck_tile::bf16_t, ck_tile::bf16_t, float, 16, 16, 32, false> { using Type = WarpGemmMfmaBf16Bf16F32M16N16K32; };
template<> struct WarpGemmMfmaDispatcher<ck_tile::bf16_t, ck_tile::bf16_t, float, 16, 16, 32, true> { using Type = WarpGemmMfmaBf16Bf16F32M16N16K32TransposedCDistribution; };

// fp8
template<> struct WarpGemmMfmaDispatcher<ck_tile::fp8_t, ck_tile::fp8_t, float, 32, 32,  16, false> { using Type = WarpGemmMfma_f32_32x32x16_fp8_fp8; };
template<> struct WarpGemmMfmaDispatcher<ck_tile::fp8_t, ck_tile::fp8_t, float, 32, 32,  16, true> { using Type = WarpGemmMfma_f32_32x32x16_fp8_fp8_CTransposed; };
template<> struct WarpGemmMfmaDispatcher<ck_tile::fp8_t, ck_tile::bf8_t, float, 32, 32,  16, false> { using Type = WarpGemmMfma_f32_32x32x16_fp8_bf8; };
template<> struct WarpGemmMfmaDispatcher<ck_tile::fp8_t, ck_tile::bf8_t, float, 32, 32,  16, true> { using Type = WarpGemmMfma_f32_32x32x16_fp8_bf8_CTransposed; };
template<> struct WarpGemmMfmaDispatcher<ck_tile::bf8_t, ck_tile::fp8_t, float, 32, 32,  16, false> { using Type = WarpGemmMfma_f32_32x32x16_bf8_fp8; };
template<> struct WarpGemmMfmaDispatcher<ck_tile::bf8_t, ck_tile::fp8_t, float, 32, 32,  16, true> { using Type = WarpGemmMfma_f32_32x32x16_bf8_fp8_CTransposed; };
template<> struct WarpGemmMfmaDispatcher<ck_tile::bf8_t, ck_tile::bf8_t, float, 32, 32,  16, false> { using Type = WarpGemmMfma_f32_32x32x16_bf8_bf8; };
template<> struct WarpGemmMfmaDispatcher<ck_tile::bf8_t, ck_tile::bf8_t, float, 32, 32,  16, true> { using Type = WarpGemmMfma_f32_32x32x16_bf8_bf8_CTransposed; };

// clang-format on
} // namespace impl

template <typename AType,
          typename BType,
          typename CType,
          index_t MPerWave,
          index_t NPerWave,
          index_t KPerWave,
          bool TransposeC>
using WarpGemmMfmaDispatcher = typename impl::
    WarpGemmMfmaDispatcher<AType, BType, CType, MPerWave, NPerWave, KPerWave, TransposeC>::Type;

} // namespace ck_tile
