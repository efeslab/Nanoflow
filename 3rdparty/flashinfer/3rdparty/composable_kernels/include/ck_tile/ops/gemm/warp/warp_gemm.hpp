// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_impl.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_attribute_mfma.hpp"

namespace ck_tile {

// fp16
using WarpGemmMfmaF16F16F32M32N32K8 =
    WarpGemmImpl<WarpGemmAtrributeMfma<WarpGemmAttributeMfmaImplF16F16F32M32N32K8>>;

using WarpGemmMfmaF16F16F32M16N16K16 =
    WarpGemmImpl<WarpGemmAtrributeMfma<WarpGemmAttributeMfmaImplF16F16F32M16N16K16>>;

using WarpGemmMfmaF16F16F32M32N32K16 =
    WarpGemmImpl<WarpGemmAtrributeMfmaIterateK<WarpGemmAttributeMfmaImplF16F16F32M32N32K8, 2>>;

using WarpGemmMfmaF16F16F32M16N16K32 =
    WarpGemmImpl<WarpGemmAtrributeMfmaIterateK<WarpGemmAttributeMfmaImplF16F16F32M16N16K16, 2>>;

using WarpGemmMfmaF16F16F32M32N32K8TransposedCDistribution = WarpGemmImpl<
    WarpGemmAtrributeMfmaTransposedCDistribution<WarpGemmAttributeMfmaImplF16F16F32M32N32K8>>;

using WarpGemmMfmaF16F16F32M16N16K16TransposedCDistribution = WarpGemmImpl<
    WarpGemmAtrributeMfmaTransposedCDistribution<WarpGemmAttributeMfmaImplF16F16F32M16N16K16>>;

using WarpGemmMfmaF16F16F32M32N32K16TransposedCDistribution =
    WarpGemmImpl<WarpGemmAtrributeMfmaIterateKAndTransposedCDistribution<
        WarpGemmAttributeMfmaImplF16F16F32M32N32K8,
        2>>;

using WarpGemmMfmaF16F16F32M16N16K32TransposedCDistribution =
    WarpGemmImpl<WarpGemmAtrributeMfmaIterateKAndTransposedCDistribution<
        WarpGemmAttributeMfmaImplF16F16F32M16N16K16,
        2>>;

using WarpGemmMfmaF16F16F32M16N16K32SwizzleBTransposedCDistribution =
    WarpGemmImpl<WarpGemmAtrributeMfmaIterateKAndTransposedCDistribution_SwizzleB<
        WarpGemmAttributeMfmaImplF16F16F32M32N32K8,
        2>>;

// bf16
using WarpGemmMfmaBf16Bf16F32M32N32K8 =
    WarpGemmImpl<WarpGemmAtrributeMfma<WarpGemmAttributeMfmaImplBf16Bf16F32M32N32K8>>;

using WarpGemmMfmaBf16Bf16F32M16N16K16 =
    WarpGemmImpl<WarpGemmAtrributeMfma<WarpGemmAttributeMfmaImplBf16Bf16F32M16N16K16>>;

using WarpGemmMfmaBf16Bf16F32M32N32K16 =
    WarpGemmImpl<WarpGemmAtrributeMfmaIterateK<WarpGemmAttributeMfmaImplBf16Bf16F32M32N32K8, 2>>;

using WarpGemmMfmaBf16Bf16F32M16N16K32 =
    WarpGemmImpl<WarpGemmAtrributeMfmaIterateK<WarpGemmAttributeMfmaImplBf16Bf16F32M16N16K16, 2>>;

using WarpGemmMfmaBf16Bf16F32M32N32K8TransposedCDistribution = WarpGemmImpl<
    WarpGemmAtrributeMfmaTransposedCDistribution<WarpGemmAttributeMfmaImplBf16Bf16F32M32N32K8>>;

using WarpGemmMfmaBf16Bf16F32M16N16K16TransposedCDistribution = WarpGemmImpl<
    WarpGemmAtrributeMfmaTransposedCDistribution<WarpGemmAttributeMfmaImplBf16Bf16F32M16N16K16>>;

using WarpGemmMfmaBf16Bf16F32M32N32K16TransposedCDistribution =
    WarpGemmImpl<WarpGemmAtrributeMfmaIterateKAndTransposedCDistribution<
        WarpGemmAttributeMfmaImplBf16Bf16F32M32N32K8,
        2>>;

using WarpGemmMfmaBf16Bf16F32M16N16K32TransposedCDistribution =
    WarpGemmImpl<WarpGemmAtrributeMfmaIterateKAndTransposedCDistribution<
        WarpGemmAttributeMfmaImplBf16Bf16F32M16N16K16,
        2>>;

using WarpGemmMfmaBf16Bf16F32M16N16K32SwizzleBTransposedCDistribution =
    WarpGemmImpl<WarpGemmAtrributeMfmaIterateKAndTransposedCDistribution_SwizzleB<
        WarpGemmAttributeMfmaImplBf16Bf16F32M32N32K8,
        2>>;

// fp8
using WarpGemmMfma_f32_32x32x16_fp8_fp8 =
    WarpGemmImpl<WarpGemmAtrributeMfma<WarpGemmAttributeMfmaImpl_f32_32x32x16_fp8_fp8>>;

using WarpGemmMfma_f32_32x32x16_fp8_bf8 =
    WarpGemmImpl<WarpGemmAtrributeMfma<WarpGemmAttributeMfmaImpl_f32_32x32x16_fp8_bf8>>;

using WarpGemmMfma_f32_32x32x16_bf8_fp8 =
    WarpGemmImpl<WarpGemmAtrributeMfma<WarpGemmAttributeMfmaImpl_f32_32x32x16_bf8_fp8>>;

using WarpGemmMfma_f32_32x32x16_bf8_bf8 =
    WarpGemmImpl<WarpGemmAtrributeMfma<WarpGemmAttributeMfmaImpl_f32_32x32x16_bf8_bf8>>;

using WarpGemmMfma_f32_32x32x16_fp8_fp8_CTransposed = WarpGemmImpl<
    WarpGemmAtrributeMfmaTransposedCDistribution<WarpGemmAttributeMfmaImpl_f32_32x32x16_fp8_fp8>>;

using WarpGemmMfma_f32_32x32x16_fp8_bf8_CTransposed = WarpGemmImpl<
    WarpGemmAtrributeMfmaTransposedCDistribution<WarpGemmAttributeMfmaImpl_f32_32x32x16_fp8_bf8>>;

using WarpGemmMfma_f32_32x32x16_bf8_fp8_CTransposed = WarpGemmImpl<
    WarpGemmAtrributeMfmaTransposedCDistribution<WarpGemmAttributeMfmaImpl_f32_32x32x16_bf8_fp8>>;

using WarpGemmMfma_f32_32x32x16_bf8_bf8_CTransposed = WarpGemmImpl<
    WarpGemmAtrributeMfmaTransposedCDistribution<WarpGemmAttributeMfmaImpl_f32_32x32x16_bf8_bf8>>;

template <index_t swizzle_factor = 2>
using WarpGemmMfmaFp8Fp8F32M32N32K16SwizzleBTransposedCDistribution =
    WarpGemmImpl<WarpGemmAtrributeMfmaIterateKAndTransposedCDistribution_SwizzleB<
        WarpGemmAttributeMfmaImpl_f32_32x32x16_f8_base<fp8_t, fp8_t>,
        2,
        swizzle_factor>>;

} // namespace ck_tile
