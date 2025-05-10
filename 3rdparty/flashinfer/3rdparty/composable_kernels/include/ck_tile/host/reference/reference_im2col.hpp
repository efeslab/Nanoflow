// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/host_tensor.hpp"
#include <thread>

namespace ck_tile {

template <typename T>
CK_TILE_HOST void reference_im2col(HostTensor<T>& in_mtx_host_ref,
                                   const HostTensor<T>& in_host,
                                   int /*N*/,
                                   int /*K*/,
                                   int C,
                                   int /*Y*/,
                                   int X,
                                   int Hi,
                                   int Wi,
                                   int Ho,
                                   int Wo,
                                   int ConvStrideH,
                                   int ConvStrideW,
                                   int ConvDilationH,
                                   int ConvDilationW,
                                   int InLeftPadH,
                                   int InLeftPadW,
                                   int /*InRightPadH*/,
                                   int /*InRightPadW*/)
{
    int GemmM = in_mtx_host_ref.get_lengths()[0];
    int GemmK = in_mtx_host_ref.get_lengths()[1];

    for(int gemm_m = 0; gemm_m < GemmM; ++gemm_m)
    {
        int mtmp = gemm_m;
        int n    = mtmp / (Ho * Wo);
        mtmp -= n * Ho * Wo;
        int ho = mtmp / Wo;
        int wo = mtmp - ho * Wo;

        for(int gemm_k = 0; gemm_k < GemmK; ++gemm_k)
        {
            int ktmp = gemm_k;
            int y    = ktmp / (X * C);
            ktmp -= y * X * C;
            int x = ktmp / C;
            int c = ktmp - x * C;

            int hi = y * ConvDilationH + ho * ConvStrideH - InLeftPadH;
            int wi = x * ConvDilationW + wo * ConvStrideW - InLeftPadW;

            bool inbound = (hi >= 0 && hi < Hi && wi >= 0 && wi < Wi);

            in_mtx_host_ref(gemm_m, gemm_k) = inbound ? in_host(n, hi, wi, c) : 0;
        }
    }
}
} // namespace ck_tile
