// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.
#include "batched_transpose_example.hpp"
#include <iostream>

template <typename ts_type,
          ck_tile::index_t block_x,
          ck_tile::index_t block_y,
          ck_tile::index_t warp_x,
          ck_tile::index_t warp_y,
          ck_tile::index_t thread_x,
          ck_tile::index_t thread_y>
float batched_transpose_dispatch(batched_transpose_kargs& a, ck_tile::stream_config& s)
{
    uint32_t dim_block_h = (a.height + block_y - 1) / block_y;
    uint32_t dim_block_w = (a.width + block_x - 1) / block_x;
    uint32_t dim_stride  = a.height * a.width;

    a.dim_stride  = dim_stride;
    a.dim_block_h = dim_block_h;
    a.dim_block_w = dim_block_w;

    using block_tile  = ck_tile::sequence<block_x, block_y>;
    using warp_tile   = ck_tile::sequence<warp_x, warp_y>;
    using thread_tile = ck_tile::sequence<thread_x, thread_y>;

    using ts_problem =
        ck_tile::BatchedTransposeProblem<ts_type, block_tile, warp_tile, thread_tile>;
    using ts_pipeline = ck_tile::BatchedTransposePipeline<ts_problem>;

    using kernel = ck_tile::BatchedTransposeKernel<ts_pipeline>;

    auto kargs = kernel::MakeKargs(a);

    const dim3 grids      = kernel::GridSize(a);
    constexpr dim3 blocks = kernel::BlockSize();

    float ave_time = ck_tile::launch_kernel(
        s, ck_tile::make_kernel<blocks.x, 1>(kernel{}, grids, blocks, 0, kargs));

    return ave_time;
}

// Param Comb: type_size, block_x & y, warp_x & y, thread_x & y
#define FOREACH_TRANSPOSE_PARAM(F)               \
    F(fp16, ck_tile::fp16_t, 16, 16, 8, 8, 1, 1) \
    F(bf16, ck_tile::bf16_t, 16, 16, 8, 8, 1, 1) \
    F(fp32, ck_tile::fp32_t, 16, 16, 8, 8, 1, 1) \
    F(int8, ck_tile::int8_t, 16, 16, 8, 8, 1, 1)

// Macro that defines one static function per line
#define GEN_TRANSPOSE_FN(SHORT_NAME, REAL_TYPE, BX, BY, WX, WY, TX, TY)               \
    static float transpose_fn_##SHORT_NAME##_##BX##_##BY##_##WX##_##WY##_##TX##_##TY( \
        batched_transpose_kargs& a, ck_tile::stream_config& s)                        \
    {                                                                                 \
        return batched_transpose_dispatch<REAL_TYPE, BX, BY, WX, WY, TX, TY>(a, s);   \
    }

FOREACH_TRANSPOSE_PARAM(GEN_TRANSPOSE_FN)

float batched_transpose(batched_transpose_trait t,
                        batched_transpose_kargs a,
                        ck_tile::stream_config s)
{
    if(t.type == "fp16")
    {
        return transpose_fn_fp16_16_16_8_8_1_1(a, s);
    }
    else if(t.type == "bf16")
    {
        return transpose_fn_bf16_16_16_8_8_1_1(a, s);
    }
    else if(t.type == "fp32")
    {
        return transpose_fn_fp32_16_16_8_8_1_1(a, s);
    }
    else if(t.type == "int8")
    {
        return transpose_fn_int8_16_16_8_8_1_1(a, s);
    }
    return -1;
}
