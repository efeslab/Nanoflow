// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "moe_sorting_api.hpp"

#define MOE_SORTING_DISPATCH_ETILE(unroll_num_, expert_tile_)                         \
    constexpr ck_tile::index_t unroll_num  = unroll_num_;                             \
    constexpr ck_tile::index_t expert_tile = expert_tile_;                            \
    using ms_problem =                                                                \
        ck_tile::MoeSortingProblem<index_t, ms_weight_type, unroll_num, expert_tile>; \
    using kernel         = ck_tile::MoeSortingKernel<ms_problem>;                     \
    auto kargs           = kernel::MakeKargs(a);                                      \
    const dim3 grids     = kernel::GridSize(a);                                       \
    const dim3 blocks    = kernel::BlockSize(a);                                      \
    const auto lds_bytes = kernel::GetSmemSize(a);                                    \
    float ave_time       = ck_tile::launch_kernel(                                    \
        s, ck_tile::make_kernel(kernel{}, grids, blocks, lds_bytes, kargs));    \
    return ave_time;

#define MOE_SORTING_DISPATCH(unroll_num_)           \
    if(a.num_experts <= 8)                          \
    {                                               \
        MOE_SORTING_DISPATCH_ETILE(unroll_num_, 8)  \
    }                                               \
    else if(a.num_experts <= 16)                    \
    {                                               \
        MOE_SORTING_DISPATCH_ETILE(unroll_num_, 16) \
    }                                               \
    else if(a.num_experts <= 32)                    \
    {                                               \
        MOE_SORTING_DISPATCH_ETILE(unroll_num_, 32) \
    }                                               \
    else if(a.num_experts <= 64)                    \
    {                                               \
        MOE_SORTING_DISPATCH_ETILE(unroll_num_, 64) \
    }                                               \
    else                                            \
    {                                               \
        MOE_SORTING_DISPATCH_ETILE(unroll_num_, 0)  \
    }

float moe_sorting(moe_sorting_trait t, moe_sorting_args a, ck_tile::stream_config s)
{
    if(t.weight_type == "fp32" && t.index_type == "int32")
    {
        if(a.num_experts > 127)
        {
            printf("lds size exceed, only support experts <127 \n");
            return -1;
        }
        if(a.moe_buf_bytes % 16)
        {
            printf("buf set size %d unaligned, must be multiple of 16\n", a.moe_buf_bytes);
            return -1;
        }
        using index_t              = ck_tile::index_t;
        using ms_weight_type       = float;
        index_t smem_io_unroll_num = ck_tile::integer_divide_ceil(a.tokens * a.topk, 64);
        switch(smem_io_unroll_num)
        {
        case(1): {
            MOE_SORTING_DISPATCH(1);
        }
        case(2): {
            MOE_SORTING_DISPATCH(2);
        }
        case(3): {
            MOE_SORTING_DISPATCH(3);
        }
        case(5): {
            MOE_SORTING_DISPATCH(5);
        }
        case(6): {
            MOE_SORTING_DISPATCH(6);
        }
        case(8): {
            MOE_SORTING_DISPATCH(8);
        }
        case(10): {
            MOE_SORTING_DISPATCH(10);
        }
        default: {
            MOE_SORTING_DISPATCH(4);
        }
        }
    }
    return -1;
}
