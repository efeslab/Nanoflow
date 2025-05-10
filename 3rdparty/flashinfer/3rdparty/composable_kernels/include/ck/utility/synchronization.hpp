// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"

namespace ck {

__device__ void block_sync_lds()
{
#if CK_EXPERIMENTAL_BLOCK_SYNC_LDS_WITHOUT_SYNC_VMEM
    // asm volatile("\
    // s_waitcnt lgkmcnt(0) \n \
    // s_barrier \
    // " ::);
    __builtin_amdgcn_s_waitcnt(0xc07f);
    __builtin_amdgcn_s_barrier();
#else
    __syncthreads();
#endif
}

__device__ void block_sync_lds_direct_load()
{
    asm volatile("\
    s_waitcnt vmcnt(0) \n \
    s_waitcnt lgkmcnt(0) \n \
    s_barrier \
    " ::);
}

__device__ void s_nop()
{
#if 1
    asm volatile("\
    s_nop 0 \n \
    " ::);
#else
    __builtin_amdgcn_sched_barrier(0);
#endif
}

} // namespace ck
