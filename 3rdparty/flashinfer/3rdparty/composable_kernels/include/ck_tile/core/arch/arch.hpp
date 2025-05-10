// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

// Address Space for AMDGCN
// https://llvm.org/docs/AMDGPUUsage.html#address-space

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/integral_constant.hpp"

namespace ck_tile {

enum struct address_space_enum
{
    generic,
    global,
    lds,
    sgpr,
    vgpr,
};

enum struct memory_operation_enum
{
    set,
    atomic_add,
    atomic_max,
    add
};

CK_TILE_HOST_DEVICE constexpr index_t get_warp_size()
{
    // warpSize is defined by HIP
    return warpSize;
}

CK_TILE_DEVICE index_t get_grid_size() { return gridDim.x; }

CK_TILE_DEVICE index_t get_block_size() { return blockDim.x; }

// TODO: deprecate these
CK_TILE_DEVICE index_t get_thread_local_1d_id() { return threadIdx.x; }

CK_TILE_DEVICE index_t get_thread_global_1d_id() { return blockIdx.x * blockDim.x + threadIdx.x; }

CK_TILE_DEVICE index_t get_block_1d_id() { return blockIdx.x; }

// Use these instead
CK_TILE_DEVICE index_t get_lane_id() { return __lane_id(); }

CK_TILE_DEVICE index_t get_warp_id()
{
    return __builtin_amdgcn_readfirstlane(threadIdx.x / get_warp_size());
}

CK_TILE_DEVICE index_t get_thread_id() { return threadIdx.x; }

CK_TILE_DEVICE index_t get_block_id() { return blockIdx.x; }

CK_TILE_DEVICE void block_sync_lds()
{
#if CK_TILE_EXPERIMENTAL_BLOCK_SYNC_LDS_WITHOUT_SYNC_VMEM
    asm volatile("\
    s_waitcnt lgkmcnt(0) \n \
    s_barrier \
    " ::);
#else
    __syncthreads();
#endif
}

CK_TILE_DEVICE void block_sync_lds_direct_load()
{
    asm volatile("\
    s_waitcnt vmcnt(0) \n \
    s_waitcnt lgkmcnt(0) \n \
    s_barrier \
    " ::);
}

CK_TILE_DEVICE void s_nop()
{
#if 1
    asm volatile("\
    s_nop 0 \n \
    " ::);
#else
    __builtin_amdgcn_sched_barrier(0);
#endif
}

} // namespace ck_tile
