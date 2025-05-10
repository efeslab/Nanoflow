// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#if defined(__gfx908__) || defined(__gfx90a__) || defined(__gfx940__) || defined(__gfx941__) || \
    defined(__gfx942__)
#define __gfx9__
#endif
#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
#define __gfx94__
#endif
#if defined(__gfx1030__) || defined(__gfx1031__) || defined(__gfx1032__) || \
    defined(__gfx1034__) || defined(__gfx1035__) || defined(__gfx1036__)
#define __gfx103__
#endif
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__)
#define __gfx11__
#endif

#ifndef CK_TILE_DONT_USE_HIP_RUNTIME_HEADERS
#include "hip/hip_runtime.h"
#include "hip/hip_fp16.h"
#endif

#ifdef __HIPCC__
#define CK_TILE_HOST inline __host__
#define CK_TILE_DEVICE inline __device__
#define CK_TILE_HOST_DEVICE inline __host__ __device__
#define CK_TILE_DEVICE_EXTERN __device__
#else
#define CK_TILE_HOST inline
#define CK_TILE_DEVICE inline
#define CK_TILE_HOST_DEVICE inline
#define CK_TILE_DEVICE_EXTERN
#endif

#ifndef CK_TILE_USE_CUSTOM_DATA_TYPE
#define CK_TILE_USE_CUSTOM_DATA_TYPE 0 // custom data type will generate extra move/bfi code
#endif

#define CK_TILE_FLOAT_TO_BFLOAT16_STANDARD 0
#define CK_TILE_FLOAT_TO_BFLOAT16_TRUNCATE_WITH_NAN 1
#define CK_TILE_FLOAT_TO_BFLOAT16_TRUNCATE 2

#ifndef CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT
#define CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT CK_TILE_FLOAT_TO_BFLOAT16_TRUNCATE
#endif

#define CK_TILE_FLOAT_TO_FP8_STANDARD 0
#define CK_TILE_FLOAT_TO_FP8_STOCHASTIC 1

#ifndef CK_TILE_FLOAT_TO_FP8_DEFAULT
#define CK_TILE_FLOAT_TO_FP8_DEFAULT CK_TILE_FLOAT_TO_FP8_STANDARD
#endif

// in the old rocm period, we have to use tuple array implementation to implement this
// so turn on the _USE_TUPLE if meet compiler error, otherwise _USE_ARRAY by default.
#define CK_TILE_STATICALLY_INDEXED_ARRAY_USE_ARRAY 0
#define CK_TILE_STATICALLY_INDEXED_ARRAY_USE_TUPLE 1
#ifndef CK_TILE_STATICALLY_INDEXED_ARRAY_DEFAULT
#define CK_TILE_STATICALLY_INDEXED_ARRAY_DEFAULT CK_TILE_STATICALLY_INDEXED_ARRAY_USE_TUPLE
#endif

#define CK_TILE_THREAD_BUFFER_USE_ARRAY 0
#define CK_TILE_THREAD_BUFFER_USE_TUPLE 1
#ifndef CK_TILE_THREAD_BUFFER_DEFAULT
#define CK_TILE_THREAD_BUFFER_DEFAULT CK_TILE_THREAD_BUFFER_USE_ARRAY
#endif

#ifndef CK_TILE_TUPLE_CTOR_WITH_INITIALIZER_LIST
#if CK_TILE_THREAD_BUFFER_DEFAULT == CK_TILE_THREAD_BUFFER_USE_TUPLE
// if using tuple-array as thread_buffer implementation, need to support {} brace init
// ... with similiar behavior as array
#define CK_TILE_TUPLE_CTOR_WITH_INITIALIZER_LIST 1
#else
#define CK_TILE_TUPLE_CTOR_WITH_INITIALIZER_LIST 0
#endif
#endif

#ifndef CK_TILE_USE_LAUNCH_BOUNDS
#define CK_TILE_USE_LAUNCH_BOUNDS 1
#endif

#ifndef CK_TILE_TIME_KERNEL
#define CK_TILE_TIME_KERNEL 1
#endif

#define CK_TILE_MAX_THREAD_PER_BLOCK 256
#define CK_TILE_MIN_BLOCK_PER_CU 2

#ifndef CK_TILE_EXPERIMENTAL_USE_BUFFER_LOAD_OOB_CHECK_OFFSET_TRICK
#define CK_TILE_EXPERIMENTAL_USE_BUFFER_LOAD_OOB_CHECK_OFFSET_TRICK 0
#endif

#ifndef CK_TILE_EXPERIMENTAL_USE_BUFFER_STORE_OOB_CHECK_OFFSET_TRICK
#define CK_TILE_EXPERIMENTAL_USE_BUFFER_STORE_OOB_CHECK_OFFSET_TRICK 1
#endif

#ifndef CK_TILE_EXPERIMENTAL_USE_BUFFER_ATOMIC_ADD_OOB_CHECK_OFFSET_TRICK
#define CK_TILE_EXPERIMENTAL_USE_BUFFER_ATOMIC_ADD_OOB_CHECK_OFFSET_TRICK 1
#endif

#ifndef CK_TILE_EXPERIMENTAL_USE_BUFFER_ATOMIC_MAX_OOB_CHECK_OFFSET_TRICK
#define CK_TILE_EXPERIMENTAL_USE_BUFFER_ATOMIC_MAX_OOB_CHECK_OFFSET_TRICK 1
#endif

#ifndef CK_TILE_USE_AMD_LDS_DIRECT_LOAD_INLINE_ASM
#define CK_TILE_USE_AMD_LDS_DIRECT_LOAD_INLINE_ASM 1
#endif

#ifndef CK_TILE_USE_AMD_BUFFER_LOAD
#define CK_TILE_USE_AMD_BUFFER_LOAD 1
#endif

#ifndef CK_TILE_USE_AMD_BUFFER_STORE
#define CK_TILE_USE_AMD_BUFFER_STORE 1
#endif

#ifndef CK_TILE_USE_AMD_BUFFER_ATOMIC_ADD_INTEGER
#define CK_TILE_USE_AMD_BUFFER_ATOMIC_ADD_INTEGER 1
#endif

// buffer atomic add: floating point
#ifndef __HIP_DEVICE_COMPILE__ // for host code
#define CK_TILE_USE_AMD_BUFFER_ATOMIC_ADD_FLOAT 1
#elif defined(__gfx9__) // for GPU code
#define CK_TILE_USE_AMD_BUFFER_ATOMIC_ADD_FLOAT 1
#else // for GPU code
#define CK_TILE_USE_AMD_BUFFER_ATOMIC_ADD_FLOAT 0
#endif

#if(defined(__gfx90a__) || defined(__gfx94__)) // for GPU code
#define CK_TILE_USE_AMD_BUFFER_ATOMIC_MAX_FLOAT64 1
#else
#define CK_TILE_USE_AMD_BUFFER_ATOMIC_MAX_FLOAT64 0
#endif

#ifndef CK_TILE_EXPERIMENTAL_USE_MEMCPY_FOR_VECTOR_ACCESS
#define CK_TILE_EXPERIMENTAL_USE_MEMCPY_FOR_VECTOR_ACCESS 0
#endif

#ifndef CK_TILE_WORKAROUND_SWDEV_XXXXXX_INT8_DS_WRITE_ISSUE
#define CK_TILE_WORKAROUND_SWDEV_XXXXXX_INT8_DS_WRITE_ISSUE 1
#endif

#ifndef CK_TILE_DEBUG_LOG
#define CK_TILE_DEBUG_LOG 0
#endif

#ifndef __HIP_DEVICE_COMPILE__ // for host code
#define CK_TILE_BUFFER_RESOURCE_3RD_DWORD 0xffffffff
#elif defined(__gfx803__) || defined(__gfx900__) || defined(__gfx906__) || \
    defined(__gfx9__) // for GPU code
#define CK_TILE_BUFFER_RESOURCE_3RD_DWORD 0x00020000
#elif defined(__gfx103__) // for GPU code
#define CK_TILE_BUFFER_RESOURCE_3RD_DWORD 0x31014000
#elif defined(__gfx11__) // for GPU code
#define CK_TILE_BUFFER_RESOURCE_3RD_DWORD 0x31004000
#endif

#ifndef CK_TILE_EXPERIMENTAL_BLOCK_SYNC_LDS_WITHOUT_SYNC_VMEM
#define CK_TILE_EXPERIMENTAL_BLOCK_SYNC_LDS_WITHOUT_SYNC_VMEM 1
#endif

#ifndef CK_TILE_USE_SUBDWORD_TILE_CAST
#define CK_TILE_USE_SUBDWORD_TILE_CAST 0
#endif

// TODO: better solve this inside compiler
#ifndef CK_TILE_FMHA_FWD_FAST_EXP2
#define CK_TILE_FMHA_FWD_FAST_EXP2 0
#endif

#ifndef CK_TILE_BUFFER_LOAD_RAW_BF16_WA
#define CK_TILE_BUFFER_LOAD_RAW_BF16_WA 1
#endif
