// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_GPU_DATA_TYPES_HPP_
#define MSCCLPP_GPU_DATA_TYPES_HPP_

#if defined(__HIP_PLATFORM_AMD__)

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>

#else

#include <cuda_fp16.h>
#if (CUDART_VERSION >= 11000)
#include <cuda_bf16.h>
#endif
#if (CUDART_VERSION >= 11080)
#include <cuda_fp8.h>
#endif

#endif

#endif  // MSCCLPP_GPU_DATA_TYPES_HPP_
