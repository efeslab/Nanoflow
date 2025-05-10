/*
 *  Copyright 2021 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 with the LLVM exception
 *  (the "License"); you may not use this file except in compliance with
 *  the License.
 *
 *  You may obtain a copy of the License at
 *
 *      http://llvm.org/foundation/relicensing/LICENSE.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

#include <nvbench/types.cuh>

#include <cuda/std/chrono>

#include <cuda_runtime.h>

/*!
 * @file test_kernels.cuh
 * A collection of simple kernels for testing purposes.
 *
 * Note that these kernels are written to be short and simple, not performant.
 */

namespace nvbench
{

/*!
 * Each launched thread just sleeps for `seconds`.
 */
__global__ void sleep_kernel(double seconds)
{
  const auto start = cuda::std::chrono::high_resolution_clock::now();
  const auto ns =
    cuda::std::chrono::nanoseconds(static_cast<nvbench::int64_t>(seconds * 1000 * 1000 * 1000));
  const auto finish = start + ns;

  auto now = cuda::std::chrono::high_resolution_clock::now();
  while (now < finish)
  {
    now = cuda::std::chrono::high_resolution_clock::now();
  }
}

/*!
 * Naive copy of `n` values from `in` -> `out`.
 */
template <typename T, typename U>
__global__ void copy_kernel(const T *in, U *out, std::size_t n)
{
  const auto init = blockIdx.x * blockDim.x + threadIdx.x;
  const auto step = blockDim.x * gridDim.x;

  for (auto i = init; i < n; i += step)
  {
    out[i] = static_cast<U>(in[i]);
  }
}

/*!
 * For `i <- [0,n)`, `out[i] = in[i] % 2`.
 */
template <typename T, typename U>
__global__ void mod2_kernel(const T *in, U *out, std::size_t n)
{
  const auto init = blockIdx.x * blockDim.x + threadIdx.x;
  const auto step = blockDim.x * gridDim.x;

  for (auto i = init; i < n; i += step)
  {
    out[i] = static_cast<U>(in[i] % 2);
  }
}

} // namespace nvbench
