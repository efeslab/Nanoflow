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

#include <nvbench/cuda_call.cuh>

#include <nvbench/types.cuh>

#include <cuda_runtime_api.h>

namespace nvbench
{

struct cuda_timer
{
  __forceinline__ cuda_timer()
  {
    NVBENCH_CUDA_CALL(cudaEventCreate(&m_start));
    NVBENCH_CUDA_CALL(cudaEventCreate(&m_stop));
  }

  __forceinline__ ~cuda_timer()
  {
    NVBENCH_CUDA_CALL(cudaEventDestroy(m_start));
    NVBENCH_CUDA_CALL(cudaEventDestroy(m_stop));
  }

  // move-only
  cuda_timer(const cuda_timer &)            = delete;
  cuda_timer(cuda_timer &&)                 = default;
  cuda_timer &operator=(const cuda_timer &) = delete;
  cuda_timer &operator=(cuda_timer &&)      = default;

  __forceinline__ void start(cudaStream_t stream)
  {
    NVBENCH_CUDA_CALL(cudaEventRecord(m_start, stream));
  }

  __forceinline__ void stop(cudaStream_t stream)
  {
    NVBENCH_CUDA_CALL(cudaEventRecord(m_stop, stream));
  }

  [[nodiscard]] __forceinline__ bool ready() const
  {
    const cudaError_t state = cudaEventQuery(m_stop);
    if (state == cudaErrorNotReady)
    {
      return false;
    }
    NVBENCH_CUDA_CALL(state);
    return true;
  }

  // In seconds:
  [[nodiscard]] __forceinline__ nvbench::float64_t get_duration() const
  {
    NVBENCH_CUDA_CALL(cudaEventSynchronize(m_stop));
    float elapsed_time;
    // According to docs, this is in ms with a resolution of ~0.5 microseconds.
    NVBENCH_CUDA_CALL(cudaEventElapsedTime(&elapsed_time, m_start, m_stop));
    return elapsed_time / 1000.0;
  }

private:
  cudaEvent_t m_start;
  cudaEvent_t m_stop;
};

} // namespace nvbench
