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

#include <nvbench/cuda_timer.cuh>

#include <nvbench/cuda_stream.cuh>
#include <nvbench/test_kernels.cuh>
#include <nvbench/types.cuh>

#include "test_asserts.cuh"

#include <fmt/format.h>

void test_basic(cudaStream_t time_stream,
                cudaStream_t exec_stream,
                bool expected)
{
  nvbench::cuda_timer timer;

  NVBENCH_CUDA_CALL(cudaDeviceSynchronize());

  timer.start(time_stream);
  nvbench::sleep_kernel<<<1, 1, 0, exec_stream>>>(0.25);
  timer.stop(time_stream);

  NVBENCH_CUDA_CALL(cudaDeviceSynchronize());
  const bool captured = timer.get_duration() > 0.25;
  ASSERT_MSG(captured == expected,
             "Unexpected result from timer: {} seconds (expected {})",
             timer.get_duration(),
             (expected ? "> 0.25s" : "< 0.25s"));
}

void test_basic()
{
  nvbench::cuda_stream stream1;
  nvbench::cuda_stream stream2;

  test_basic(stream1, stream1, true);
  test_basic(stream1, stream2, false);
}

int main() { test_basic(); }
