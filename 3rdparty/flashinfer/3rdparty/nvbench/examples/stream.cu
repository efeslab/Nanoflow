/*
 *  Copyright 2022 NVIDIA Corporation
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

#include <nvbench/nvbench.cuh>

// Grab some testing kernels from NVBench:
#include <nvbench/test_kernels.cuh>

// Thrust vectors simplify memory management:
#include <thrust/device_vector.h>

// A function to benchmark but does not expose an explicit stream argument.
void copy(int32_t *input, int32_t *output, std::size_t const num_values)
{
  nvbench::copy_kernel<<<256, 256>>>(input, output, num_values);
}

// `stream_bench` copies a 64 MiB buffer of int32_t on a CUDA stream specified
// by the user.
//
// By default, NVBench creates and provides an explicit stream via
// `launch::get_stream()` to pass to every stream-ordered operation. Sometimes
// it is inconvenient or impossible to specify an explicit CUDA stream to every
// stream-ordered operation. In this case, users may specify a target stream via
// `state::set_cuda_stream`. It is assumed that all work of interest executes on
// or synchronizes with this stream.
void stream_bench(nvbench::state &state)
{
  // Allocate input data:
  const std::size_t num_values = 64 * 1024 * 1024 / sizeof(nvbench::int32_t);
  thrust::device_vector<nvbench::int32_t> input(num_values);
  thrust::device_vector<nvbench::int32_t> output(num_values);

  // Set CUDA default stream as the target stream. Note the default stream
  // is non-owning.
  cudaStream_t default_stream = 0;
  state.set_cuda_stream(nvbench::make_cuda_stream_view(default_stream));

  state.exec([&input, &output, num_values](nvbench::launch &) {
    (void) num_values; // clang thinks this is unused...
    copy(thrust::raw_pointer_cast(input.data()),
         thrust::raw_pointer_cast(output.data()),
         num_values);
  });
}
NVBENCH_BENCH(stream_bench);
