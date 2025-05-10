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

#include <nvbench/nvbench.cuh>

// Grab some testing kernels from NVBench:
#include <nvbench/test_kernels.cuh>

// Thrust vectors simplify memory management:
#include <thrust/device_vector.h>

// `throughput_bench` copies a 64 MiB buffer of int32_t, and reports throughput
// in a variety of ways.
//
// Calling `state.add_element_count(num_elements)` with the number of input
// items will report the item throughput rate in elements-per-second.
//
// Calling `state.add_global_memory_reads<T>(num_elements)` and/or
// `state.add_global_memory_writes<T>(num_elements)` will report global device
// memory throughput as a percentage of the current device's peak global memory
// bandwidth, and also in bytes-per-second.
//
// All of these methods take an optional second `column_name` argument, which
// will add a new column to the output with the reported element count / buffer
// size and column name.
void throughput_bench(nvbench::state &state)
{
  // Allocate input data:
  const std::size_t num_values = 64 * 1024 * 1024 / sizeof(nvbench::int32_t);
  thrust::device_vector<nvbench::int32_t> input(num_values);
  thrust::device_vector<nvbench::int32_t> output(num_values);

  // Provide throughput information:
  state.add_element_count(num_values, "NumElements");
  state.add_global_memory_reads<nvbench::int32_t>(num_values, "DataSize");
  state.add_global_memory_writes<nvbench::int32_t>(num_values);

  state.exec([&input, &output, num_values](nvbench::launch &launch) {
    (void) num_values; // clang thinks this is unused...
    nvbench::copy_kernel<<<256, 256, 0, launch.get_stream()>>>(
      thrust::raw_pointer_cast(input.data()),
      thrust::raw_pointer_cast(output.data()),
      num_values);
  });
}
NVBENCH_BENCH(throughput_bench);
