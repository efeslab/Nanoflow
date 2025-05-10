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

// Used to initialize input data:
#include <thrust/sequence.h>

// Used to run the benchmark on a CUDA stream
#include <thrust/execution_policy.h>

// `sequence_bench` measures the execution time of `thrust::sequence`. Since
// algorithms in `thrust::` implicitly sync the CUDA device, the
// `nvbench::exec_tag::sync` must be passed to `state.exec(...)`.
//
// By default, NVBench uses some tricks to improve the GPU timing stability.
// This provides more accurate results, but will cause a deadlock if the lambda
// passed to `state.exec(...)` synchronizes. The `nvbench::exec_tag::sync` tag
// tells NVBench to run the benchmark safely.
//
// This tag will also disable the batch measurements, since the synchronization
// will throw off the batch results.
void sequence_bench(nvbench::state &state)
{
  // Allocate input data:
  const std::size_t num_values = 64 * 1024 * 1024 / sizeof(nvbench::int32_t);
  thrust::device_vector<nvbench::int32_t> data(num_values);

  // Provide throughput information:
  state.add_element_count(num_values, "Items");
  state.add_global_memory_writes<nvbench::int32_t>(num_values, "Size");

  // nvbench::exec_tag::sync indicates that this will implicitly sync:
  state.exec(nvbench::exec_tag::sync, [&data](nvbench::launch &launch) {
    thrust::sequence(thrust::device.on(launch.get_stream()),
                     data.begin(),
                     data.end());
  });
}
NVBENCH_BENCH(sequence_bench);
