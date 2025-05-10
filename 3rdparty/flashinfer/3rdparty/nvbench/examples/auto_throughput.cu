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

// Thrust vectors simplify memory management:
#include <thrust/device_vector.h>

template <int ItemsPerThread>
__global__ void kernel(std::size_t stride,
                       std::size_t elements,
                       const nvbench::int32_t * __restrict__ in,
                       nvbench::int32_t *__restrict__ out)
{
  const std::size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  const std::size_t step = gridDim.x * blockDim.x;

  for (std::size_t i = stride * tid;
       i < stride * elements;
       i += stride * step)
  {
    for (int j = 0; j < ItemsPerThread; j++)
    {
      const auto read_id = (ItemsPerThread * i + j) % elements;
      const auto write_id = tid + j * elements;
      out[write_id] = in[read_id];
    }
  }
}


// `throughput_bench` copies a 128 MiB buffer of int32_t, and reports throughput
// and cache hit rates.
//
// Calling state.collect_*() enables particular metric collection if nvbench
// was build with CUPTI support (CMake option: -DNVBench_ENABLE_CUPTI=ON).
template <int ItemsPerThread>
void throughput_bench(nvbench::state &state,
                      nvbench::type_list<nvbench::enum_type<ItemsPerThread>>)
{
  // Allocate input data:
  const std::size_t stride = static_cast<std::size_t>(state.get_int64("Stride"));
  const std::size_t elements = 128 * 1024 * 1024 / sizeof(nvbench::int32_t);
  thrust::device_vector<nvbench::int32_t> input(elements);
  thrust::device_vector<nvbench::int32_t> output(elements * ItemsPerThread);

  // Provide throughput information:
  state.add_element_count(elements, "Elements");
  state.collect_dram_throughput();
  state.collect_l1_hit_rates();
  state.collect_l2_hit_rates();
  state.collect_loads_efficiency();
  state.collect_stores_efficiency();

  const auto threads_in_block = 256;
  const auto blocks_in_grid =
    static_cast<int>((elements + threads_in_block - 1) / threads_in_block);

  state.exec([&](nvbench::launch &launch) {
    kernel<ItemsPerThread>
      <<<blocks_in_grid, threads_in_block, 0, launch.get_stream()>>>(
        stride,
        elements,
        thrust::raw_pointer_cast(input.data()),
        thrust::raw_pointer_cast(output.data()));
  });
}

using items_per_thread = nvbench::enum_type_list<1, 2>;

NVBENCH_BENCH_TYPES(throughput_bench, NVBENCH_TYPE_AXES(items_per_thread))
  .add_int64_axis("Stride", nvbench::range(1, 4, 3));
