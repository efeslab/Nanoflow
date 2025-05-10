/*
 *  Copyright 2023 NVIDIA Corporation
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

// Inherit from the stopping_criterion_base class:
class fixed_criterion final : public nvbench::stopping_criterion_base
{
  nvbench::int64_t m_num_samples{};

public:
  fixed_criterion()
      : nvbench::stopping_criterion_base{"fixed", {{"max-samples", nvbench::int64_t{42}}}}
  {}

protected:
  // Setup the criterion in the `do_initialize()` method:
  virtual void do_initialize() override
  {
    m_num_samples = 0;
  }

  // Process new measurements in the `add_measurement()` method:
  virtual void do_add_measurement(nvbench::float64_t /* measurement */) override
  {
    m_num_samples++;
  }

  // Check if the stopping criterion is met in the `is_finished()` method:
  virtual bool do_is_finished() override
  {
    return m_num_samples >= m_params.get_int64("max-samples");
  }

};

// Register the criterion with NVBench:
NVBENCH_REGISTER_CRITERION(fixed_criterion);

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

  state.exec(nvbench::exec_tag::no_batch, [&input, &output, num_values](nvbench::launch &launch) {
    (void) num_values; // clang thinks this is unused...
    nvbench::copy_kernel<<<256, 256, 0, launch.get_stream()>>>(
      thrust::raw_pointer_cast(input.data()),
      thrust::raw_pointer_cast(output.data()),
      num_values);
  });
}
NVBENCH_BENCH(throughput_bench).set_stopping_criterion("fixed");
