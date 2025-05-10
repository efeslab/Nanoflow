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

//==============================================================================
// Simple benchmark with no parameter axes:
void simple(nvbench::state &state)
{
  state.exec([](nvbench::launch &launch) {
    // Sleep for 1 millisecond:
    nvbench::sleep_kernel<<<1, 1, 0, launch.get_stream()>>>(1e-3);
  });
}
NVBENCH_BENCH(simple);

//==============================================================================
// Single parameter sweep:
void single_float64_axis(nvbench::state &state)
{
  const auto duration = state.get_float64("Duration");

  state.exec([duration](nvbench::launch &launch) {
    nvbench::sleep_kernel<<<1, 1, 0, launch.get_stream()>>>(duration);
  });
}
NVBENCH_BENCH(single_float64_axis)
  // 0 -> 1 ms in 100 us increments.
  .add_float64_axis("Duration", nvbench::range(0., 1e-3, 1e-4));

//==============================================================================
// Multiple parameters:
// Varies block_size and num_blocks while invoking a naive copy of 256 MiB worth
// of int32_t.
void copy_sweep_grid_shape(nvbench::state &state)
{
  // Get current parameters:
  const auto block_size = static_cast<unsigned int>(state.get_int64("BlockSize"));
  const auto num_blocks = static_cast<unsigned int>(state.get_int64("NumBlocks"));

  // Number of int32s in 256 MiB:
  const std::size_t num_values = 256 * 1024 * 1024 / sizeof(nvbench::int32_t);

  // Report throughput stats:
  state.add_element_count(num_values);
  state.add_global_memory_reads<nvbench::int32_t>(num_values);
  state.add_global_memory_writes<nvbench::int32_t>(num_values);

  // Allocate device memory:
  thrust::device_vector<nvbench::int32_t> in(num_values, 0);
  thrust::device_vector<nvbench::int32_t> out(num_values, 0);

  state.exec(
    [block_size,
     num_blocks,
     num_values,
     in_ptr  = thrust::raw_pointer_cast(in.data()),
     out_ptr = thrust::raw_pointer_cast(out.data())](nvbench::launch &launch) {
      (void) num_values; // clang thinks this is unused...
      nvbench::copy_kernel<<<num_blocks, block_size, 0, launch.get_stream()>>>(
        in_ptr,
        out_ptr,
        num_values);
    });
}
NVBENCH_BENCH(copy_sweep_grid_shape)
  // Every second power of two from  64->1024:
  .add_int64_power_of_two_axis("BlockSize", nvbench::range(6, 10, 2))
  .add_int64_power_of_two_axis("NumBlocks", nvbench::range(6, 10, 2));

//==============================================================================
// Type parameter sweep:
// Copy 256 MiB of data, represented with various value_types.
template <typename ValueType>
void copy_type_sweep(nvbench::state &state, nvbench::type_list<ValueType>)
{
  // Number of ValueTypes in 256 MiB:
  const std::size_t num_values = 256 * 1024 * 1024 / sizeof(ValueType);

  // Report throughput stats:
  state.add_element_count(num_values);
  state.add_global_memory_reads<ValueType>(num_values);
  state.add_global_memory_writes<ValueType>(num_values);

  // Allocate device memory:
  thrust::device_vector<ValueType> in(num_values, 0);
  thrust::device_vector<ValueType> out(num_values, 0);

  state.exec(
    [num_values,
     in_ptr  = thrust::raw_pointer_cast(in.data()),
     out_ptr = thrust::raw_pointer_cast(out.data())](nvbench::launch &launch) {
      (void) num_values; // clang thinks this is unused...
      nvbench::copy_kernel<<<256, 256, 0, launch.get_stream()>>>(in_ptr,
                                                                 out_ptr,
                                                                 num_values);
    });
}
// Define a type_list to use for the type axis:
using cts_types = nvbench::type_list<nvbench::uint8_t,
                                     nvbench::uint16_t,
                                     nvbench::uint32_t,
                                     nvbench::uint64_t,
                                     nvbench::float32_t,
                                     nvbench::float64_t>;
NVBENCH_BENCH_TYPES(copy_type_sweep, NVBENCH_TYPE_AXES(cts_types));

//==============================================================================
// Type parameter sweep:
// Convert 64 MiB of InputTypes to OutputTypes, represented with various
// value_types.
template <typename InputType, typename OutputType>
void copy_type_conversion_sweep(nvbench::state &state,
                                nvbench::type_list<InputType, OutputType>)
{
  // Optional: Skip narrowing conversions.
  if constexpr(sizeof(InputType) > sizeof(OutputType))
  {
    state.skip("Narrowing conversion: sizeof(InputType) > sizeof(OutputType).");
    return;
  }

  // Number of InputTypes in 64 MiB:
  const std::size_t num_values = 64 * 1024 * 1024 / sizeof(InputType);

  // Report throughput stats: Passing an optional string adds a column to the
  // output with the number of items/bytes.
  state.add_element_count(num_values, "Items");
  state.add_global_memory_reads<InputType>(num_values, "InSize");
  state.add_global_memory_writes<OutputType>(num_values, "OutSize");

  // Allocate device memory:
  thrust::device_vector<InputType> in(num_values, 0);
  thrust::device_vector<OutputType> out(num_values, 0);

  state.exec(
    [num_values,
     in_ptr  = thrust::raw_pointer_cast(in.data()),
     out_ptr = thrust::raw_pointer_cast(out.data())](nvbench::launch &launch) {
      (void) num_values; // clang thinks this is unused...
      nvbench::copy_kernel<<<256, 256, 0, launch.get_stream()>>>(in_ptr,
                                                                 out_ptr,
                                                                 num_values);
    });
}
// Optional: Skip when InputType == OutputType. This approach avoids
// instantiating the benchmark at all.
template <typename T>
void copy_type_conversion_sweep(nvbench::state &state, nvbench::type_list<T, T>)
{
  state.skip("Not a conversion: InputType == OutputType.");
}
// The same type_list is used for both inputs/outputs.
using ctcs_types = nvbench::type_list<nvbench::int8_t,
                                      nvbench::int16_t,
                                      nvbench::int32_t,
                                      nvbench::float32_t,
                                      nvbench::int64_t,
                                      nvbench::float64_t>;
NVBENCH_BENCH_TYPES(copy_type_conversion_sweep,
                    NVBENCH_TYPE_AXES(ctcs_types, ctcs_types))
  .set_type_axes_names({"In", "Out"});
