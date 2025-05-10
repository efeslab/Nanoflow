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

// std::enable_if_t
#include <type_traits>

//==============================================================================
// `runtime_skip` demonstrates how to skip benchmarks at runtime.
//
// Two parameter axes are swept (see axes.cu), but some configurations are
// skipped by calling `state.skip` with a skip reason string. This reason
// is printed to the log and captured in JSON output.
void runtime_skip(nvbench::state &state)
{
  const auto duration = state.get_float64("Duration");
  const auto kramble  = state.get_string("Kramble");

  // Skip Baz benchmarks with < 0.8 ms duration.
  if (kramble == "Baz" && duration < 0.8e-3)
  {
    state.skip("Short 'Baz' benchmarks are skipped.");
    return;
  }

  // Skip Foo benchmarks with > 0.3 ms duration.
  if (kramble == "Foo" && duration > 0.3e-3)
  {
    state.skip("Long 'Foo' benchmarks are skipped.");
    return;
  }

  // Run all others:
  state.exec([duration](nvbench::launch &launch) {
    nvbench::sleep_kernel<<<1, 1, 0, launch.get_stream()>>>(duration);
  });
}
NVBENCH_BENCH(runtime_skip)
  // 0, 0.25, 0.5, 0.75, and 1.0 milliseconds
  .add_float64_axis("Duration",
                    nvbench::range(0.,
                                   1.1e-3, // .1e-3 slop for fp precision
                                   0.25e-3))
  .add_string_axis("Kramble", {"Foo", "Bar", "Baz"});

//==============================================================================
// `skip_overload` demonstrates how to skip benchmarks at compile-time via
// overload resolution.
//
// Two type axes are swept, but configurations where InputType == OutputType are
// skipped.
template <typename InputType, typename OutputType>
void skip_overload(nvbench::state &state,
                   nvbench::type_list<InputType, OutputType>)
{
  // This is a contrived example that focuses on the skip overloads, so this is
  // just a sleep kernel:
  state.exec([](nvbench::launch &launch) {
    nvbench::sleep_kernel<<<1, 1, 0, launch.get_stream()>>>(1e-3);
  });
}
// Overload of skip_overload that is called when InputType == OutputType.
template <typename T>
void skip_overload(nvbench::state &state, nvbench::type_list<T, T>)
{
  state.skip("InputType == OutputType.");
}
// The same type_list is used for both inputs/outputs.
using sst_types = nvbench::type_list<nvbench::int32_t, nvbench::int64_t>;
// Setup benchmark:
NVBENCH_BENCH_TYPES(skip_overload, NVBENCH_TYPE_AXES(sst_types, sst_types))
  .set_type_axes_names({"In", "Out"});

//==============================================================================
// `skip_sfinae` demonstrates how to skip benchmarks at compile-time using
// SFINAE to handle more complex skip conditions.
//
// Two type axes are swept, but configurations where sizeof(InputType) >
// sizeof(OutputType) are skipped.

// Enable this overload if InputType is not larger than OutputType
template <typename InputType, typename OutputType>
std::enable_if_t<(sizeof(InputType) <= sizeof(OutputType)), void>
skip_sfinae(nvbench::state &state, nvbench::type_list<InputType, OutputType>)
{
  // This is a contrived example that focuses on the skip overloads, so this is
  // just a sleep kernel:
  state.exec([](nvbench::launch &launch) {
    nvbench::sleep_kernel<<<1, 1, 0, launch.get_stream()>>>(1e-3);
  });
}
// Enable this overload if InputType is larger than OutputType
template <typename InputType, typename OutputType>
std::enable_if_t<(sizeof(InputType) > sizeof(OutputType)), void>
skip_sfinae(nvbench::state &state, nvbench::type_list<InputType, OutputType>)
{
  state.skip("sizeof(InputType) > sizeof(OutputType).");
}
// The same type_list is used for both inputs/outputs.
using sn_types = nvbench::type_list<nvbench::int8_t,
                                    nvbench::int16_t,
                                    nvbench::int32_t,
                                    nvbench::int64_t>;
// Setup benchmark:
NVBENCH_BENCH_TYPES(skip_sfinae, NVBENCH_TYPE_AXES(sn_types, sn_types))
  .set_type_axes_names({"In", "Out"});
