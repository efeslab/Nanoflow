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

#include <nvbench/test_kernels.cuh>

// Enum to use as parameter axis:
enum class MyEnum
{
  ValueA,
  ValueB,
  ValueC
};

//==============================================================================
// Sweep through enum values at runtime using a string axis.
//
// Preferred way to provide access to an enum value that doesn't need to be
// compile-time constant. Create a string axis with unique values for each
// enum value of interest, and convert the string to an enum value in the
// benchmark.
//
// This approach is preferred since it gives nicer output (readable names
// instead of integral enum values), but takes a bit of extra work to convert
// the strings back to an enum value.
//
// `--list` output:
// ```
// * `MyEnum` : string
//   * `A`
//   * `B`
//   * `C`
// ```
void runtime_enum_sweep_string(nvbench::state &state)
{
  const auto enum_string = state.get_string("MyEnum");
  [[maybe_unused]] MyEnum enum_value{};
  if (enum_string == "A")
  {
    enum_value = MyEnum::ValueA;
  }
  else if (enum_string == "B")
  {
    enum_value = MyEnum::ValueB;
  }
  else if (enum_string == "C")
  {
    enum_value = MyEnum::ValueC;
  }

  // Do stuff with enum_value.
  // Create inputs, etc, configure runtime kernel parameters, etc.

  // Just a dummy kernel.
  state.exec([](nvbench::launch &launch) {
    nvbench::sleep_kernel<<<1, 1, 0, launch.get_stream()>>>(1e-3);
  });
}
NVBENCH_BENCH(runtime_enum_sweep_string)
  .add_string_axis("MyEnum", {"A", "B", "C"});

//==============================================================================
// Sweep through enum values at runtime using an int64 axis.
//
// This may be useful for doing quick tests / prototyping, but does not provide
// readable output / command-line args since numeric values will be used for the
// axis.
//
// `--list` output:
// ```
// * `MyEnum` : int64
//   * `0`
//   * `1`
//   * `2`
// ```
void runtime_enum_sweep_int64(nvbench::state &state)
{
  [[maybe_unused]] const auto enum_value = static_cast<MyEnum>(state.get_int64("MyEnum"));

  // Do stuff with enum_value.
  // Create inputs, etc, configure runtime kernel parameters, etc.

  // Just a dummy kernel.
  state.exec([](nvbench::launch &launch) {
    nvbench::sleep_kernel<<<1, 1, 0, launch.get_stream()>>>(1e-3);
  });
}
NVBENCH_BENCH(runtime_enum_sweep_int64)
  .add_int64_axis("MyEnum",
                  {static_cast<nvbench::int64_t>(MyEnum::ValueA),
                   static_cast<nvbench::int64_t>(MyEnum::ValueB),
                   static_cast<nvbench::int64_t>(MyEnum::ValueC)});

//==============================================================================
// Sweep through enum values at compile time using an `enum_type_list`.
//
// If an enum value needs to be available at compile time (for example, if it's
// used as a template parameter), the `nvbench::enum_type_list` helper can be
// used to create a type axis of `nvbench::enum_type<Value>`s.
//
// The `NVBENCH_DECLARE_ENUM_TYPE_STRINGS(T, InputGenerator, DescGenerator)`
// utility configures an `nvbench::type_strings<nvbench::enum_type<...>>`
// specialization for the integral constants, improving readability of
// input/output, as shown below.
//
// `--list` output:
// ```
// * `MyEnum` : type
//   * `A` (MyEnum::ValueA)
//   * `B` (MyEnum::ValueB)
//   * `C` (MyEnum::ValueC)
// ```

// Optional:
// Tell NVBench how to turn your enum into strings for display, commandline
// args, `--list` output, tables, etc.
// Reasonable defaults will be used if omitted.
NVBENCH_DECLARE_ENUM_TYPE_STRINGS(
  // Enum type:
  MyEnum,
  // Callable to generate input strings:
  // Short identifier used for tables, command-line args, etc.
  // Used when context is available to figure out the enum type.
  [](MyEnum value) {
    switch (value)
    {
      case MyEnum::ValueA:
        return "A";
      case MyEnum::ValueB:
        return "B";
      case MyEnum::ValueC:
        return "C";
      default:
        return "Unknown";
    }
  },
  // Callable to generate descriptions:
  // If non-empty, these are used in `--list` to describe values.
  // Used when context may not be available to figure out the type from the
  // input string.
  // Just use `[](auto) { return std::string{}; }` if you don't want these.
  [](MyEnum value) {
    switch (value)
    {
      case MyEnum::ValueA:
        return "MyEnum::ValueA";
      case MyEnum::ValueB:
        return "MyEnum::ValueB";
      case MyEnum::ValueC:
        return "MyEnum::ValueC";
      default:
        return "Unknown";
    }
  })

// The actual compile-time enum sweep benchmark:
template <MyEnum EnumValue>
void compile_time_enum_sweep(nvbench::state &state,
                             nvbench::type_list<nvbench::enum_type<EnumValue>>)
{
  // Use EnumValue in compile-time contexts.
  // Template parameters, static dispatch, etc.

  // Just a dummy kernel.
  state.exec([](nvbench::launch &launch) {
    nvbench::sleep_kernel<<<1, 1, 0, launch.get_stream()>>>(1e-3);
  });
}
using MyEnumList =
  nvbench::enum_type_list<MyEnum::ValueA, MyEnum::ValueB, MyEnum::ValueC>;
NVBENCH_BENCH_TYPES(compile_time_enum_sweep, NVBENCH_TYPE_AXES(MyEnumList))
  .set_type_axes_names({"MyEnum"});

//==============================================================================
// `enum_type_list` works for other integral types, too.
//
// `--list` output:
// ```
// * `SomeInts` : type
//  * `0` (struct std::integral_constant<int,0>)
//  * `16` (struct std::integral_constant<int,16>)
//  * `4096` (struct std::integral_constant<int,4096>)
//  * `-12` (struct std::integral_constant<int,-12>)
// ```
template <nvbench::int32_t IntValue>
void compile_time_int_sweep(nvbench::state &state,
                            nvbench::type_list<nvbench::enum_type<IntValue>>)
{
  // Use IntValue in compile time contexts.
  // Template parameters, static dispatch, etc.

  // Just a dummy kernel.
  state.exec([](nvbench::launch &launch) {
    nvbench::sleep_kernel<<<1, 1, 0, launch.get_stream()>>>(1e-3);
  });
}
using MyInts = nvbench::enum_type_list<0, 16, 4096, -12>;
NVBENCH_BENCH_TYPES(compile_time_int_sweep, NVBENCH_TYPE_AXES(MyInts))
  .set_type_axes_names({"SomeInts"});
