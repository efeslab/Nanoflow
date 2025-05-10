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

#include <nvbench/runner.cuh>

#include <nvbench/benchmark.cuh>
#include <nvbench/callable.cuh>
#include <nvbench/state.cuh>
#include <nvbench/type_list.cuh>
#include <nvbench/type_strings.cuh>
#include <nvbench/types.cuh>

#include "test_asserts.cuh"

#include <fmt/format.h>

#include <algorithm>
#include <variant>
#include <vector>

template <typename T>
std::vector<T> sort(std::vector<T> &&vec)
{
  std::sort(vec.begin(), vec.end());
  return std::move(vec);
}

void no_op_generator(nvbench::state &state)
{
  fmt::memory_buffer params;
  fmt::format_to(std::back_inserter(params), "Params:");
  const auto &axis_values = state.get_axis_values();
  for (const auto &name : sort(axis_values.get_names()))
  {
    std::visit(
      [&params, &name](const auto &value) {
        fmt::format_to(std::back_inserter(params), " {}: {}", name, value);
      },
      axis_values.get_value(name));
  }

  // Marking as skipped to signal that this state is run:
  state.skip(fmt::to_string(std::move(params)));
}
NVBENCH_DEFINE_CALLABLE(no_op_generator, no_op_callable);

using float_types = nvbench::type_list<nvbench::float32_t, nvbench::float64_t>;
using int_types   = nvbench::type_list<nvbench::int32_t, nvbench::int64_t>;
using misc_types  = nvbench::type_list<bool, void>;
using type_axes   = nvbench::type_list<float_types, int_types, misc_types>;

template <typename FloatT, typename IntT, typename MiscT>
void template_no_op_generator(nvbench::state &state,
                              nvbench::type_list<FloatT, IntT, MiscT>)
{
  ASSERT(nvbench::type_strings<FloatT>::input_string() ==
         state.get_string("FloatT"));
  ASSERT(nvbench::type_strings<IntT>::input_string() ==
         state.get_string("IntT"));
  ASSERT(nvbench::type_strings<IntT>::input_string() ==
         state.get_string("IntT"));

  // Enum params using non-templated version:
  no_op_generator(state);
}
NVBENCH_DEFINE_CALLABLE_TEMPLATE(template_no_op_generator,
                                 template_no_op_callable);

void test_empty()
{
  using benchmark_type = nvbench::benchmark<no_op_callable>;
  using runner_type    = nvbench::runner<benchmark_type>;

  benchmark_type bench;
  bench.set_devices(std::vector<int>{});
  runner_type runner{bench};

  runner.generate_states();
  ASSERT(bench.get_states().size() == 1);
  ASSERT(bench.get_states().front().is_skipped() == false);
  runner.run();
  ASSERT(bench.get_states().size() == 1);
  ASSERT(bench.get_states().front().is_skipped() == true);
}

void test_non_types()
{
  using benchmark_type = nvbench::benchmark<no_op_callable>;
  using runner_type    = nvbench::runner<benchmark_type>;

  benchmark_type bench;
  bench.set_devices(std::vector<int>{});
  bench.add_int64_axis("Int", {1, 2, 3});
  bench.add_float64_axis("Float", {11.0, 12.0, 13.0});
  bench.add_string_axis("String", {"One", "Two", "Three"});

  runner_type runner{bench};

  runner.generate_states();
  ASSERT(bench.get_states().size() == 27);
  for (const auto &state : bench.get_states())
  {
    ASSERT(state.is_skipped() == false);
  }

  fmt::memory_buffer buffer;
  runner.run();
  ASSERT(bench.get_states().size() == 27);
  for (const auto &state : bench.get_states())
  {
    ASSERT(state.is_skipped() == true);
    fmt::format_to(std::back_inserter(buffer), "{}\n", state.get_skip_reason());
  }

  const std::string ref = R"expected(Params: Float: 11 Int: 1 String: One
Params: Float: 11 Int: 2 String: One
Params: Float: 11 Int: 3 String: One
Params: Float: 12 Int: 1 String: One
Params: Float: 12 Int: 2 String: One
Params: Float: 12 Int: 3 String: One
Params: Float: 13 Int: 1 String: One
Params: Float: 13 Int: 2 String: One
Params: Float: 13 Int: 3 String: One
Params: Float: 11 Int: 1 String: Two
Params: Float: 11 Int: 2 String: Two
Params: Float: 11 Int: 3 String: Two
Params: Float: 12 Int: 1 String: Two
Params: Float: 12 Int: 2 String: Two
Params: Float: 12 Int: 3 String: Two
Params: Float: 13 Int: 1 String: Two
Params: Float: 13 Int: 2 String: Two
Params: Float: 13 Int: 3 String: Two
Params: Float: 11 Int: 1 String: Three
Params: Float: 11 Int: 2 String: Three
Params: Float: 11 Int: 3 String: Three
Params: Float: 12 Int: 1 String: Three
Params: Float: 12 Int: 2 String: Three
Params: Float: 12 Int: 3 String: Three
Params: Float: 13 Int: 1 String: Three
Params: Float: 13 Int: 2 String: Three
Params: Float: 13 Int: 3 String: Three
)expected";

  const std::string test = fmt::to_string(buffer);
  ASSERT_MSG(test == ref, "Expected:\n\"{}\"\n\nActual:\n\"{}\"", ref, test);
}

void test_types()
{
  using benchmark_type = nvbench::benchmark<template_no_op_callable, type_axes>;
  using runner_type    = nvbench::runner<benchmark_type>;

  benchmark_type bench;
  bench.set_devices(std::vector<int>{});
  bench.set_type_axes_names({"FloatT", "IntT", "MiscT"});

  runner_type runner{bench};

  runner.generate_states();
  ASSERT(bench.get_states().size() == 8);
  for (const auto &state : bench.get_states())
  {
    ASSERT(state.is_skipped() == false);
  }

  fmt::memory_buffer buffer;
  runner.run();
  ASSERT(bench.get_states().size() == 8);
  for (const auto &state : bench.get_states())
  {
    ASSERT(state.is_skipped() == true);
    fmt::format_to(std::back_inserter(buffer), "{}\n", state.get_skip_reason());
  }

  const std::string ref = R"expected(Params: FloatT: F32 IntT: I32 MiscT: bool
Params: FloatT: F32 IntT: I32 MiscT: void
Params: FloatT: F32 IntT: I64 MiscT: bool
Params: FloatT: F32 IntT: I64 MiscT: void
Params: FloatT: F64 IntT: I32 MiscT: bool
Params: FloatT: F64 IntT: I32 MiscT: void
Params: FloatT: F64 IntT: I64 MiscT: bool
Params: FloatT: F64 IntT: I64 MiscT: void
)expected";

  const std::string test = fmt::to_string(buffer);
  ASSERT_MSG(test == ref, "Expected:\n\"{}\"\n\nActual:\n\"{}\"", ref, test);
}

void test_both()
{
  using benchmark_type = nvbench::benchmark<template_no_op_callable, type_axes>;
  using runner_type    = nvbench::runner<benchmark_type>;

  benchmark_type bench;
  bench.set_devices(std::vector<int>{});
  bench.set_type_axes_names({"FloatT", "IntT", "MiscT"});
  bench.add_int64_axis("Int", {1, 2, 3});
  bench.add_float64_axis("Float", {11.0, 12.0, 13.0});
  bench.add_string_axis("String", {"One", "Two", "Three"});

  runner_type runner{bench};

  runner.generate_states();
  ASSERT(bench.get_states().size() == 8 * 27);
  for (const auto &state : bench.get_states())
  {
    ASSERT(state.is_skipped() == false);
  }

  fmt::memory_buffer buffer;
  runner.run();
  ASSERT(bench.get_states().size() == 8 * 27);
  for (const auto &state : bench.get_states())
  {
    ASSERT(state.is_skipped() == true);
    fmt::format_to(std::back_inserter(buffer), "{}\n", state.get_skip_reason());
  }

  const std::string ref =
    R"expected(Params: Float: 11 FloatT: F32 Int: 1 IntT: I32 MiscT: bool String: One
Params: Float: 11 FloatT: F32 Int: 2 IntT: I32 MiscT: bool String: One
Params: Float: 11 FloatT: F32 Int: 3 IntT: I32 MiscT: bool String: One
Params: Float: 12 FloatT: F32 Int: 1 IntT: I32 MiscT: bool String: One
Params: Float: 12 FloatT: F32 Int: 2 IntT: I32 MiscT: bool String: One
Params: Float: 12 FloatT: F32 Int: 3 IntT: I32 MiscT: bool String: One
Params: Float: 13 FloatT: F32 Int: 1 IntT: I32 MiscT: bool String: One
Params: Float: 13 FloatT: F32 Int: 2 IntT: I32 MiscT: bool String: One
Params: Float: 13 FloatT: F32 Int: 3 IntT: I32 MiscT: bool String: One
Params: Float: 11 FloatT: F32 Int: 1 IntT: I32 MiscT: bool String: Two
Params: Float: 11 FloatT: F32 Int: 2 IntT: I32 MiscT: bool String: Two
Params: Float: 11 FloatT: F32 Int: 3 IntT: I32 MiscT: bool String: Two
Params: Float: 12 FloatT: F32 Int: 1 IntT: I32 MiscT: bool String: Two
Params: Float: 12 FloatT: F32 Int: 2 IntT: I32 MiscT: bool String: Two
Params: Float: 12 FloatT: F32 Int: 3 IntT: I32 MiscT: bool String: Two
Params: Float: 13 FloatT: F32 Int: 1 IntT: I32 MiscT: bool String: Two
Params: Float: 13 FloatT: F32 Int: 2 IntT: I32 MiscT: bool String: Two
Params: Float: 13 FloatT: F32 Int: 3 IntT: I32 MiscT: bool String: Two
Params: Float: 11 FloatT: F32 Int: 1 IntT: I32 MiscT: bool String: Three
Params: Float: 11 FloatT: F32 Int: 2 IntT: I32 MiscT: bool String: Three
Params: Float: 11 FloatT: F32 Int: 3 IntT: I32 MiscT: bool String: Three
Params: Float: 12 FloatT: F32 Int: 1 IntT: I32 MiscT: bool String: Three
Params: Float: 12 FloatT: F32 Int: 2 IntT: I32 MiscT: bool String: Three
Params: Float: 12 FloatT: F32 Int: 3 IntT: I32 MiscT: bool String: Three
Params: Float: 13 FloatT: F32 Int: 1 IntT: I32 MiscT: bool String: Three
Params: Float: 13 FloatT: F32 Int: 2 IntT: I32 MiscT: bool String: Three
Params: Float: 13 FloatT: F32 Int: 3 IntT: I32 MiscT: bool String: Three
Params: Float: 11 FloatT: F32 Int: 1 IntT: I32 MiscT: void String: One
Params: Float: 11 FloatT: F32 Int: 2 IntT: I32 MiscT: void String: One
Params: Float: 11 FloatT: F32 Int: 3 IntT: I32 MiscT: void String: One
Params: Float: 12 FloatT: F32 Int: 1 IntT: I32 MiscT: void String: One
Params: Float: 12 FloatT: F32 Int: 2 IntT: I32 MiscT: void String: One
Params: Float: 12 FloatT: F32 Int: 3 IntT: I32 MiscT: void String: One
Params: Float: 13 FloatT: F32 Int: 1 IntT: I32 MiscT: void String: One
Params: Float: 13 FloatT: F32 Int: 2 IntT: I32 MiscT: void String: One
Params: Float: 13 FloatT: F32 Int: 3 IntT: I32 MiscT: void String: One
Params: Float: 11 FloatT: F32 Int: 1 IntT: I32 MiscT: void String: Two
Params: Float: 11 FloatT: F32 Int: 2 IntT: I32 MiscT: void String: Two
Params: Float: 11 FloatT: F32 Int: 3 IntT: I32 MiscT: void String: Two
Params: Float: 12 FloatT: F32 Int: 1 IntT: I32 MiscT: void String: Two
Params: Float: 12 FloatT: F32 Int: 2 IntT: I32 MiscT: void String: Two
Params: Float: 12 FloatT: F32 Int: 3 IntT: I32 MiscT: void String: Two
Params: Float: 13 FloatT: F32 Int: 1 IntT: I32 MiscT: void String: Two
Params: Float: 13 FloatT: F32 Int: 2 IntT: I32 MiscT: void String: Two
Params: Float: 13 FloatT: F32 Int: 3 IntT: I32 MiscT: void String: Two
Params: Float: 11 FloatT: F32 Int: 1 IntT: I32 MiscT: void String: Three
Params: Float: 11 FloatT: F32 Int: 2 IntT: I32 MiscT: void String: Three
Params: Float: 11 FloatT: F32 Int: 3 IntT: I32 MiscT: void String: Three
Params: Float: 12 FloatT: F32 Int: 1 IntT: I32 MiscT: void String: Three
Params: Float: 12 FloatT: F32 Int: 2 IntT: I32 MiscT: void String: Three
Params: Float: 12 FloatT: F32 Int: 3 IntT: I32 MiscT: void String: Three
Params: Float: 13 FloatT: F32 Int: 1 IntT: I32 MiscT: void String: Three
Params: Float: 13 FloatT: F32 Int: 2 IntT: I32 MiscT: void String: Three
Params: Float: 13 FloatT: F32 Int: 3 IntT: I32 MiscT: void String: Three
Params: Float: 11 FloatT: F32 Int: 1 IntT: I64 MiscT: bool String: One
Params: Float: 11 FloatT: F32 Int: 2 IntT: I64 MiscT: bool String: One
Params: Float: 11 FloatT: F32 Int: 3 IntT: I64 MiscT: bool String: One
Params: Float: 12 FloatT: F32 Int: 1 IntT: I64 MiscT: bool String: One
Params: Float: 12 FloatT: F32 Int: 2 IntT: I64 MiscT: bool String: One
Params: Float: 12 FloatT: F32 Int: 3 IntT: I64 MiscT: bool String: One
Params: Float: 13 FloatT: F32 Int: 1 IntT: I64 MiscT: bool String: One
Params: Float: 13 FloatT: F32 Int: 2 IntT: I64 MiscT: bool String: One
Params: Float: 13 FloatT: F32 Int: 3 IntT: I64 MiscT: bool String: One
Params: Float: 11 FloatT: F32 Int: 1 IntT: I64 MiscT: bool String: Two
Params: Float: 11 FloatT: F32 Int: 2 IntT: I64 MiscT: bool String: Two
Params: Float: 11 FloatT: F32 Int: 3 IntT: I64 MiscT: bool String: Two
Params: Float: 12 FloatT: F32 Int: 1 IntT: I64 MiscT: bool String: Two
Params: Float: 12 FloatT: F32 Int: 2 IntT: I64 MiscT: bool String: Two
Params: Float: 12 FloatT: F32 Int: 3 IntT: I64 MiscT: bool String: Two
Params: Float: 13 FloatT: F32 Int: 1 IntT: I64 MiscT: bool String: Two
Params: Float: 13 FloatT: F32 Int: 2 IntT: I64 MiscT: bool String: Two
Params: Float: 13 FloatT: F32 Int: 3 IntT: I64 MiscT: bool String: Two
Params: Float: 11 FloatT: F32 Int: 1 IntT: I64 MiscT: bool String: Three
Params: Float: 11 FloatT: F32 Int: 2 IntT: I64 MiscT: bool String: Three
Params: Float: 11 FloatT: F32 Int: 3 IntT: I64 MiscT: bool String: Three
Params: Float: 12 FloatT: F32 Int: 1 IntT: I64 MiscT: bool String: Three
Params: Float: 12 FloatT: F32 Int: 2 IntT: I64 MiscT: bool String: Three
Params: Float: 12 FloatT: F32 Int: 3 IntT: I64 MiscT: bool String: Three
Params: Float: 13 FloatT: F32 Int: 1 IntT: I64 MiscT: bool String: Three
Params: Float: 13 FloatT: F32 Int: 2 IntT: I64 MiscT: bool String: Three
Params: Float: 13 FloatT: F32 Int: 3 IntT: I64 MiscT: bool String: Three
Params: Float: 11 FloatT: F32 Int: 1 IntT: I64 MiscT: void String: One
Params: Float: 11 FloatT: F32 Int: 2 IntT: I64 MiscT: void String: One
Params: Float: 11 FloatT: F32 Int: 3 IntT: I64 MiscT: void String: One
Params: Float: 12 FloatT: F32 Int: 1 IntT: I64 MiscT: void String: One
Params: Float: 12 FloatT: F32 Int: 2 IntT: I64 MiscT: void String: One
Params: Float: 12 FloatT: F32 Int: 3 IntT: I64 MiscT: void String: One
Params: Float: 13 FloatT: F32 Int: 1 IntT: I64 MiscT: void String: One
Params: Float: 13 FloatT: F32 Int: 2 IntT: I64 MiscT: void String: One
Params: Float: 13 FloatT: F32 Int: 3 IntT: I64 MiscT: void String: One
Params: Float: 11 FloatT: F32 Int: 1 IntT: I64 MiscT: void String: Two
Params: Float: 11 FloatT: F32 Int: 2 IntT: I64 MiscT: void String: Two
Params: Float: 11 FloatT: F32 Int: 3 IntT: I64 MiscT: void String: Two
Params: Float: 12 FloatT: F32 Int: 1 IntT: I64 MiscT: void String: Two
Params: Float: 12 FloatT: F32 Int: 2 IntT: I64 MiscT: void String: Two
Params: Float: 12 FloatT: F32 Int: 3 IntT: I64 MiscT: void String: Two
Params: Float: 13 FloatT: F32 Int: 1 IntT: I64 MiscT: void String: Two
Params: Float: 13 FloatT: F32 Int: 2 IntT: I64 MiscT: void String: Two
Params: Float: 13 FloatT: F32 Int: 3 IntT: I64 MiscT: void String: Two
Params: Float: 11 FloatT: F32 Int: 1 IntT: I64 MiscT: void String: Three
Params: Float: 11 FloatT: F32 Int: 2 IntT: I64 MiscT: void String: Three
Params: Float: 11 FloatT: F32 Int: 3 IntT: I64 MiscT: void String: Three
Params: Float: 12 FloatT: F32 Int: 1 IntT: I64 MiscT: void String: Three
Params: Float: 12 FloatT: F32 Int: 2 IntT: I64 MiscT: void String: Three
Params: Float: 12 FloatT: F32 Int: 3 IntT: I64 MiscT: void String: Three
Params: Float: 13 FloatT: F32 Int: 1 IntT: I64 MiscT: void String: Three
Params: Float: 13 FloatT: F32 Int: 2 IntT: I64 MiscT: void String: Three
Params: Float: 13 FloatT: F32 Int: 3 IntT: I64 MiscT: void String: Three
Params: Float: 11 FloatT: F64 Int: 1 IntT: I32 MiscT: bool String: One
Params: Float: 11 FloatT: F64 Int: 2 IntT: I32 MiscT: bool String: One
Params: Float: 11 FloatT: F64 Int: 3 IntT: I32 MiscT: bool String: One
Params: Float: 12 FloatT: F64 Int: 1 IntT: I32 MiscT: bool String: One
Params: Float: 12 FloatT: F64 Int: 2 IntT: I32 MiscT: bool String: One
Params: Float: 12 FloatT: F64 Int: 3 IntT: I32 MiscT: bool String: One
Params: Float: 13 FloatT: F64 Int: 1 IntT: I32 MiscT: bool String: One
Params: Float: 13 FloatT: F64 Int: 2 IntT: I32 MiscT: bool String: One
Params: Float: 13 FloatT: F64 Int: 3 IntT: I32 MiscT: bool String: One
Params: Float: 11 FloatT: F64 Int: 1 IntT: I32 MiscT: bool String: Two
Params: Float: 11 FloatT: F64 Int: 2 IntT: I32 MiscT: bool String: Two
Params: Float: 11 FloatT: F64 Int: 3 IntT: I32 MiscT: bool String: Two
Params: Float: 12 FloatT: F64 Int: 1 IntT: I32 MiscT: bool String: Two
Params: Float: 12 FloatT: F64 Int: 2 IntT: I32 MiscT: bool String: Two
Params: Float: 12 FloatT: F64 Int: 3 IntT: I32 MiscT: bool String: Two
Params: Float: 13 FloatT: F64 Int: 1 IntT: I32 MiscT: bool String: Two
Params: Float: 13 FloatT: F64 Int: 2 IntT: I32 MiscT: bool String: Two
Params: Float: 13 FloatT: F64 Int: 3 IntT: I32 MiscT: bool String: Two
Params: Float: 11 FloatT: F64 Int: 1 IntT: I32 MiscT: bool String: Three
Params: Float: 11 FloatT: F64 Int: 2 IntT: I32 MiscT: bool String: Three
Params: Float: 11 FloatT: F64 Int: 3 IntT: I32 MiscT: bool String: Three
Params: Float: 12 FloatT: F64 Int: 1 IntT: I32 MiscT: bool String: Three
Params: Float: 12 FloatT: F64 Int: 2 IntT: I32 MiscT: bool String: Three
Params: Float: 12 FloatT: F64 Int: 3 IntT: I32 MiscT: bool String: Three
Params: Float: 13 FloatT: F64 Int: 1 IntT: I32 MiscT: bool String: Three
Params: Float: 13 FloatT: F64 Int: 2 IntT: I32 MiscT: bool String: Three
Params: Float: 13 FloatT: F64 Int: 3 IntT: I32 MiscT: bool String: Three
Params: Float: 11 FloatT: F64 Int: 1 IntT: I32 MiscT: void String: One
Params: Float: 11 FloatT: F64 Int: 2 IntT: I32 MiscT: void String: One
Params: Float: 11 FloatT: F64 Int: 3 IntT: I32 MiscT: void String: One
Params: Float: 12 FloatT: F64 Int: 1 IntT: I32 MiscT: void String: One
Params: Float: 12 FloatT: F64 Int: 2 IntT: I32 MiscT: void String: One
Params: Float: 12 FloatT: F64 Int: 3 IntT: I32 MiscT: void String: One
Params: Float: 13 FloatT: F64 Int: 1 IntT: I32 MiscT: void String: One
Params: Float: 13 FloatT: F64 Int: 2 IntT: I32 MiscT: void String: One
Params: Float: 13 FloatT: F64 Int: 3 IntT: I32 MiscT: void String: One
Params: Float: 11 FloatT: F64 Int: 1 IntT: I32 MiscT: void String: Two
Params: Float: 11 FloatT: F64 Int: 2 IntT: I32 MiscT: void String: Two
Params: Float: 11 FloatT: F64 Int: 3 IntT: I32 MiscT: void String: Two
Params: Float: 12 FloatT: F64 Int: 1 IntT: I32 MiscT: void String: Two
Params: Float: 12 FloatT: F64 Int: 2 IntT: I32 MiscT: void String: Two
Params: Float: 12 FloatT: F64 Int: 3 IntT: I32 MiscT: void String: Two
Params: Float: 13 FloatT: F64 Int: 1 IntT: I32 MiscT: void String: Two
Params: Float: 13 FloatT: F64 Int: 2 IntT: I32 MiscT: void String: Two
Params: Float: 13 FloatT: F64 Int: 3 IntT: I32 MiscT: void String: Two
Params: Float: 11 FloatT: F64 Int: 1 IntT: I32 MiscT: void String: Three
Params: Float: 11 FloatT: F64 Int: 2 IntT: I32 MiscT: void String: Three
Params: Float: 11 FloatT: F64 Int: 3 IntT: I32 MiscT: void String: Three
Params: Float: 12 FloatT: F64 Int: 1 IntT: I32 MiscT: void String: Three
Params: Float: 12 FloatT: F64 Int: 2 IntT: I32 MiscT: void String: Three
Params: Float: 12 FloatT: F64 Int: 3 IntT: I32 MiscT: void String: Three
Params: Float: 13 FloatT: F64 Int: 1 IntT: I32 MiscT: void String: Three
Params: Float: 13 FloatT: F64 Int: 2 IntT: I32 MiscT: void String: Three
Params: Float: 13 FloatT: F64 Int: 3 IntT: I32 MiscT: void String: Three
Params: Float: 11 FloatT: F64 Int: 1 IntT: I64 MiscT: bool String: One
Params: Float: 11 FloatT: F64 Int: 2 IntT: I64 MiscT: bool String: One
Params: Float: 11 FloatT: F64 Int: 3 IntT: I64 MiscT: bool String: One
Params: Float: 12 FloatT: F64 Int: 1 IntT: I64 MiscT: bool String: One
Params: Float: 12 FloatT: F64 Int: 2 IntT: I64 MiscT: bool String: One
Params: Float: 12 FloatT: F64 Int: 3 IntT: I64 MiscT: bool String: One
Params: Float: 13 FloatT: F64 Int: 1 IntT: I64 MiscT: bool String: One
Params: Float: 13 FloatT: F64 Int: 2 IntT: I64 MiscT: bool String: One
Params: Float: 13 FloatT: F64 Int: 3 IntT: I64 MiscT: bool String: One
Params: Float: 11 FloatT: F64 Int: 1 IntT: I64 MiscT: bool String: Two
Params: Float: 11 FloatT: F64 Int: 2 IntT: I64 MiscT: bool String: Two
Params: Float: 11 FloatT: F64 Int: 3 IntT: I64 MiscT: bool String: Two
Params: Float: 12 FloatT: F64 Int: 1 IntT: I64 MiscT: bool String: Two
Params: Float: 12 FloatT: F64 Int: 2 IntT: I64 MiscT: bool String: Two
Params: Float: 12 FloatT: F64 Int: 3 IntT: I64 MiscT: bool String: Two
Params: Float: 13 FloatT: F64 Int: 1 IntT: I64 MiscT: bool String: Two
Params: Float: 13 FloatT: F64 Int: 2 IntT: I64 MiscT: bool String: Two
Params: Float: 13 FloatT: F64 Int: 3 IntT: I64 MiscT: bool String: Two
Params: Float: 11 FloatT: F64 Int: 1 IntT: I64 MiscT: bool String: Three
Params: Float: 11 FloatT: F64 Int: 2 IntT: I64 MiscT: bool String: Three
Params: Float: 11 FloatT: F64 Int: 3 IntT: I64 MiscT: bool String: Three
Params: Float: 12 FloatT: F64 Int: 1 IntT: I64 MiscT: bool String: Three
Params: Float: 12 FloatT: F64 Int: 2 IntT: I64 MiscT: bool String: Three
Params: Float: 12 FloatT: F64 Int: 3 IntT: I64 MiscT: bool String: Three
Params: Float: 13 FloatT: F64 Int: 1 IntT: I64 MiscT: bool String: Three
Params: Float: 13 FloatT: F64 Int: 2 IntT: I64 MiscT: bool String: Three
Params: Float: 13 FloatT: F64 Int: 3 IntT: I64 MiscT: bool String: Three
Params: Float: 11 FloatT: F64 Int: 1 IntT: I64 MiscT: void String: One
Params: Float: 11 FloatT: F64 Int: 2 IntT: I64 MiscT: void String: One
Params: Float: 11 FloatT: F64 Int: 3 IntT: I64 MiscT: void String: One
Params: Float: 12 FloatT: F64 Int: 1 IntT: I64 MiscT: void String: One
Params: Float: 12 FloatT: F64 Int: 2 IntT: I64 MiscT: void String: One
Params: Float: 12 FloatT: F64 Int: 3 IntT: I64 MiscT: void String: One
Params: Float: 13 FloatT: F64 Int: 1 IntT: I64 MiscT: void String: One
Params: Float: 13 FloatT: F64 Int: 2 IntT: I64 MiscT: void String: One
Params: Float: 13 FloatT: F64 Int: 3 IntT: I64 MiscT: void String: One
Params: Float: 11 FloatT: F64 Int: 1 IntT: I64 MiscT: void String: Two
Params: Float: 11 FloatT: F64 Int: 2 IntT: I64 MiscT: void String: Two
Params: Float: 11 FloatT: F64 Int: 3 IntT: I64 MiscT: void String: Two
Params: Float: 12 FloatT: F64 Int: 1 IntT: I64 MiscT: void String: Two
Params: Float: 12 FloatT: F64 Int: 2 IntT: I64 MiscT: void String: Two
Params: Float: 12 FloatT: F64 Int: 3 IntT: I64 MiscT: void String: Two
Params: Float: 13 FloatT: F64 Int: 1 IntT: I64 MiscT: void String: Two
Params: Float: 13 FloatT: F64 Int: 2 IntT: I64 MiscT: void String: Two
Params: Float: 13 FloatT: F64 Int: 3 IntT: I64 MiscT: void String: Two
Params: Float: 11 FloatT: F64 Int: 1 IntT: I64 MiscT: void String: Three
Params: Float: 11 FloatT: F64 Int: 2 IntT: I64 MiscT: void String: Three
Params: Float: 11 FloatT: F64 Int: 3 IntT: I64 MiscT: void String: Three
Params: Float: 12 FloatT: F64 Int: 1 IntT: I64 MiscT: void String: Three
Params: Float: 12 FloatT: F64 Int: 2 IntT: I64 MiscT: void String: Three
Params: Float: 12 FloatT: F64 Int: 3 IntT: I64 MiscT: void String: Three
Params: Float: 13 FloatT: F64 Int: 1 IntT: I64 MiscT: void String: Three
Params: Float: 13 FloatT: F64 Int: 2 IntT: I64 MiscT: void String: Three
Params: Float: 13 FloatT: F64 Int: 3 IntT: I64 MiscT: void String: Three
)expected";

  const std::string test = fmt::to_string(buffer);
  ASSERT_MSG(test == ref, "Expected:\n\"{}\"\n\nActual:\n\"{}\"", ref, test);
}

int main()
{
  test_empty();
  test_non_types();
  test_types();
  test_both();
}
