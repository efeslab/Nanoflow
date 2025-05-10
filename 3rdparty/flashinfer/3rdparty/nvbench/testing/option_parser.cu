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

#include <nvbench/option_parser.cuh>

#include <nvbench/create.cuh>
#include <nvbench/type_list.cuh>

#include "test_asserts.cuh"

#include <fmt/format.h>

//==============================================================================
// Declare a couple benchmarks for testing:
void DummyBench(nvbench::state &state) { state.skip("Skipping for testing."); }
NVBENCH_BENCH(DummyBench).clear_devices();

using Ts = nvbench::type_list<void, nvbench::int8_t, nvbench::uint8_t>;
using Us = nvbench::type_list<bool, nvbench::float32_t, nvbench::float64_t>;

template <typename T, typename U>
void TestBench(nvbench::state &state, nvbench::type_list<T, U>)
{
  DummyBench(state);
}
NVBENCH_BENCH_TYPES(TestBench, NVBENCH_TYPE_AXES(Ts, Us))
  .set_type_axes_names({"T", "U"})
  .add_int64_axis("Ints", {42})
  .add_int64_power_of_two_axis("PO2s", {3})
  .add_float64_axis("Floats", {3.14})
  .add_string_axis("Strings", {"S1"})
  .clear_devices();
//==============================================================================

namespace
{

[[nodiscard]] std::string
states_to_string(const std::vector<nvbench::state> &states)
{
  fmt::memory_buffer buffer;
  std::string table_format = "| {:^5} | {:^10} | {:^4} | {:^4} | {:^4} "
                             "| {:^4} | {:^6} | {:^8} |\n";

  fmt::format_to(std::back_inserter(buffer), "\n");
  fmt::format_to(std::back_inserter(buffer),
                 table_format,
                 "State",
                 "TypeConfig",
                 "T",
                 "U",
                 "Ints",
                 "PO2s",
                 "Floats",
                 "Strings");

  std::size_t config = 0;
  for (const auto &state : states)
  {
    fmt::format_to(std::back_inserter(buffer),
                   table_format,
                   config++,
                   state.get_type_config_index(),
                   state.get_string("T"),
                   state.get_string("U"),
                   state.get_int64("Ints"),
                   state.get_int64("PO2s"),
                   state.get_float64("Floats"),
                   std::string{"\'"} + state.get_string("Strings") + "'");
  }
  return fmt::to_string(buffer);
}

// Expects the parser to have a single TestBench benchmark. Runs the benchmark
// and returns the resulting states.
[[nodiscard]] const auto& parser_to_states(nvbench::option_parser &parser)
{
  const auto &benches = parser.get_benchmarks();
  ASSERT(benches.size() == 1);
  const auto &bench = benches.front();
  ASSERT(bench != nullptr);

  bench->run();

  return bench->get_states();
}

// Expects the parser to have a single TestBench benchmark. Runs the benchmark
// and converts the generated states into a fingerprint string for regression
// testing.
[[nodiscard]] std::string parser_to_state_string(nvbench::option_parser &parser)
{
  return states_to_string(parser_to_states(parser));
}

} // namespace

void test_empty()
{
  {
    nvbench::option_parser parser;
    parser.parse({});
    ASSERT(parser.get_benchmarks().size() == 2);
    ASSERT(parser.get_args().empty());
  }

  {
    nvbench::option_parser parser;
    parser.parse(0, nullptr);
    ASSERT(parser.get_benchmarks().size() == 2);
    ASSERT(parser.get_args().empty());
  }
}

void test_exec_name_tolerance()
{
  nvbench::option_parser parser;
  parser.parse({"TestExec"});
  ASSERT(parser.get_benchmarks().size() == 2);
  ASSERT(parser.get_args() == std::vector<std::string>{"TestExec"});
}

void test_argc_argv_parse()
{
  char const *const argv[] = {"TestExec"};
  {
    nvbench::option_parser parser;
    parser.parse(1, argv);
    ASSERT(parser.get_benchmarks().size() == 2);
    ASSERT(parser.get_args() == std::vector<std::string>{"TestExec"});
  }

  {
    nvbench::option_parser parser;
    parser.parse(0, nullptr);
    ASSERT(parser.get_benchmarks().size() == 2);
    ASSERT(parser.get_args().empty());
  }
}

void test_invalid_option()
{
  nvbench::option_parser parser;
  ASSERT_THROWS_ANY(parser.parse({"--not-a-real-option"}));
}

void test_benchmark_long() // --benchmark
{
  const std::string ref =
    R"expected(
| State | TypeConfig |  T   |  U   | Ints | PO2s | Floats | Strings  |
|   0   |     0      | void | bool |  42  |  8   |  3.14  |   'S1'   |
|   1   |     1      | void | F32  |  42  |  8   |  3.14  |   'S1'   |
|   2   |     2      | void | F64  |  42  |  8   |  3.14  |   'S1'   |
|   3   |     3      |  I8  | bool |  42  |  8   |  3.14  |   'S1'   |
|   4   |     4      |  I8  | F32  |  42  |  8   |  3.14  |   'S1'   |
|   5   |     5      |  I8  | F64  |  42  |  8   |  3.14  |   'S1'   |
|   6   |     6      |  U8  | bool |  42  |  8   |  3.14  |   'S1'   |
|   7   |     7      |  U8  | F32  |  42  |  8   |  3.14  |   'S1'   |
|   8   |     8      |  U8  | F64  |  42  |  8   |  3.14  |   'S1'   |
)expected";

  {
    nvbench::option_parser parser;
    parser.parse({"--benchmark", "TestBench"});
    const auto test = parser_to_state_string(parser);
    ASSERT_MSG(test == ref, "Expected:\n\"{}\"\n\nActual:\n\"{}\"", ref, test);
  }

  {
    nvbench::option_parser parser;
    parser.parse({"--benchmark", "1"});
    const auto test = parser_to_state_string(parser);
    ASSERT_MSG(test == ref, "Expected:\n\"{}\"\n\nActual:\n\"{}\"", ref, test);
  }
}

void test_benchmark_short() // -b
{
  const std::string ref =
    R"expected(
| State | TypeConfig |  T   |  U   | Ints | PO2s | Floats | Strings  |
|   0   |     0      | void | bool |  42  |  8   |  3.14  |   'S1'   |
|   1   |     1      | void | F32  |  42  |  8   |  3.14  |   'S1'   |
|   2   |     2      | void | F64  |  42  |  8   |  3.14  |   'S1'   |
|   3   |     3      |  I8  | bool |  42  |  8   |  3.14  |   'S1'   |
|   4   |     4      |  I8  | F32  |  42  |  8   |  3.14  |   'S1'   |
|   5   |     5      |  I8  | F64  |  42  |  8   |  3.14  |   'S1'   |
|   6   |     6      |  U8  | bool |  42  |  8   |  3.14  |   'S1'   |
|   7   |     7      |  U8  | F32  |  42  |  8   |  3.14  |   'S1'   |
|   8   |     8      |  U8  | F64  |  42  |  8   |  3.14  |   'S1'   |
)expected";

  {
    nvbench::option_parser parser;
    parser.parse({"-b", "TestBench"});
    const auto test = parser_to_state_string(parser);
    ASSERT_MSG(test == ref, "Expected:\n\"{}\"\n\nActual:\n\"{}\"", ref, test);
  }

  {
    nvbench::option_parser parser;
    parser.parse({"-b", "1"});
    const auto test = parser_to_state_string(parser);
    ASSERT_MSG(test == ref, "Expected:\n\"{}\"\n\nActual:\n\"{}\"", ref, test);
  }
}

void test_int64_axis_single()
{
  const std::string ref =
    R"expected(
| State | TypeConfig |  T   |  U   | Ints | PO2s | Floats | Strings  |
|   0   |     0      | void | bool |  2   |  8   |  3.14  |   'S1'   |
|   1   |     1      | void | F32  |  2   |  8   |  3.14  |   'S1'   |
|   2   |     2      | void | F64  |  2   |  8   |  3.14  |   'S1'   |
|   3   |     3      |  I8  | bool |  2   |  8   |  3.14  |   'S1'   |
|   4   |     4      |  I8  | F32  |  2   |  8   |  3.14  |   'S1'   |
|   5   |     5      |  I8  | F64  |  2   |  8   |  3.14  |   'S1'   |
|   6   |     6      |  U8  | bool |  2   |  8   |  3.14  |   'S1'   |
|   7   |     7      |  U8  | F32  |  2   |  8   |  3.14  |   'S1'   |
|   8   |     8      |  U8  | F64  |  2   |  8   |  3.14  |   'S1'   |
)expected";

  {
    nvbench::option_parser parser;
    parser.parse({"--benchmark", "TestBench", "--axis", " Ints [ ] =  2 "});
    const auto test = parser_to_state_string(parser);
    ASSERT_MSG(test == ref, "Expected:\n\"{}\"\n\nActual:\n\"{}\"", ref, test);
  }

  {
    nvbench::option_parser parser;
    parser.parse({"--benchmark", "TestBench", "--axis", " Ints=2"});
    const auto test = parser_to_state_string(parser);
    ASSERT_MSG(test == ref, "Expected:\n\"{}\"\n\nActual:\n\"{}\"", ref, test);
  }

  {
    nvbench::option_parser parser;
    parser.parse({"--benchmark", "TestBench", "--axis", " Ints [ ] = [ 2 ]"});
    const auto test = parser_to_state_string(parser);
    ASSERT_MSG(test == ref, "Expected:\n\"{}\"\n\nActual:\n\"{}\"", ref, test);
  }

  {
    nvbench::option_parser parser;
    parser.parse({"--benchmark", "TestBench", "--axis", "Ints=[2]"});
    const auto test = parser_to_state_string(parser);
    ASSERT_MSG(test == ref, "Expected:\n\"{}\"\n\nActual:\n\"{}\"", ref, test);
  }

  {
    nvbench::option_parser parser;
    parser.parse(
      {"--benchmark", "TestBench", "--axis", " Ints [ ] = [ 2 : 2 : 1 ] "});
    const auto test = parser_to_state_string(parser);
    ASSERT_MSG(test == ref, "Expected:\n\"{}\"\n\nActual:\n\"{}\"", ref, test);
  }

  {
    nvbench::option_parser parser;
    parser.parse({"--benchmark", "TestBench", "--axis", "Ints=[2:2]"});
    const auto test = parser_to_state_string(parser);
    ASSERT_MSG(test == ref, "Expected:\n\"{}\"\n\nActual:\n\"{}\"", ref, test);
  }
}

void test_int64_axis_multi()
{
  const std::string ref =
    R"expected(
| State | TypeConfig |  T   |  U   | Ints | PO2s | Floats | Strings  |
|   0   |     0      | void | bool |  2   |  8   |  3.14  |   'S1'   |
|   1   |     0      | void | bool |  7   |  8   |  3.14  |   'S1'   |
|   2   |     1      | void | F32  |  2   |  8   |  3.14  |   'S1'   |
|   3   |     1      | void | F32  |  7   |  8   |  3.14  |   'S1'   |
|   4   |     2      | void | F64  |  2   |  8   |  3.14  |   'S1'   |
|   5   |     2      | void | F64  |  7   |  8   |  3.14  |   'S1'   |
|   6   |     3      |  I8  | bool |  2   |  8   |  3.14  |   'S1'   |
|   7   |     3      |  I8  | bool |  7   |  8   |  3.14  |   'S1'   |
|   8   |     4      |  I8  | F32  |  2   |  8   |  3.14  |   'S1'   |
|   9   |     4      |  I8  | F32  |  7   |  8   |  3.14  |   'S1'   |
|  10   |     5      |  I8  | F64  |  2   |  8   |  3.14  |   'S1'   |
|  11   |     5      |  I8  | F64  |  7   |  8   |  3.14  |   'S1'   |
|  12   |     6      |  U8  | bool |  2   |  8   |  3.14  |   'S1'   |
|  13   |     6      |  U8  | bool |  7   |  8   |  3.14  |   'S1'   |
|  14   |     7      |  U8  | F32  |  2   |  8   |  3.14  |   'S1'   |
|  15   |     7      |  U8  | F32  |  7   |  8   |  3.14  |   'S1'   |
|  16   |     8      |  U8  | F64  |  2   |  8   |  3.14  |   'S1'   |
|  17   |     8      |  U8  | F64  |  7   |  8   |  3.14  |   'S1'   |
)expected";

  {
    nvbench::option_parser parser;
    parser.parse(
      {"--benchmark", "TestBench", "--axis", " Ints [ ] = [ 2 , 7 ] "});
    const auto test = parser_to_state_string(parser);
    ASSERT_MSG(test == ref, "Expected:\n\"{}\"\n\nActual:\n\"{}\"", ref, test);
  }

  {
    nvbench::option_parser parser;
    parser.parse({"--benchmark", "TestBench", "--axis", "Ints=[2,7]"});
    const auto test = parser_to_state_string(parser);
    ASSERT_MSG(test == ref, "Expected:\n\"{}\"\n\nActual:\n\"{}\"", ref, test);
  }

  {
    nvbench::option_parser parser;
    parser.parse(
      {"--benchmark", "TestBench", "--axis", " Ints [ ] = [ 2 : 7 : 5 ] "});
    const auto test = parser_to_state_string(parser);
    ASSERT_MSG(test == ref, "Expected:\n\"{}\"\n\nActual:\n\"{}\"", ref, test);
  }

  {
    nvbench::option_parser parser;
    parser.parse({"--benchmark", "TestBench", "--axis", "Ints=[2:7:5]"});
    const auto test = parser_to_state_string(parser);
    ASSERT_MSG(test == ref, "Expected:\n\"{}\"\n\nActual:\n\"{}\"", ref, test);
  }
}

void test_int64_axis_pow2_single()
{
  const std::string ref =
    R"expected(
| State | TypeConfig |  T   |  U   | Ints | PO2s | Floats | Strings  |
|   0   |     0      | void | bool |  42  | 128  |  3.14  |   'S1'   |
|   1   |     1      | void | F32  |  42  | 128  |  3.14  |   'S1'   |
|   2   |     2      | void | F64  |  42  | 128  |  3.14  |   'S1'   |
|   3   |     3      |  I8  | bool |  42  | 128  |  3.14  |   'S1'   |
|   4   |     4      |  I8  | F32  |  42  | 128  |  3.14  |   'S1'   |
|   5   |     5      |  I8  | F64  |  42  | 128  |  3.14  |   'S1'   |
|   6   |     6      |  U8  | bool |  42  | 128  |  3.14  |   'S1'   |
|   7   |     7      |  U8  | F32  |  42  | 128  |  3.14  |   'S1'   |
|   8   |     8      |  U8  | F64  |  42  | 128  |  3.14  |   'S1'   |
)expected";

  {
    nvbench::option_parser parser;
    parser.parse({"--benchmark", "TestBench", "--axis", " PO2s [ pow2 ] = 7 "});
    const auto test = parser_to_state_string(parser);
    ASSERT_MSG(test == ref, "Expected:\n\"{}\"\n\nActual:\n\"{}\"", ref, test);
  }

  {
    nvbench::option_parser parser;
    parser.parse({"--benchmark", "TestBench", "--axis", "PO2s[pow2]=7"});
    const auto test = parser_to_state_string(parser);
    ASSERT_MSG(test == ref, "Expected:\n\"{}\"\n\nActual:\n\"{}\"", ref, test);
  }

  {
    nvbench::option_parser parser;
    parser.parse(
      {"--benchmark", "TestBench", "--axis", " PO2s [ pow2 ] = [ 7 ] "});
    const auto test = parser_to_state_string(parser);
    ASSERT_MSG(test == ref, "Expected:\n\"{}\"\n\nActual:\n\"{}\"", ref, test);
  }

  {
    nvbench::option_parser parser;
    parser.parse({"--benchmark", "TestBench", "--axis", "PO2s[pow2]=[7]"});
    const auto test = parser_to_state_string(parser);
    ASSERT_MSG(test == ref, "Expected:\n\"{}\"\n\nActual:\n\"{}\"", ref, test);
  }

  {
    nvbench::option_parser parser;
    parser.parse(
      {"--benchmark", "TestBench", "--axis", " PO2s [ pow2 ] = [ 7 : 7 : 1 ] "});
    const auto test = parser_to_state_string(parser);
    ASSERT_MSG(test == ref, "Expected:\n\"{}\"\n\nActual:\n\"{}\"", ref, test);
  }

  {
    nvbench::option_parser parser;
    parser.parse({"--benchmark", "TestBench", "--axis", "PO2s[pow2]=[7:7]"});
    const auto test = parser_to_state_string(parser);
    ASSERT_MSG(test == ref, "Expected:\n\"{}\"\n\nActual:\n\"{}\"", ref, test);
  }
}

void test_int64_axis_pow2_multi()
{
  const std::string ref =
    R"expected(
| State | TypeConfig |  T   |  U   | Ints | PO2s | Floats | Strings  |
|   0   |     0      | void | bool |  42  |  4   |  3.14  |   'S1'   |
|   1   |     0      | void | bool |  42  | 128  |  3.14  |   'S1'   |
|   2   |     1      | void | F32  |  42  |  4   |  3.14  |   'S1'   |
|   3   |     1      | void | F32  |  42  | 128  |  3.14  |   'S1'   |
|   4   |     2      | void | F64  |  42  |  4   |  3.14  |   'S1'   |
|   5   |     2      | void | F64  |  42  | 128  |  3.14  |   'S1'   |
|   6   |     3      |  I8  | bool |  42  |  4   |  3.14  |   'S1'   |
|   7   |     3      |  I8  | bool |  42  | 128  |  3.14  |   'S1'   |
|   8   |     4      |  I8  | F32  |  42  |  4   |  3.14  |   'S1'   |
|   9   |     4      |  I8  | F32  |  42  | 128  |  3.14  |   'S1'   |
|  10   |     5      |  I8  | F64  |  42  |  4   |  3.14  |   'S1'   |
|  11   |     5      |  I8  | F64  |  42  | 128  |  3.14  |   'S1'   |
|  12   |     6      |  U8  | bool |  42  |  4   |  3.14  |   'S1'   |
|  13   |     6      |  U8  | bool |  42  | 128  |  3.14  |   'S1'   |
|  14   |     7      |  U8  | F32  |  42  |  4   |  3.14  |   'S1'   |
|  15   |     7      |  U8  | F32  |  42  | 128  |  3.14  |   'S1'   |
|  16   |     8      |  U8  | F64  |  42  |  4   |  3.14  |   'S1'   |
|  17   |     8      |  U8  | F64  |  42  | 128  |  3.14  |   'S1'   |
)expected";

  {
    nvbench::option_parser parser;
    parser.parse(
      {"--benchmark", "TestBench", "--axis", " PO2s [ pow2 ] = [ 2 , 7 ] "});
    const auto test = parser_to_state_string(parser);
    ASSERT_MSG(test == ref, "Expected:\n\"{}\"\n\nActual:\n\"{}\"", ref, test);
  }

  {
    nvbench::option_parser parser;
    parser.parse({"--benchmark", "TestBench", "--axis", "PO2s[pow2]=[2,7]"});
    const auto test = parser_to_state_string(parser);
    ASSERT_MSG(test == ref, "Expected:\n\"{}\"\n\nActual:\n\"{}\"", ref, test);
  }

  {
    nvbench::option_parser parser;
    parser.parse(
      {"--benchmark", "TestBench", "--axis", " PO2s [ pow2 ] = [ 2 : 7 : 5 ] "});
    const auto test = parser_to_state_string(parser);
    ASSERT_MSG(test == ref, "Expected:\n\"{}\"\n\nActual:\n\"{}\"", ref, test);
  }

  {
    nvbench::option_parser parser;
    parser.parse({"--benchmark", "TestBench", "--axis", "PO2s[pow2]=[2:7:5]"});
    const auto test = parser_to_state_string(parser);
    ASSERT_MSG(test == ref, "Expected:\n\"{}\"\n\nActual:\n\"{}\"", ref, test);
  }
}

void test_int64_axis_none_to_pow2_single()
{
  const std::string ref =
    R"expected(
| State | TypeConfig |  T   |  U   | Ints | PO2s | Floats | Strings  |
|   0   |     0      | void | bool | 128  |  8   |  3.14  |   'S1'   |
|   1   |     1      | void | F32  | 128  |  8   |  3.14  |   'S1'   |
|   2   |     2      | void | F64  | 128  |  8   |  3.14  |   'S1'   |
|   3   |     3      |  I8  | bool | 128  |  8   |  3.14  |   'S1'   |
|   4   |     4      |  I8  | F32  | 128  |  8   |  3.14  |   'S1'   |
|   5   |     5      |  I8  | F64  | 128  |  8   |  3.14  |   'S1'   |
|   6   |     6      |  U8  | bool | 128  |  8   |  3.14  |   'S1'   |
|   7   |     7      |  U8  | F32  | 128  |  8   |  3.14  |   'S1'   |
|   8   |     8      |  U8  | F64  | 128  |  8   |  3.14  |   'S1'   |
)expected";

  {
    nvbench::option_parser parser;
    parser.parse({"--benchmark", "TestBench", "--axis", " Ints [ pow2 ] = 7 "});
    const auto test = parser_to_state_string(parser);
    ASSERT_MSG(test == ref, "Expected:\n\"{}\"\n\nActual:\n\"{}\"", ref, test);
  }

  {
    nvbench::option_parser parser;
    parser.parse({"--benchmark", "TestBench", "--axis", "Ints[pow2]=7"});
    const auto test = parser_to_state_string(parser);
    ASSERT_MSG(test == ref, "Expected:\n\"{}\"\n\nActual:\n\"{}\"", ref, test);
  }

  {
    nvbench::option_parser parser;
    parser.parse(
      {"--benchmark", "TestBench", "--axis", " Ints [ pow2 ] = [ 7 ] "});
    const auto test = parser_to_state_string(parser);
    ASSERT_MSG(test == ref, "Expected:\n\"{}\"\n\nActual:\n\"{}\"", ref, test);
  }

  {
    nvbench::option_parser parser;
    parser.parse({"--benchmark", "TestBench", "--axis", "Ints[pow2]=[7]"});
    const auto test = parser_to_state_string(parser);
    ASSERT_MSG(test == ref, "Expected:\n\"{}\"\n\nActual:\n\"{}\"", ref, test);
  }

  {
    nvbench::option_parser parser;
    parser.parse(
      {"--benchmark", "TestBench", "--axis", " Ints [ pow2 ] = [ 7 : 7 : 1 ] "});
    const auto test = parser_to_state_string(parser);
    ASSERT_MSG(test == ref, "Expected:\n\"{}\"\n\nActual:\n\"{}\"", ref, test);
  }

  {
    nvbench::option_parser parser;
    parser.parse({"--benchmark", "TestBench", "--axis", "Ints[pow2]=[7:7]"});
    const auto test = parser_to_state_string(parser);
    ASSERT_MSG(test == ref, "Expected:\n\"{}\"\n\nActual:\n\"{}\"", ref, test);
  }
}

void test_int64_axis_none_to_pow2_multi()
{
  const std::string ref =
    R"expected(
| State | TypeConfig |  T   |  U   | Ints | PO2s | Floats | Strings  |
|   0   |     0      | void | bool |  4   |  8   |  3.14  |   'S1'   |
|   1   |     0      | void | bool | 128  |  8   |  3.14  |   'S1'   |
|   2   |     1      | void | F32  |  4   |  8   |  3.14  |   'S1'   |
|   3   |     1      | void | F32  | 128  |  8   |  3.14  |   'S1'   |
|   4   |     2      | void | F64  |  4   |  8   |  3.14  |   'S1'   |
|   5   |     2      | void | F64  | 128  |  8   |  3.14  |   'S1'   |
|   6   |     3      |  I8  | bool |  4   |  8   |  3.14  |   'S1'   |
|   7   |     3      |  I8  | bool | 128  |  8   |  3.14  |   'S1'   |
|   8   |     4      |  I8  | F32  |  4   |  8   |  3.14  |   'S1'   |
|   9   |     4      |  I8  | F32  | 128  |  8   |  3.14  |   'S1'   |
|  10   |     5      |  I8  | F64  |  4   |  8   |  3.14  |   'S1'   |
|  11   |     5      |  I8  | F64  | 128  |  8   |  3.14  |   'S1'   |
|  12   |     6      |  U8  | bool |  4   |  8   |  3.14  |   'S1'   |
|  13   |     6      |  U8  | bool | 128  |  8   |  3.14  |   'S1'   |
|  14   |     7      |  U8  | F32  |  4   |  8   |  3.14  |   'S1'   |
|  15   |     7      |  U8  | F32  | 128  |  8   |  3.14  |   'S1'   |
|  16   |     8      |  U8  | F64  |  4   |  8   |  3.14  |   'S1'   |
|  17   |     8      |  U8  | F64  | 128  |  8   |  3.14  |   'S1'   |
)expected";

  {
    nvbench::option_parser parser;
    parser.parse(
      {"--benchmark", "TestBench", "--axis", " Ints [ pow2 ] = [ 2 , 7 ] "});
    const auto test = parser_to_state_string(parser);
    ASSERT_MSG(test == ref, "Expected:\n\"{}\"\n\nActual:\n\"{}\"", ref, test);
  }

  {
    nvbench::option_parser parser;
    parser.parse({"--benchmark", "TestBench", "--axis", "Ints[pow2]=[2,7]"});
    const auto test = parser_to_state_string(parser);
    ASSERT_MSG(test == ref, "Expected:\n\"{}\"\n\nActual:\n\"{}\"", ref, test);
  }

  {
    nvbench::option_parser parser;
    parser.parse(
      {"--benchmark", "TestBench", "--axis", " Ints [ pow2 ] = [ 2 : 7 : 5 ] "});
    const auto test = parser_to_state_string(parser);
    ASSERT_MSG(test == ref, "Expected:\n\"{}\"\n\nActual:\n\"{}\"", ref, test);
  }

  {
    nvbench::option_parser parser;
    parser.parse({"--benchmark", "TestBench", "--axis", "Ints[pow2]=[2:7:5]"});
    const auto test = parser_to_state_string(parser);
    ASSERT_MSG(test == ref, "Expected:\n\"{}\"\n\nActual:\n\"{}\"", ref, test);
  }
}

void test_int64_axis_pow2_to_none_single()
{
  const std::string ref =
    R"expected(
| State | TypeConfig |  T   |  U   | Ints | PO2s | Floats | Strings  |
|   0   |     0      | void | bool |  42  |  2   |  3.14  |   'S1'   |
|   1   |     1      | void | F32  |  42  |  2   |  3.14  |   'S1'   |
|   2   |     2      | void | F64  |  42  |  2   |  3.14  |   'S1'   |
|   3   |     3      |  I8  | bool |  42  |  2   |  3.14  |   'S1'   |
|   4   |     4      |  I8  | F32  |  42  |  2   |  3.14  |   'S1'   |
|   5   |     5      |  I8  | F64  |  42  |  2   |  3.14  |   'S1'   |
|   6   |     6      |  U8  | bool |  42  |  2   |  3.14  |   'S1'   |
|   7   |     7      |  U8  | F32  |  42  |  2   |  3.14  |   'S1'   |
|   8   |     8      |  U8  | F64  |  42  |  2   |  3.14  |   'S1'   |
)expected";

  {
    nvbench::option_parser parser;
    parser.parse({"--benchmark", "TestBench", "--axis", " PO2s [ ] = 2 "});
    const auto test = parser_to_state_string(parser);
    ASSERT_MSG(test == ref, "Expected:\n\"{}\"\n\nActual:\n\"{}\"", ref, test);
  }

  {
    nvbench::option_parser parser;
    parser.parse({"--benchmark", "TestBench", "--axis", "PO2s=2"});
    const auto test = parser_to_state_string(parser);
    ASSERT_MSG(test == ref, "Expected:\n\"{}\"\n\nActual:\n\"{}\"", ref, test);
  }

  {
    nvbench::option_parser parser;
    parser.parse({"--benchmark", "TestBench", "--axis", " PO2s [ ] = [ 2 ] "});
    const auto test = parser_to_state_string(parser);
    ASSERT_MSG(test == ref, "Expected:\n\"{}\"\n\nActual:\n\"{}\"", ref, test);
  }

  {
    nvbench::option_parser parser;
    parser.parse({"--benchmark", "TestBench", "--axis", "PO2s=[2]"});
    const auto test = parser_to_state_string(parser);
    ASSERT_MSG(test == ref, "Expected:\n\"{}\"\n\nActual:\n\"{}\"", ref, test);
  }

  {
    nvbench::option_parser parser;
    parser.parse(
      {"--benchmark", "TestBench", "--axis", " PO2s [ ] = [ 2 : 2 : 1 ] "});
    const auto test = parser_to_state_string(parser);
    ASSERT_MSG(test == ref, "Expected:\n\"{}\"\n\nActual:\n\"{}\"", ref, test);
  }

  {
    nvbench::option_parser parser;
    parser.parse({"--benchmark", "TestBench", "--axis", "PO2s=[2:2]"});
    const auto test = parser_to_state_string(parser);
    ASSERT_MSG(test == ref, "Expected:\n\"{}\"\n\nActual:\n\"{}\"", ref, test);
  }
}

void test_int64_axis_pow2_to_none_multi()
{
  const std::string ref =
    R"expected(
| State | TypeConfig |  T   |  U   | Ints | PO2s | Floats | Strings  |
|   0   |     0      | void | bool |  42  |  2   |  3.14  |   'S1'   |
|   1   |     0      | void | bool |  42  |  7   |  3.14  |   'S1'   |
|   2   |     1      | void | F32  |  42  |  2   |  3.14  |   'S1'   |
|   3   |     1      | void | F32  |  42  |  7   |  3.14  |   'S1'   |
|   4   |     2      | void | F64  |  42  |  2   |  3.14  |   'S1'   |
|   5   |     2      | void | F64  |  42  |  7   |  3.14  |   'S1'   |
|   6   |     3      |  I8  | bool |  42  |  2   |  3.14  |   'S1'   |
|   7   |     3      |  I8  | bool |  42  |  7   |  3.14  |   'S1'   |
|   8   |     4      |  I8  | F32  |  42  |  2   |  3.14  |   'S1'   |
|   9   |     4      |  I8  | F32  |  42  |  7   |  3.14  |   'S1'   |
|  10   |     5      |  I8  | F64  |  42  |  2   |  3.14  |   'S1'   |
|  11   |     5      |  I8  | F64  |  42  |  7   |  3.14  |   'S1'   |
|  12   |     6      |  U8  | bool |  42  |  2   |  3.14  |   'S1'   |
|  13   |     6      |  U8  | bool |  42  |  7   |  3.14  |   'S1'   |
|  14   |     7      |  U8  | F32  |  42  |  2   |  3.14  |   'S1'   |
|  15   |     7      |  U8  | F32  |  42  |  7   |  3.14  |   'S1'   |
|  16   |     8      |  U8  | F64  |  42  |  2   |  3.14  |   'S1'   |
|  17   |     8      |  U8  | F64  |  42  |  7   |  3.14  |   'S1'   |
)expected";

  {
    nvbench::option_parser parser;
    parser.parse(
      {"--benchmark", "TestBench", "--axis", " PO2s [ ] = [ 2 , 7 ] "});
    const auto test = parser_to_state_string(parser);
    ASSERT_MSG(test == ref, "Expected:\n\"{}\"\n\nActual:\n\"{}\"", ref, test);
  }

  {
    nvbench::option_parser parser;
    parser.parse({"--benchmark", "TestBench", "--axis", "PO2s=[2,7]"});
    const auto test = parser_to_state_string(parser);
    ASSERT_MSG(test == ref, "Expected:\n\"{}\"\n\nActual:\n\"{}\"", ref, test);
  }

  {
    nvbench::option_parser parser;
    parser.parse(
      {"--benchmark", "TestBench", "--axis", " PO2s [ ] = [ 2 : 7 : 5 ] "});
    const auto test = parser_to_state_string(parser);
    ASSERT_MSG(test == ref, "Expected:\n\"{}\"\n\nActual:\n\"{}\"", ref, test);
  }

  {
    nvbench::option_parser parser;
    parser.parse({"--benchmark", "TestBench", "--axis", "PO2s=[2:7:5]"});
    const auto test = parser_to_state_string(parser);
    ASSERT_MSG(test == ref, "Expected:\n\"{}\"\n\nActual:\n\"{}\"", ref, test);
  }
}

void test_float64_axis_single()
{
  const std::string ref =
    R"expected(
| State | TypeConfig |  T   |  U   | Ints | PO2s | Floats | Strings  |
|   0   |     0      | void | bool |  42  |  8   |  3.5   |   'S1'   |
|   1   |     1      | void | F32  |  42  |  8   |  3.5   |   'S1'   |
|   2   |     2      | void | F64  |  42  |  8   |  3.5   |   'S1'   |
|   3   |     3      |  I8  | bool |  42  |  8   |  3.5   |   'S1'   |
|   4   |     4      |  I8  | F32  |  42  |  8   |  3.5   |   'S1'   |
|   5   |     5      |  I8  | F64  |  42  |  8   |  3.5   |   'S1'   |
|   6   |     6      |  U8  | bool |  42  |  8   |  3.5   |   'S1'   |
|   7   |     7      |  U8  | F32  |  42  |  8   |  3.5   |   'S1'   |
|   8   |     8      |  U8  | F64  |  42  |  8   |  3.5   |   'S1'   |
)expected";

  {
    nvbench::option_parser parser;
    parser.parse({"--benchmark", "TestBench", "--axis", " Floats [ ] = 3.5 "});
    const auto test = parser_to_state_string(parser);
    ASSERT_MSG(test == ref, "Expected:\n\"{}\"\n\nActual:\n\"{}\"", ref, test);
  }

  {
    nvbench::option_parser parser;
    parser.parse({"--benchmark", "TestBench", "--axis", "Floats=3.5"});
    const auto test = parser_to_state_string(parser);
    ASSERT_MSG(test == ref, "Expected:\n\"{}\"\n\nActual:\n\"{}\"", ref, test);
  }

  {
    nvbench::option_parser parser;
    parser.parse(
      {"--benchmark", "TestBench", "--axis", " Floats [ ] = [ 3.5 ] "});
    const auto test = parser_to_state_string(parser);
    ASSERT_MSG(test == ref, "Expected:\n\"{}\"\n\nActual:\n\"{}\"", ref, test);
  }

  {
    nvbench::option_parser parser;
    parser.parse({"--benchmark", "TestBench", "--axis", "Floats=[3.5]"});
    const auto test = parser_to_state_string(parser);
    ASSERT_MSG(test == ref, "Expected:\n\"{}\"\n\nActual:\n\"{}\"", ref, test);
  }

  {
    nvbench::option_parser parser;
    parser.parse({"--benchmark",
                  "TestBench",
                  "--axis",
                  " Floats [ ] = [ 3.5 : 3.6 : 1 ] "});
    const auto test = parser_to_state_string(parser);
    ASSERT_MSG(test == ref, "Expected:\n\"{}\"\n\nActual:\n\"{}\"", ref, test);
  }

  {
    nvbench::option_parser parser;
    parser.parse({"--benchmark", "TestBench", "--axis", "Floats=[3.5:3.6]"});
    const auto test = parser_to_state_string(parser);
    ASSERT_MSG(test == ref, "Expected:\n\"{}\"\n\nActual:\n\"{}\"", ref, test);
  }
}

void test_float64_axis_multi()
{
  const std::string ref =
    R"expected(
| State | TypeConfig |  T   |  U   | Ints | PO2s | Floats | Strings  |
|   0   |     0      | void | bool |  42  |  8   |  3.5   |   'S1'   |
|   1   |     0      | void | bool |  42  |  8   |  4.1   |   'S1'   |
|   2   |     1      | void | F32  |  42  |  8   |  3.5   |   'S1'   |
|   3   |     1      | void | F32  |  42  |  8   |  4.1   |   'S1'   |
|   4   |     2      | void | F64  |  42  |  8   |  3.5   |   'S1'   |
|   5   |     2      | void | F64  |  42  |  8   |  4.1   |   'S1'   |
|   6   |     3      |  I8  | bool |  42  |  8   |  3.5   |   'S1'   |
|   7   |     3      |  I8  | bool |  42  |  8   |  4.1   |   'S1'   |
|   8   |     4      |  I8  | F32  |  42  |  8   |  3.5   |   'S1'   |
|   9   |     4      |  I8  | F32  |  42  |  8   |  4.1   |   'S1'   |
|  10   |     5      |  I8  | F64  |  42  |  8   |  3.5   |   'S1'   |
|  11   |     5      |  I8  | F64  |  42  |  8   |  4.1   |   'S1'   |
|  12   |     6      |  U8  | bool |  42  |  8   |  3.5   |   'S1'   |
|  13   |     6      |  U8  | bool |  42  |  8   |  4.1   |   'S1'   |
|  14   |     7      |  U8  | F32  |  42  |  8   |  3.5   |   'S1'   |
|  15   |     7      |  U8  | F32  |  42  |  8   |  4.1   |   'S1'   |
|  16   |     8      |  U8  | F64  |  42  |  8   |  3.5   |   'S1'   |
|  17   |     8      |  U8  | F64  |  42  |  8   |  4.1   |   'S1'   |
)expected";

  {
    nvbench::option_parser parser;
    parser.parse(
      {"--benchmark", "TestBench", "--axis", " Floats [ ] = [ 3.5 , 4.1 ] "});
    const auto test = parser_to_state_string(parser);
    ASSERT_MSG(test == ref, "Expected:\n\"{}\"\n\nActual:\n\"{}\"", ref, test);
  }

  {
    nvbench::option_parser parser;
    parser.parse({"--benchmark", "TestBench", "--axis", "Floats=[3.5,4.1]"});
    const auto test = parser_to_state_string(parser);
    ASSERT_MSG(test == ref, "Expected:\n\"{}\"\n\nActual:\n\"{}\"", ref, test);
  }

  {
    nvbench::option_parser parser;
    parser.parse({"--benchmark",
                  "TestBench",
                  "--axis",
                  " Floats [ ] = [ 3.5 : 4.2 : 0.6 ] "});
    const auto test = parser_to_state_string(parser);
    ASSERT_MSG(test == ref, "Expected:\n\"{}\"\n\nActual:\n\"{}\"", ref, test);
  }

  {
    nvbench::option_parser parser;
    parser.parse(
      {"--benchmark", "TestBench", "--axis", "Floats=[3.5:4.2:0.6]"});
    const auto test = parser_to_state_string(parser);
    ASSERT_MSG(test == ref, "Expected:\n\"{}\"\n\nActual:\n\"{}\"", ref, test);
  }
}

void test_string_axis_single()
{
  const std::string ref =
    R"expected(
| State | TypeConfig |  T   |  U   | Ints | PO2s | Floats | Strings  |
|   0   |     0      | void | bool |  42  |  8   |  3.14  | 'fo br'  |
|   1   |     1      | void | F32  |  42  |  8   |  3.14  | 'fo br'  |
|   2   |     2      | void | F64  |  42  |  8   |  3.14  | 'fo br'  |
|   3   |     3      |  I8  | bool |  42  |  8   |  3.14  | 'fo br'  |
|   4   |     4      |  I8  | F32  |  42  |  8   |  3.14  | 'fo br'  |
|   5   |     5      |  I8  | F64  |  42  |  8   |  3.14  | 'fo br'  |
|   6   |     6      |  U8  | bool |  42  |  8   |  3.14  | 'fo br'  |
|   7   |     7      |  U8  | F32  |  42  |  8   |  3.14  | 'fo br'  |
|   8   |     8      |  U8  | F64  |  42  |  8   |  3.14  | 'fo br'  |
)expected";

  {
    nvbench::option_parser parser;
    parser.parse(
      {"--benchmark", "TestBench", "--axis", " Strings [ ] = fo br "});
    const auto test = parser_to_state_string(parser);
    ASSERT_MSG(test == ref, "Expected:\n\"{}\"\n\nActual:\n\"{}\"", ref, test);
  }

  {
    nvbench::option_parser parser;
    parser.parse({"--benchmark", "TestBench", "--axis", "Strings=fo br"});
    const auto test = parser_to_state_string(parser);
    ASSERT_MSG(test == ref, "Expected:\n\"{}\"\n\nActual:\n\"{}\"", ref, test);
  }

  {
    nvbench::option_parser parser;
    parser.parse(
      {"--benchmark", "TestBench", "--axis", " Strings [ ] = [ fo br ] "});
    const auto test = parser_to_state_string(parser);
    ASSERT_MSG(test == ref, "Expected:\n\"{}\"\n\nActual:\n\"{}\"", ref, test);
  }

  {
    nvbench::option_parser parser;
    parser.parse({"--benchmark", "TestBench", "--axis", "Strings=[fo br]"});
    const auto test = parser_to_state_string(parser);
    ASSERT_MSG(test == ref, "Expected:\n\"{}\"\n\nActual:\n\"{}\"", ref, test);
  }
}

void test_string_axis_multi()
{
  const std::string ref =
    R"expected(
| State | TypeConfig |  T   |  U   | Ints | PO2s | Floats | Strings  |
|   0   |     0      | void | bool |  42  |  8   |  3.14  | 'fo br'  |
|   1   |     0      | void | bool |  42  |  8   |  3.14  |  'baz'   |
|   2   |     1      | void | F32  |  42  |  8   |  3.14  | 'fo br'  |
|   3   |     1      | void | F32  |  42  |  8   |  3.14  |  'baz'   |
|   4   |     2      | void | F64  |  42  |  8   |  3.14  | 'fo br'  |
|   5   |     2      | void | F64  |  42  |  8   |  3.14  |  'baz'   |
|   6   |     3      |  I8  | bool |  42  |  8   |  3.14  | 'fo br'  |
|   7   |     3      |  I8  | bool |  42  |  8   |  3.14  |  'baz'   |
|   8   |     4      |  I8  | F32  |  42  |  8   |  3.14  | 'fo br'  |
|   9   |     4      |  I8  | F32  |  42  |  8   |  3.14  |  'baz'   |
|  10   |     5      |  I8  | F64  |  42  |  8   |  3.14  | 'fo br'  |
|  11   |     5      |  I8  | F64  |  42  |  8   |  3.14  |  'baz'   |
|  12   |     6      |  U8  | bool |  42  |  8   |  3.14  | 'fo br'  |
|  13   |     6      |  U8  | bool |  42  |  8   |  3.14  |  'baz'   |
|  14   |     7      |  U8  | F32  |  42  |  8   |  3.14  | 'fo br'  |
|  15   |     7      |  U8  | F32  |  42  |  8   |  3.14  |  'baz'   |
|  16   |     8      |  U8  | F64  |  42  |  8   |  3.14  | 'fo br'  |
|  17   |     8      |  U8  | F64  |  42  |  8   |  3.14  |  'baz'   |
)expected";

  {
    nvbench::option_parser parser;
    parser.parse(
      {"--benchmark", "TestBench", "--axis", " Strings [ ] = [ fo br , baz ] "});
    const auto test = parser_to_state_string(parser);
    ASSERT_MSG(test == ref, "Expected:\n\"{}\"\n\nActual:\n\"{}\"", ref, test);
  }

  {
    nvbench::option_parser parser;
    parser.parse({"--benchmark", "TestBench", "--axis", "Strings=[fo br,baz]"});
    const auto test = parser_to_state_string(parser);
    ASSERT_MSG(test == ref, "Expected:\n\"{}\"\n\nActual:\n\"{}\"", ref, test);
  }
}

void test_type_axis_single()
{
  const std::string ref =
    R"expected(
| State | TypeConfig |  T   |  U   | Ints | PO2s | Floats | Strings  |
|   0   |     6      |  U8  | bool |  42  |  8   |  3.14  |   'S1'   |
|   1   |     7      |  U8  | F32  |  42  |  8   |  3.14  |   'S1'   |
|   2   |     8      |  U8  | F64  |  42  |  8   |  3.14  |   'S1'   |
)expected";

  {
    nvbench::option_parser parser;
    parser.parse({"--benchmark", "TestBench", "--axis", " T [ ] = U8 "});
    const auto test = parser_to_state_string(parser);
    ASSERT_MSG(test == ref, "Expected:\n\"{}\"\n\nActual:\n\"{}\"", ref, test);
  }

  {
    nvbench::option_parser parser;
    parser.parse({"--benchmark", "TestBench", "--axis", "T=U8"});
    const auto test = parser_to_state_string(parser);
    ASSERT_MSG(test == ref, "Expected:\n\"{}\"\n\nActual:\n\"{}\"", ref, test);
  }

  {
    nvbench::option_parser parser;
    parser.parse({"--benchmark", "TestBench", "--axis", " T [ ] = [ U8 ] "});
    const auto test = parser_to_state_string(parser);
    ASSERT_MSG(test == ref, "Expected:\n\"{}\"\n\nActual:\n\"{}\"", ref, test);
  }

  {
    nvbench::option_parser parser;
    parser.parse({"--benchmark", "TestBench", "--axis", "T=[U8]"});
    const auto test = parser_to_state_string(parser);
    ASSERT_MSG(test == ref, "Expected:\n\"{}\"\n\nActual:\n\"{}\"", ref, test);
  }
}

void test_type_axis_multi()
{
  const std::string ref =
    R"expected(
| State | TypeConfig |  T   |  U   | Ints | PO2s | Floats | Strings  |
|   0   |     0      | void | bool |  42  |  8   |  3.14  |   'S1'   |
|   1   |     1      | void | F32  |  42  |  8   |  3.14  |   'S1'   |
|   2   |     2      | void | F64  |  42  |  8   |  3.14  |   'S1'   |
|   3   |     6      |  U8  | bool |  42  |  8   |  3.14  |   'S1'   |
|   4   |     7      |  U8  | F32  |  42  |  8   |  3.14  |   'S1'   |
|   5   |     8      |  U8  | F64  |  42  |  8   |  3.14  |   'S1'   |
)expected";

  {
    nvbench::option_parser parser;
    parser.parse(
      {"--benchmark", "TestBench", "--axis", " T [ ] = [ U8, void ] "});
    const auto test = parser_to_state_string(parser);
    ASSERT_MSG(test == ref, "Expected:\n\"{}\"\n\nActual:\n\"{}\"", ref, test);
  }

  {
    nvbench::option_parser parser;
    parser.parse({"--benchmark", "TestBench", "--axis", "T=[void,U8]"});
    const auto test = parser_to_state_string(parser);
    ASSERT_MSG(test == ref, "Expected:\n\"{}\"\n\nActual:\n\"{}\"", ref, test);
  }
}

void test_multi_axis()
{

  const std::string ref =
    R"expected(
| State | TypeConfig |  T   |  U   | Ints | PO2s | Floats | Strings  |
|   0   |     0      | void | bool |  2   |  4   |  0.25  |  'foo'   |
|   1   |     0      | void | bool |  5   |  4   |  0.25  |  'foo'   |
|   2   |     0      | void | bool |  2   |  32  |  0.25  |  'foo'   |
|   3   |     0      | void | bool |  5   |  32  |  0.25  |  'foo'   |
|   4   |     0      | void | bool |  2   | 256  |  0.25  |  'foo'   |
|   5   |     0      | void | bool |  5   | 256  |  0.25  |  'foo'   |
|   6   |     0      | void | bool |  2   |  4   |  0.5   |  'foo'   |
|   7   |     0      | void | bool |  5   |  4   |  0.5   |  'foo'   |
|   8   |     0      | void | bool |  2   |  32  |  0.5   |  'foo'   |
|   9   |     0      | void | bool |  5   |  32  |  0.5   |  'foo'   |
|  10   |     0      | void | bool |  2   | 256  |  0.5   |  'foo'   |
|  11   |     0      | void | bool |  5   | 256  |  0.5   |  'foo'   |
|  12   |     0      | void | bool |  2   |  4   |  0.75  |  'foo'   |
|  13   |     0      | void | bool |  5   |  4   |  0.75  |  'foo'   |
|  14   |     0      | void | bool |  2   |  32  |  0.75  |  'foo'   |
|  15   |     0      | void | bool |  5   |  32  |  0.75  |  'foo'   |
|  16   |     0      | void | bool |  2   | 256  |  0.75  |  'foo'   |
|  17   |     0      | void | bool |  5   | 256  |  0.75  |  'foo'   |
|  18   |     0      | void | bool |  2   |  4   |   1    |  'foo'   |
|  19   |     0      | void | bool |  5   |  4   |   1    |  'foo'   |
|  20   |     0      | void | bool |  2   |  32  |   1    |  'foo'   |
|  21   |     0      | void | bool |  5   |  32  |   1    |  'foo'   |
|  22   |     0      | void | bool |  2   | 256  |   1    |  'foo'   |
|  23   |     0      | void | bool |  5   | 256  |   1    |  'foo'   |
|  24   |     0      | void | bool |  2   |  4   |  0.25  |  'bar'   |
|  25   |     0      | void | bool |  5   |  4   |  0.25  |  'bar'   |
|  26   |     0      | void | bool |  2   |  32  |  0.25  |  'bar'   |
|  27   |     0      | void | bool |  5   |  32  |  0.25  |  'bar'   |
|  28   |     0      | void | bool |  2   | 256  |  0.25  |  'bar'   |
|  29   |     0      | void | bool |  5   | 256  |  0.25  |  'bar'   |
|  30   |     0      | void | bool |  2   |  4   |  0.5   |  'bar'   |
|  31   |     0      | void | bool |  5   |  4   |  0.5   |  'bar'   |
|  32   |     0      | void | bool |  2   |  32  |  0.5   |  'bar'   |
|  33   |     0      | void | bool |  5   |  32  |  0.5   |  'bar'   |
|  34   |     0      | void | bool |  2   | 256  |  0.5   |  'bar'   |
|  35   |     0      | void | bool |  5   | 256  |  0.5   |  'bar'   |
|  36   |     0      | void | bool |  2   |  4   |  0.75  |  'bar'   |
|  37   |     0      | void | bool |  5   |  4   |  0.75  |  'bar'   |
|  38   |     0      | void | bool |  2   |  32  |  0.75  |  'bar'   |
|  39   |     0      | void | bool |  5   |  32  |  0.75  |  'bar'   |
|  40   |     0      | void | bool |  2   | 256  |  0.75  |  'bar'   |
|  41   |     0      | void | bool |  5   | 256  |  0.75  |  'bar'   |
|  42   |     0      | void | bool |  2   |  4   |   1    |  'bar'   |
|  43   |     0      | void | bool |  5   |  4   |   1    |  'bar'   |
|  44   |     0      | void | bool |  2   |  32  |   1    |  'bar'   |
|  45   |     0      | void | bool |  5   |  32  |   1    |  'bar'   |
|  46   |     0      | void | bool |  2   | 256  |   1    |  'bar'   |
|  47   |     0      | void | bool |  5   | 256  |   1    |  'bar'   |
|  48   |     0      | void | bool |  2   |  4   |  0.25  |  'baz'   |
|  49   |     0      | void | bool |  5   |  4   |  0.25  |  'baz'   |
|  50   |     0      | void | bool |  2   |  32  |  0.25  |  'baz'   |
|  51   |     0      | void | bool |  5   |  32  |  0.25  |  'baz'   |
|  52   |     0      | void | bool |  2   | 256  |  0.25  |  'baz'   |
|  53   |     0      | void | bool |  5   | 256  |  0.25  |  'baz'   |
|  54   |     0      | void | bool |  2   |  4   |  0.5   |  'baz'   |
|  55   |     0      | void | bool |  5   |  4   |  0.5   |  'baz'   |
|  56   |     0      | void | bool |  2   |  32  |  0.5   |  'baz'   |
|  57   |     0      | void | bool |  5   |  32  |  0.5   |  'baz'   |
|  58   |     0      | void | bool |  2   | 256  |  0.5   |  'baz'   |
|  59   |     0      | void | bool |  5   | 256  |  0.5   |  'baz'   |
|  60   |     0      | void | bool |  2   |  4   |  0.75  |  'baz'   |
|  61   |     0      | void | bool |  5   |  4   |  0.75  |  'baz'   |
|  62   |     0      | void | bool |  2   |  32  |  0.75  |  'baz'   |
|  63   |     0      | void | bool |  5   |  32  |  0.75  |  'baz'   |
|  64   |     0      | void | bool |  2   | 256  |  0.75  |  'baz'   |
|  65   |     0      | void | bool |  5   | 256  |  0.75  |  'baz'   |
|  66   |     0      | void | bool |  2   |  4   |   1    |  'baz'   |
|  67   |     0      | void | bool |  5   |  4   |   1    |  'baz'   |
|  68   |     0      | void | bool |  2   |  32  |   1    |  'baz'   |
|  69   |     0      | void | bool |  5   |  32  |   1    |  'baz'   |
|  70   |     0      | void | bool |  2   | 256  |   1    |  'baz'   |
|  71   |     0      | void | bool |  5   | 256  |   1    |  'baz'   |
|  72   |     6      |  U8  | bool |  2   |  4   |  0.25  |  'foo'   |
|  73   |     6      |  U8  | bool |  5   |  4   |  0.25  |  'foo'   |
|  74   |     6      |  U8  | bool |  2   |  32  |  0.25  |  'foo'   |
|  75   |     6      |  U8  | bool |  5   |  32  |  0.25  |  'foo'   |
|  76   |     6      |  U8  | bool |  2   | 256  |  0.25  |  'foo'   |
|  77   |     6      |  U8  | bool |  5   | 256  |  0.25  |  'foo'   |
|  78   |     6      |  U8  | bool |  2   |  4   |  0.5   |  'foo'   |
|  79   |     6      |  U8  | bool |  5   |  4   |  0.5   |  'foo'   |
|  80   |     6      |  U8  | bool |  2   |  32  |  0.5   |  'foo'   |
|  81   |     6      |  U8  | bool |  5   |  32  |  0.5   |  'foo'   |
|  82   |     6      |  U8  | bool |  2   | 256  |  0.5   |  'foo'   |
|  83   |     6      |  U8  | bool |  5   | 256  |  0.5   |  'foo'   |
|  84   |     6      |  U8  | bool |  2   |  4   |  0.75  |  'foo'   |
|  85   |     6      |  U8  | bool |  5   |  4   |  0.75  |  'foo'   |
|  86   |     6      |  U8  | bool |  2   |  32  |  0.75  |  'foo'   |
|  87   |     6      |  U8  | bool |  5   |  32  |  0.75  |  'foo'   |
|  88   |     6      |  U8  | bool |  2   | 256  |  0.75  |  'foo'   |
|  89   |     6      |  U8  | bool |  5   | 256  |  0.75  |  'foo'   |
|  90   |     6      |  U8  | bool |  2   |  4   |   1    |  'foo'   |
|  91   |     6      |  U8  | bool |  5   |  4   |   1    |  'foo'   |
|  92   |     6      |  U8  | bool |  2   |  32  |   1    |  'foo'   |
|  93   |     6      |  U8  | bool |  5   |  32  |   1    |  'foo'   |
|  94   |     6      |  U8  | bool |  2   | 256  |   1    |  'foo'   |
|  95   |     6      |  U8  | bool |  5   | 256  |   1    |  'foo'   |
|  96   |     6      |  U8  | bool |  2   |  4   |  0.25  |  'bar'   |
|  97   |     6      |  U8  | bool |  5   |  4   |  0.25  |  'bar'   |
|  98   |     6      |  U8  | bool |  2   |  32  |  0.25  |  'bar'   |
|  99   |     6      |  U8  | bool |  5   |  32  |  0.25  |  'bar'   |
|  100  |     6      |  U8  | bool |  2   | 256  |  0.25  |  'bar'   |
|  101  |     6      |  U8  | bool |  5   | 256  |  0.25  |  'bar'   |
|  102  |     6      |  U8  | bool |  2   |  4   |  0.5   |  'bar'   |
|  103  |     6      |  U8  | bool |  5   |  4   |  0.5   |  'bar'   |
|  104  |     6      |  U8  | bool |  2   |  32  |  0.5   |  'bar'   |
|  105  |     6      |  U8  | bool |  5   |  32  |  0.5   |  'bar'   |
|  106  |     6      |  U8  | bool |  2   | 256  |  0.5   |  'bar'   |
|  107  |     6      |  U8  | bool |  5   | 256  |  0.5   |  'bar'   |
|  108  |     6      |  U8  | bool |  2   |  4   |  0.75  |  'bar'   |
|  109  |     6      |  U8  | bool |  5   |  4   |  0.75  |  'bar'   |
|  110  |     6      |  U8  | bool |  2   |  32  |  0.75  |  'bar'   |
|  111  |     6      |  U8  | bool |  5   |  32  |  0.75  |  'bar'   |
|  112  |     6      |  U8  | bool |  2   | 256  |  0.75  |  'bar'   |
|  113  |     6      |  U8  | bool |  5   | 256  |  0.75  |  'bar'   |
|  114  |     6      |  U8  | bool |  2   |  4   |   1    |  'bar'   |
|  115  |     6      |  U8  | bool |  5   |  4   |   1    |  'bar'   |
|  116  |     6      |  U8  | bool |  2   |  32  |   1    |  'bar'   |
|  117  |     6      |  U8  | bool |  5   |  32  |   1    |  'bar'   |
|  118  |     6      |  U8  | bool |  2   | 256  |   1    |  'bar'   |
|  119  |     6      |  U8  | bool |  5   | 256  |   1    |  'bar'   |
|  120  |     6      |  U8  | bool |  2   |  4   |  0.25  |  'baz'   |
|  121  |     6      |  U8  | bool |  5   |  4   |  0.25  |  'baz'   |
|  122  |     6      |  U8  | bool |  2   |  32  |  0.25  |  'baz'   |
|  123  |     6      |  U8  | bool |  5   |  32  |  0.25  |  'baz'   |
|  124  |     6      |  U8  | bool |  2   | 256  |  0.25  |  'baz'   |
|  125  |     6      |  U8  | bool |  5   | 256  |  0.25  |  'baz'   |
|  126  |     6      |  U8  | bool |  2   |  4   |  0.5   |  'baz'   |
|  127  |     6      |  U8  | bool |  5   |  4   |  0.5   |  'baz'   |
|  128  |     6      |  U8  | bool |  2   |  32  |  0.5   |  'baz'   |
|  129  |     6      |  U8  | bool |  5   |  32  |  0.5   |  'baz'   |
|  130  |     6      |  U8  | bool |  2   | 256  |  0.5   |  'baz'   |
|  131  |     6      |  U8  | bool |  5   | 256  |  0.5   |  'baz'   |
|  132  |     6      |  U8  | bool |  2   |  4   |  0.75  |  'baz'   |
|  133  |     6      |  U8  | bool |  5   |  4   |  0.75  |  'baz'   |
|  134  |     6      |  U8  | bool |  2   |  32  |  0.75  |  'baz'   |
|  135  |     6      |  U8  | bool |  5   |  32  |  0.75  |  'baz'   |
|  136  |     6      |  U8  | bool |  2   | 256  |  0.75  |  'baz'   |
|  137  |     6      |  U8  | bool |  5   | 256  |  0.75  |  'baz'   |
|  138  |     6      |  U8  | bool |  2   |  4   |   1    |  'baz'   |
|  139  |     6      |  U8  | bool |  5   |  4   |   1    |  'baz'   |
|  140  |     6      |  U8  | bool |  2   |  32  |   1    |  'baz'   |
|  141  |     6      |  U8  | bool |  5   |  32  |   1    |  'baz'   |
|  142  |     6      |  U8  | bool |  2   | 256  |   1    |  'baz'   |
|  143  |     6      |  U8  | bool |  5   | 256  |   1    |  'baz'   |
)expected";

  {
    nvbench::option_parser parser;
    parser.parse({
      // clang-format off
      "--benchmark", "TestBench",
      "--axis", "T=[U8,void]",
      "--axis", "U=bool",
      "--axis", "Ints=[2:6:3]",
      "--axis", "PO2s[pow2]=[2:10:3]",
      "--axis", "Floats=[0.25:1:0.25]",
      "--axis", "Strings=[foo,bar,baz]",
      // clang-format on
    });
    const auto test = parser_to_state_string(parser);
    ASSERT_MSG(test == ref, "Expected:\n\"{}\"\n\nActual:\n\"{}\"", ref, test);
  }

  {
    nvbench::option_parser parser;
    parser.parse({
      // clang-format off
      "-b", "TestBench",
      "-a", "Strings=[foo,bar,baz]",
      "-a", "U=bool",
      "-a", "Floats=[0.25:1:0.25]",
      "-a", "Ints=[2:6:3]",
      "-a", "PO2s[pow2]=[2:10:3]",
      "-a", "T=[U8,void]",
      // clang-format on
    });
    const auto test = parser_to_state_string(parser);
    ASSERT_MSG(test == ref, "Expected:\n\"{}\"\n\nActual:\n\"{}\"", ref, test);
  }
}

// `--axis` affects the last `--benchmark`. An exception is thrown if there is
// no benchmark specified for an axis.
void test_axis_before_benchmark()
{
  {
    nvbench::option_parser parser;
    ASSERT_THROWS_ANY(parser.parse({"--axis", "--benchmark"}));
  }
  {
    nvbench::option_parser parser;
    ASSERT_THROWS_ANY(parser.parse({"--axis", "-b"}));
  }
  {
    nvbench::option_parser parser;
    ASSERT_THROWS_ANY(parser.parse({"-a", "--benchmark"}));
  }
  {
    nvbench::option_parser parser;
    ASSERT_THROWS_ANY(parser.parse({"-a", "-b"}));
  }
}

void test_min_samples()
{
  nvbench::option_parser parser;
  parser.parse(
    {"--benchmark", "DummyBench", "--min-samples", "12345"});
  const auto& states = parser_to_states(parser);

  ASSERT(states.size() == 1);
  ASSERT(states[0].get_min_samples() == 12345);
}

void test_min_time()
{
  nvbench::option_parser parser;
  parser.parse(
    {"--benchmark", "DummyBench", "--min-time", "12345e2"});
  const auto& states = parser_to_states(parser);

  ASSERT(states.size() == 1);
  ASSERT(std::abs(states[0].get_min_time() - 12345e2) < 1.);
}

void test_max_noise()
{
  nvbench::option_parser parser;
  parser.parse(
    {"--benchmark", "DummyBench", "--max-noise", "50.3"});
  const auto& states = parser_to_states(parser);

  ASSERT(states.size() == 1);
  ASSERT(std::abs(states[0].get_max_noise() - 0.503) < 1.e-4);
}

void test_skip_time()
{
  nvbench::option_parser parser;
  parser.parse(
    {"--benchmark", "DummyBench", "--skip-time", "12345e2"});
  const auto& states = parser_to_states(parser);

  ASSERT(states.size() == 1);
  ASSERT(std::abs(states[0].get_skip_time() - 12345e2) < 1.);
}

void test_timeout()
{
  nvbench::option_parser parser;
  parser.parse(
    {"--benchmark", "DummyBench", "--timeout", "12345e2"});
  const auto& states = parser_to_states(parser);

  ASSERT(states.size() == 1);
  ASSERT(std::abs(states[0].get_timeout() - 12345e2) < 1.);
}

void test_stopping_criterion()
{
  nvbench::option_parser parser;
  parser.parse(
    {"--benchmark", "DummyBench", 
     "--stopping-criterion", "entropy",
     "--max-angle", "0.42",
     "--min-r2", "0.6"});
  const auto& states = parser_to_states(parser);

  ASSERT(states.size() == 1);
  ASSERT(states[0].get_stopping_criterion() == "entropy");

  const nvbench::criterion_params &criterion_params = states[0].get_criterion_params();
  ASSERT(criterion_params.has_value("max-angle"));
  ASSERT(criterion_params.has_value("min-r2"));

  ASSERT(criterion_params.get_float64("max-angle") == 0.42);
  ASSERT(criterion_params.get_float64("min-r2") == 0.6);
}

int main()
try
{
  test_empty();
  test_exec_name_tolerance();
  test_argc_argv_parse();
  test_invalid_option();

  test_benchmark_long();
  test_benchmark_short();

  test_int64_axis_single();
  test_int64_axis_multi();
  test_int64_axis_pow2_single();
  test_int64_axis_pow2_multi();
  test_int64_axis_none_to_pow2_single();
  test_int64_axis_none_to_pow2_multi();
  test_int64_axis_pow2_to_none_single();
  test_int64_axis_pow2_to_none_multi();
  test_float64_axis_single();
  test_float64_axis_multi();
  test_string_axis_single();
  test_string_axis_multi();
  test_type_axis_single();
  test_type_axis_multi();

  test_multi_axis();

  test_axis_before_benchmark();

  test_min_samples();
  test_min_time();
  test_max_noise();
  test_skip_time();
  test_timeout();

  test_stopping_criterion();

  return 0;
}
catch (std::exception &err)
{
  fmt::print(stderr, "{}", err.what());
  return 1;
}
