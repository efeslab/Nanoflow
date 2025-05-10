/*
 *  Copyright 2021-2022 NVIDIA Corporation
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

#include <nvbench/state.cuh>

#include <nvbench/benchmark.cuh>
#include <nvbench/callable.cuh>
#include <nvbench/summary.cuh>
#include <nvbench/types.cuh>

#include "test_asserts.cuh"

// Mock up a benchmark for testing:
void dummy_generator(nvbench::state &) {}
NVBENCH_DEFINE_CALLABLE(dummy_generator, dummy_callable);
using dummy_bench = nvbench::benchmark<dummy_callable>;

// Subclass to gain access to protected members for testing:
namespace nvbench::detail
{
struct state_tester : public nvbench::state
{
  state_tester(const nvbench::benchmark_base &bench)
      : nvbench::state{bench}
  {}

  template <typename T>
  void set_param(std::string name, T &&value)
  {
    this->state::m_axis_values.set_value(std::move(name),
                                         nvbench::named_values::value_type{
                                           std::forward<T>(value)});
  }
};
} // namespace nvbench::detail

using nvbench::detail::state_tester;

void test_streams()
{
  dummy_bench bench;

  state_tester state{bench};

  // Test non-owning stream
  cudaStream_t default_stream = 0;
  state.set_cuda_stream(nvbench::cuda_stream{default_stream, false});
  ASSERT(state.get_cuda_stream() == default_stream);

  // Test owning stream
  auto stream = nvbench::cuda_stream{};
  auto gold   = stream.get_stream();
  state.set_cuda_stream(std::move(stream));
  ASSERT(state.get_cuda_stream() == gold);
}

void test_params()
{
  dummy_bench bench;

  // Build a state param by param
  state_tester state{bench};
  state.set_param("TestInt", nvbench::int64_t{22});
  state.set_param("TestFloat", nvbench::float64_t{3.14});
  state.set_param("TestString", "A String!");

  ASSERT(state.get_int64("TestInt") == nvbench::int64_t{22});
  ASSERT(state.get_float64("TestFloat") == nvbench::float64_t{3.14});
  ASSERT(state.get_string("TestString") == "A String!");
}

void test_summaries()
{
  dummy_bench bench;
  state_tester state{bench};
  ASSERT(state.get_summaries().size() == 0);

  {
    nvbench::summary &summary = state.add_summary("Test Summary1");
    summary.set_float64("Float", 3.14);
    summary.set_int64("Int", 128);
    summary.set_string("String", "str");
  }

  ASSERT(state.get_summaries().size() == 1);
  ASSERT(state.get_summary("Test Summary1").get_size() == 3);
  ASSERT(state.get_summary("Test Summary1").get_float64("Float") == 3.14);
  ASSERT(state.get_summary("Test Summary1").get_int64("Int") == 128);
  ASSERT(state.get_summary("Test Summary1").get_string("String") == "str");

  {
    nvbench::summary summary{"Test Summary2"};
    state.add_summary(std::move(summary));
  }

  ASSERT(state.get_summaries().size() == 2);
  ASSERT(state.get_summary("Test Summary1").get_size() == 3);
  ASSERT(state.get_summary("Test Summary1").get_float64("Float") == 3.14);
  ASSERT(state.get_summary("Test Summary1").get_int64("Int") == 128);
  ASSERT(state.get_summary("Test Summary1").get_string("String") == "str");
  ASSERT(state.get_summary("Test Summary2").get_size() == 0);
}

void test_defaults()
{
  dummy_bench bench;
  state_tester state{bench};

  ASSERT(state.get_int64_or_default("Foo", 42) == 42);
  ASSERT(state.get_float64_or_default("Baz", 42.4) == 42.4);
  ASSERT(state.get_string_or_default("Bar", "Kramble") == "Kramble");
}

int main()
{
  test_streams();
  test_params();
  test_summaries();
  test_defaults();
}
