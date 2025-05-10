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

#include <nvbench/axes_metadata.cuh>

#include <nvbench/type_list.cuh>
#include <nvbench/type_strings.cuh>
#include <nvbench/types.cuh>

#include "test_asserts.cuh"

#include <fmt/format.h>

#include <algorithm>
#include <string_view>

using int_list = nvbench::type_list<nvbench::int8_t,
                                    nvbench::int16_t,
                                    nvbench::int32_t,
                                    nvbench::int64_t>;

using float_list = nvbench::type_list<nvbench::float32_t, nvbench::float64_t>;

using misc_list = nvbench::type_list<bool, void>;

using three_type_axes = nvbench::type_list<int_list, float_list, misc_list>;

using no_types = nvbench::type_list<>;

void test_default_type_axes_names()
{
  using TL = nvbench::type_list<int>;

  {
    nvbench::axes_metadata axes{};
    ASSERT(axes.get_axes().size() == 0);
  }

  {
    nvbench::axes_metadata axes{nvbench::type_list<>{}};
    ASSERT(axes.get_axes().size() == 0);
  }

  {
    nvbench::axes_metadata axes{nvbench::type_list<TL>{}};
    ASSERT(axes.get_axes().size() == 1);
    ASSERT(axes.get_type_axis(0).get_name() == "T");
    ASSERT(axes.get_type_axis(0).get_axis_index() == 0);
  }

  {
    nvbench::axes_metadata axes{nvbench::type_list<TL, TL>{}};
    ASSERT(axes.get_axes().size() == 2);
    ASSERT(axes.get_type_axis(0).get_name() == "T");
    ASSERT(axes.get_type_axis(0).get_axis_index() == 0);
    ASSERT(axes.get_type_axis(1).get_name() == "U");
    ASSERT(axes.get_type_axis(1).get_axis_index() == 1);
  }

  {
    nvbench::axes_metadata axes{nvbench::type_list<TL, TL, TL>{}};
    ASSERT(axes.get_axes().size() == 3);
    ASSERT(axes.get_type_axis(0).get_name() == "T");
    ASSERT(axes.get_type_axis(0).get_axis_index() == 0);
    ASSERT(axes.get_type_axis(1).get_name() == "U");
    ASSERT(axes.get_type_axis(1).get_axis_index() == 1);
    ASSERT(axes.get_type_axis(2).get_name() == "V");
    ASSERT(axes.get_type_axis(2).get_axis_index() == 2);
  }

  {
    nvbench::axes_metadata axes{nvbench::type_list<TL, TL, TL, TL>{}};
    ASSERT(axes.get_axes().size() == 4);
    ASSERT(axes.get_type_axis(0).get_name() == "T");
    ASSERT(axes.get_type_axis(0).get_axis_index() == 0);
    ASSERT(axes.get_type_axis(1).get_name() == "U");
    ASSERT(axes.get_type_axis(1).get_axis_index() == 1);
    ASSERT(axes.get_type_axis(2).get_name() == "V");
    ASSERT(axes.get_type_axis(2).get_axis_index() == 2);
    ASSERT(axes.get_type_axis(3).get_name() == "W");
    ASSERT(axes.get_type_axis(3).get_axis_index() == 3);
  }

  {
    nvbench::axes_metadata axes{nvbench::type_list<TL, TL, TL, TL, TL>{}};
    ASSERT(axes.get_axes().size() == 5);
    ASSERT(axes.get_type_axis(0).get_name() == "T0");
    ASSERT(axes.get_type_axis(0).get_axis_index() == 0);
    ASSERT(axes.get_type_axis(1).get_name() == "T1");
    ASSERT(axes.get_type_axis(1).get_axis_index() == 1);
    ASSERT(axes.get_type_axis(2).get_name() == "T2");
    ASSERT(axes.get_type_axis(2).get_axis_index() == 2);
    ASSERT(axes.get_type_axis(3).get_name() == "T3");
    ASSERT(axes.get_type_axis(3).get_axis_index() == 3);
    ASSERT(axes.get_type_axis(4).get_name() == "T4");
    ASSERT(axes.get_type_axis(4).get_axis_index() == 4);
  }

}

void test_type_axes()
{
  nvbench::axes_metadata axes{three_type_axes{}};
  axes.set_type_axes_names({"Integer", "Float", "Other"});

  ASSERT(axes.get_type_axis("Integer").get_name() == "Integer");
  ASSERT(axes.get_type_axis("Float").get_name() == "Float");
  ASSERT(axes.get_type_axis("Other").get_name() == "Other");

  ASSERT(axes.get_type_axis(0).get_name() == "Integer");
  ASSERT(axes.get_type_axis(1).get_name() == "Float");
  ASSERT(axes.get_type_axis(2).get_name() == "Other");

  fmt::memory_buffer buffer;
  for (const auto &axis : axes.get_axes())
  {
    fmt::format_to(std::back_inserter(buffer), "Axis: {}\n", axis->get_name());
    const auto num_values = axis->get_size();
    for (std::size_t i = 0; i < num_values; ++i)
    {
      auto input_string = axis->get_input_string(i);
      auto description  = axis->get_description(i);
      fmt::format_to(std::back_inserter(buffer),
                     " - {}{}\n",
                     input_string,
                     description.empty() ? ""
                                         : fmt::format(" ({})", description));
    }
  }

  const std::string ref = R"expected(Axis: Integer
 - I8 (int8_t)
 - I16 (int16_t)
 - I32 (int32_t)
 - I64 (int64_t)
Axis: Float
 - F32 (float)
 - F64 (double)
Axis: Other
 - bool
 - void
)expected";

  const std::string test = fmt::to_string(buffer);
  const auto diff =
    std::mismatch(ref.cbegin(), ref.cend(), test.cbegin(), test.cend());
  const auto idx = static_cast<std::size_t>(diff.second - test.cbegin());
  ASSERT_MSG(test == ref,
             "Differs at character {}.\n"
             "Expected:\n\"{}\"\n\n"
             "Actual:\n\"{}\"\n"
             "-- ERROR --\n"
             "\"{}\"",
             idx,
             ref,
             std::string_view(test.c_str(), idx),
             std::string_view(test.c_str() + idx, test.size() - idx));
}

void test_float64_axes()
{
  nvbench::axes_metadata axes;
  axes.add_float64_axis("F64 Axis", {0., .1, .25, .5, 1.});
  ASSERT(axes.get_axes().size() == 1);
  const auto &axis = axes.get_float64_axis("F64 Axis");
  ASSERT(axis.get_size() == 5);
  ASSERT(axis.get_value(0) == 0.);
  ASSERT(axis.get_value(1) == .1);
  ASSERT(axis.get_value(2) == .25);
  ASSERT(axis.get_value(3) == .5);
  ASSERT(axis.get_value(4) == 1.);
}

void test_int64_axes()
{
  nvbench::axes_metadata axes;
  axes.add_int64_axis("I64 Axis",
                      {10, 11, 12, 13, 14},
                      nvbench::int64_axis_flags::none);
  ASSERT(axes.get_axes().size() == 1);
  const auto &axis = axes.get_int64_axis("I64 Axis");
  ASSERT(axis.get_size() == 5);
  ASSERT(axis.get_value(0) == 10);
  ASSERT(axis.get_value(1) == 11);
  ASSERT(axis.get_value(2) == 12);
  ASSERT(axis.get_value(3) == 13);
  ASSERT(axis.get_value(4) == 14);
}

void test_int64_power_of_two_axes()
{
  nvbench::axes_metadata axes;
  axes.add_int64_axis("I64 POT Axis",
                      {1, 2, 3, 4, 5},
                      nvbench::int64_axis_flags::power_of_two);
  ASSERT(axes.get_axes().size() == 1);
  const auto &axis = axes.get_int64_axis("I64 POT Axis");
  ASSERT(axis.get_size() == 5);
  ASSERT(axis.get_value(0) == 2);
  ASSERT(axis.get_value(1) == 4);
  ASSERT(axis.get_value(2) == 8);
  ASSERT(axis.get_value(3) == 16);
  ASSERT(axis.get_value(4) == 32);
}

void test_string_axes()
{
  nvbench::axes_metadata axes;
  axes.add_string_axis("Strings", {"string a", "string b", "string c"});
  ASSERT(axes.get_axes().size() == 1);
  const auto &axis = axes.get_string_axis("Strings");
  ASSERT(axis.get_size() == 3);
  ASSERT(axis.get_value(0) == "string a");
  ASSERT(axis.get_value(1) == "string b");
  ASSERT(axis.get_value(2) == "string c");
}

int main()
{
  test_default_type_axes_names();
  test_type_axes();
  test_float64_axes();
  test_int64_axes();
  test_int64_power_of_two_axes();
  test_string_axes();
}
