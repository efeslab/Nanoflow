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

#include <nvbench/int64_axis.cuh>

#include "test_asserts.cuh"

#include <fmt/format.h>

void test_empty()
{
  nvbench::int64_axis axis("Empty");

  ASSERT(axis.get_name() == "Empty");
  ASSERT(axis.get_type() == nvbench::axis_type::int64);
  ASSERT(axis.get_size() == 0);

  axis.set_inputs({});

  ASSERT(axis.get_size() == 0);

  const auto clone_base = axis.clone();
  ASSERT(clone_base.get() != nullptr);
  const auto *clone =
    dynamic_cast<const nvbench::int64_axis *>(clone_base.get());
  ASSERT(clone != nullptr);

  ASSERT(clone->get_name() == "Empty");
  ASSERT(clone->get_type() == nvbench::axis_type::int64);
  ASSERT(clone->get_size() == 0);
}

void test_basic()
{
  nvbench::int64_axis axis{"BasicAxis"};
  axis.set_inputs({0, 1, 2, 3, 7, 6, 5, 4});
  const std::vector<nvbench::int64_t> ref{0, 1, 2, 3, 7, 6, 5, 4};

  ASSERT(axis.get_name() == "BasicAxis");
  ASSERT(axis.get_type() == nvbench::axis_type::int64);
  ASSERT(!axis.is_power_of_two());
  ASSERT(axis.get_size() == 8);

  ASSERT(axis.get_inputs() == ref);
  ASSERT(axis.get_values() == ref);
  for (size_t i = 0; i < 8; ++i)
  {
    ASSERT(axis.get_input_string(i) == fmt::to_string(ref[i]));
    ASSERT(axis.get_description(i).empty());
  }

  const auto clone_base = axis.clone();
  ASSERT(clone_base.get() != nullptr);
  const auto *clone =
    dynamic_cast<const nvbench::int64_axis *>(clone_base.get());
  ASSERT(clone != nullptr);

  ASSERT(clone->get_name() == "BasicAxis");
  ASSERT(clone->get_type() == nvbench::axis_type::int64);
  ASSERT(!clone->is_power_of_two());
  ASSERT(clone->get_size() == 8);

  ASSERT(clone->get_inputs() == ref);
  ASSERT(clone->get_values() == ref);
  for (size_t i = 0; i < 8; ++i)
  {
    ASSERT(clone->get_input_string(i) == fmt::to_string(ref[i]));
    ASSERT(clone->get_description(i).empty());
  }
}

void test_power_of_two()
{
  nvbench::int64_axis axis{"POTAxis"};
  axis.set_inputs({0, 1, 2, 3, 7, 6, 5, 4},
                  nvbench::int64_axis_flags::power_of_two);
  const std::vector<nvbench::int64_t> ref_inputs{0, 1, 2, 3, 7, 6, 5, 4};
  const std::vector<nvbench::int64_t> ref_values{1, 2, 4, 8, 128, 64, 32, 16};

  ASSERT(axis.get_name() == "POTAxis");
  ASSERT(axis.get_type() == nvbench::axis_type::int64);
  ASSERT(axis.is_power_of_two());
  ASSERT(axis.get_size() == 8);

  ASSERT(axis.get_inputs() == ref_inputs);
  ASSERT(axis.get_values() == ref_values);
  for (size_t i = 0; i < 8; ++i)
  {
    ASSERT(axis.get_input_string(i) == fmt::to_string(ref_inputs[i]));
    ASSERT(axis.get_description(i) ==
           fmt::format("2^{} = {}", ref_inputs[i], ref_values[i]));
  }

  const auto clone_base = axis.clone();
  ASSERT(clone_base.get() != nullptr);
  const auto *clone =
    dynamic_cast<const nvbench::int64_axis *>(clone_base.get());
  ASSERT(clone != nullptr);

  ASSERT(clone->get_name() == "POTAxis");
  ASSERT(clone->get_type() == nvbench::axis_type::int64);
  ASSERT(clone->is_power_of_two());
  ASSERT(clone->get_size() == 8);

  ASSERT(clone->get_inputs() == ref_inputs);
  ASSERT(clone->get_values() == ref_values);
  for (size_t i = 0; i < 8; ++i)
  {
    ASSERT(clone->get_input_string(i) == fmt::to_string(ref_inputs[i]));
    ASSERT(clone->get_description(i) ==
           fmt::format("2^{} = {}", ref_inputs[i], ref_values[i]));
  }
}

void test_update_none_to_none()
{
  nvbench::int64_axis axis{"TestAxis"};
  const std::vector<nvbench::int64_t> ref{0, 1, 2, 3, 7, 6, 5, 4};
  axis.set_inputs(ref);

  { // Update a clone with empty values
    auto clone_base = axis.clone();
    ASSERT(clone_base.get() != nullptr);
    auto *clone = dynamic_cast<nvbench::int64_axis *>(clone_base.get());
    ASSERT(clone != nullptr);

    clone->set_inputs({});
    ASSERT(clone->get_name() == "TestAxis");
    ASSERT(clone->get_type() == nvbench::axis_type::int64);
    ASSERT(!clone->is_power_of_two());
    ASSERT(clone->get_size() == 0);

    ASSERT(clone->get_inputs().empty());
    ASSERT(clone->get_values().empty());
  }

  { // Update a clone with new values
    auto clone_base = axis.clone();
    ASSERT(clone_base.get() != nullptr);
    auto *clone = dynamic_cast<nvbench::int64_axis *>(clone_base.get());
    ASSERT(clone != nullptr);

    const std::vector<nvbench::int64_t> update_ref{2, 5, 7, 9};
    clone->set_inputs(update_ref);
    ASSERT(clone->get_name() == "TestAxis");
    ASSERT(clone->get_type() == nvbench::axis_type::int64);
    ASSERT(!clone->is_power_of_two());
    ASSERT(clone->get_size() == 4);

    ASSERT(clone->get_inputs() == update_ref);
    ASSERT(clone->get_values() == update_ref);
    for (size_t i = 0; i < update_ref.size(); ++i)
    {
      ASSERT(clone->get_input_string(i) == fmt::to_string(update_ref[i]));
      ASSERT(clone->get_description(i).empty());
    }
  }

  // Check that the original axis is unchanged:
  ASSERT(axis.get_name() == "TestAxis");
  ASSERT(axis.get_type() == nvbench::axis_type::int64);
  ASSERT(!axis.is_power_of_two());
  ASSERT(axis.get_size() == 8);

  ASSERT(axis.get_inputs() == ref);
  ASSERT(axis.get_values() == ref);
  for (size_t i = 0; i < 8; ++i)
  {
    ASSERT(axis.get_input_string(i) == fmt::to_string(ref[i]));
    ASSERT(axis.get_description(i).empty());
  }
}

void test_update_none_to_pow2()
{
  nvbench::int64_axis axis{"TestAxis"};
  const std::vector<nvbench::int64_t> ref{0, 1, 2, 3, 7, 6, 5, 4};
  axis.set_inputs(ref);

  { // Update a clone with empty values
    auto clone_base = axis.clone();
    ASSERT(clone_base.get() != nullptr);
    auto *clone = dynamic_cast<nvbench::int64_axis *>(clone_base.get());
    ASSERT(clone != nullptr);

    clone->set_inputs({}, nvbench::int64_axis_flags::power_of_two);
    ASSERT(clone->get_name() == "TestAxis");
    ASSERT(clone->get_type() == nvbench::axis_type::int64);
    ASSERT(clone->is_power_of_two());
    ASSERT(clone->get_size() == 0);

    ASSERT(clone->get_inputs().empty());
    ASSERT(clone->get_values().empty());
  }

  { // Update a clone with new values
    auto clone_base = axis.clone();
    ASSERT(clone_base.get() != nullptr);
    auto *clone = dynamic_cast<nvbench::int64_axis *>(clone_base.get());
    ASSERT(clone != nullptr);

    const std::vector<nvbench::int64_t> update_inputs{2, 5, 7, 9};
    const std::vector<nvbench::int64_t> update_values{4, 32, 128, 512};
    clone->set_inputs(update_inputs, nvbench::int64_axis_flags::power_of_two);
    ASSERT(clone->get_name() == "TestAxis");
    ASSERT(clone->get_type() == nvbench::axis_type::int64);
    ASSERT(clone->is_power_of_two());
    ASSERT(clone->get_size() == 4);

    ASSERT(clone->get_inputs() == update_inputs);
    ASSERT(clone->get_values() == update_values);
    for (size_t i = 0; i < update_inputs.size(); ++i)
    {
      ASSERT(clone->get_input_string(i) == fmt::to_string(update_inputs[i]));
      ASSERT(clone->get_description(i) ==
             fmt::format("2^{} = {}", update_inputs[i], update_values[i]));
    }
  }

  // Check that the original axis is unchanged:
  ASSERT(axis.get_name() == "TestAxis");
  ASSERT(axis.get_type() == nvbench::axis_type::int64);
  ASSERT(!axis.is_power_of_two());
  ASSERT(axis.get_size() == 8);

  ASSERT(axis.get_inputs() == ref);
  ASSERT(axis.get_values() == ref);
  for (size_t i = 0; i < 8; ++i)
  {
    ASSERT(axis.get_input_string(i) == fmt::to_string(ref[i]));
    ASSERT(axis.get_description(i).empty());
  }
}

void test_update_pow2_to_none()
{
  nvbench::int64_axis axis{"TestAxis"};
  axis.set_inputs({0, 1, 2, 3, 7, 6, 5, 4},
                  nvbench::int64_axis_flags::power_of_two);
  const std::vector<nvbench::int64_t> ref_inputs{0, 1, 2, 3, 7, 6, 5, 4};
  const std::vector<nvbench::int64_t> ref_values{1, 2, 4, 8, 128, 64, 32, 16};

  { // Update a clone with empty values
    auto clone_base = axis.clone();
    ASSERT(clone_base.get() != nullptr);
    auto *clone = dynamic_cast<nvbench::int64_axis *>(clone_base.get());
    ASSERT(clone != nullptr);

    clone->set_inputs({});
    ASSERT(clone->get_name() == "TestAxis");
    ASSERT(clone->get_type() == nvbench::axis_type::int64);
    ASSERT(!clone->is_power_of_two());
    ASSERT(clone->get_size() == 0);

    ASSERT(clone->get_inputs().empty());
    ASSERT(clone->get_values().empty());
  }

  { // Update a clone with new values
    auto clone_base = axis.clone();
    ASSERT(clone_base.get() != nullptr);
    auto *clone = dynamic_cast<nvbench::int64_axis *>(clone_base.get());
    ASSERT(clone != nullptr);

    const std::vector<nvbench::int64_t> update_ref{2, 5, 7, 9};
    clone->set_inputs(update_ref);
    ASSERT(clone->get_name() == "TestAxis");
    ASSERT(clone->get_type() == nvbench::axis_type::int64);
    ASSERT(!clone->is_power_of_two());
    ASSERT(clone->get_size() == 4);

    ASSERT(clone->get_inputs() == update_ref);
    ASSERT(clone->get_values() == update_ref);
    for (size_t i = 0; i < update_ref.size(); ++i)
    {
      ASSERT(clone->get_input_string(i) == fmt::to_string(update_ref[i]));
      ASSERT(clone->get_description(i).empty());
    }
  }

  // Check that the original axis is unchanged:
  ASSERT(axis.get_name() == "TestAxis");
  ASSERT(axis.get_type() == nvbench::axis_type::int64);
  ASSERT(axis.is_power_of_two());
  ASSERT(axis.get_size() == 8);

  ASSERT(axis.get_inputs() == ref_inputs);
  ASSERT(axis.get_values() == ref_values);
  for (size_t i = 0; i < 8; ++i)
  {
    ASSERT(axis.get_input_string(i) == fmt::to_string(ref_inputs[i]));
    ASSERT(axis.get_description(i) ==
           fmt::format("2^{} = {}", ref_inputs[i], ref_values[i]));
  }
}

void test_update_pow2_to_pow2()
{

  nvbench::int64_axis axis{"TestAxis"};
  axis.set_inputs({0, 1, 2, 3, 7, 6, 5, 4},
                  nvbench::int64_axis_flags::power_of_two);
  const std::vector<nvbench::int64_t> ref_inputs{0, 1, 2, 3, 7, 6, 5, 4};
  const std::vector<nvbench::int64_t> ref_values{1, 2, 4, 8, 128, 64, 32, 16};

  { // Update a clone with empty values
    auto clone_base = axis.clone();
    ASSERT(clone_base.get() != nullptr);
    auto *clone = dynamic_cast<nvbench::int64_axis *>(clone_base.get());
    ASSERT(clone != nullptr);

    clone->set_inputs({}, nvbench::int64_axis_flags::power_of_two);
    ASSERT(clone->get_name() == "TestAxis");
    ASSERT(clone->get_type() == nvbench::axis_type::int64);
    ASSERT(clone->is_power_of_two());
    ASSERT(clone->get_size() == 0);

    ASSERT(clone->get_inputs().empty());
    ASSERT(clone->get_values().empty());
  }

  { // Update a clone with new values
    auto clone_base = axis.clone();
    ASSERT(clone_base.get() != nullptr);
    auto *clone = dynamic_cast<nvbench::int64_axis *>(clone_base.get());
    ASSERT(clone != nullptr);

    const std::vector<nvbench::int64_t> update_inputs{2, 5, 7, 9};
    const std::vector<nvbench::int64_t> update_values{4, 32, 128, 512};
    clone->set_inputs(update_inputs, nvbench::int64_axis_flags::power_of_two);
    ASSERT(clone->get_name() == "TestAxis");
    ASSERT(clone->get_type() == nvbench::axis_type::int64);
    ASSERT(clone->is_power_of_two());
    ASSERT(clone->get_size() == 4);

    ASSERT(clone->get_inputs() == update_inputs);
    ASSERT(clone->get_values() == update_values);
    for (size_t i = 0; i < update_inputs.size(); ++i)
    {
      ASSERT(clone->get_input_string(i) == fmt::to_string(update_inputs[i]));
      ASSERT(clone->get_description(i) ==
             fmt::format("2^{} = {}", update_inputs[i], update_values[i]));
    }
  }

  // Check that the original axis is unchanged:
  ASSERT(axis.get_name() == "TestAxis");
  ASSERT(axis.get_type() == nvbench::axis_type::int64);
  ASSERT(axis.is_power_of_two());
  ASSERT(axis.get_size() == 8);

  ASSERT(axis.get_inputs() == ref_inputs);
  ASSERT(axis.get_values() == ref_values);
  for (size_t i = 0; i < 8; ++i)
  {
    ASSERT(axis.get_input_string(i) == fmt::to_string(ref_inputs[i]));
    ASSERT(axis.get_description(i) ==
           fmt::format("2^{} = {}", ref_inputs[i], ref_values[i]));
  }
}

int main()
{
  test_empty();
  test_basic();
  test_power_of_two();
  test_update_none_to_none();
  test_update_none_to_pow2();
  test_update_pow2_to_none();
  test_update_pow2_to_pow2();

  return EXIT_SUCCESS;
}
