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

#include <nvbench/float64_axis.cuh>

#include "test_asserts.cuh"

void test_empty()
{
  nvbench::float64_axis axis("Empty");

  ASSERT(axis.get_name() == "Empty");
  ASSERT(axis.get_type() == nvbench::axis_type::float64);
  ASSERT(axis.get_size() == 0);

  axis.set_inputs({});

  ASSERT(axis.get_size() == 0);

  const auto clone_base = axis.clone();
  ASSERT(clone_base.get() != nullptr);
  const auto *clone =
    dynamic_cast<const nvbench::float64_axis *>(clone_base.get());
  ASSERT(clone != nullptr);

  ASSERT(clone->get_name() == "Empty");
  ASSERT(clone->get_type() == nvbench::axis_type::float64);
  ASSERT(clone->get_size() == 0);
}

void test_basic()
{
  nvbench::float64_axis axis("Basic");
  axis.set_inputs({-100.3, 0., 2064.15});

  ASSERT(axis.get_name() == "Basic");
  ASSERT(axis.get_size() == 3);
  ASSERT(axis.get_value(0) == -100.3);
  ASSERT(axis.get_input_string(0) == "-100.3");
  ASSERT(axis.get_description(0) == "");
  ASSERT(axis.get_value(1) == 0.);
  ASSERT(axis.get_input_string(1) == "0");
  ASSERT(axis.get_description(1) == "");
  ASSERT(axis.get_value(2) == 2064.15);
  ASSERT(axis.get_input_string(2) == "2064.2");
  ASSERT(axis.get_description(2) == "");

  const auto clone_base = axis.clone();
  ASSERT(clone_base.get() != nullptr);
  const auto *clone =
    dynamic_cast<const nvbench::float64_axis *>(clone_base.get());
  ASSERT(clone != nullptr);

  ASSERT(clone->get_name() == "Basic");
  ASSERT(clone->get_size() == 3);
  ASSERT(clone->get_value(0) == -100.3);
  ASSERT(clone->get_input_string(0) == "-100.3");
  ASSERT(clone->get_description(0) == "");
  ASSERT(clone->get_value(1) == 0.);
  ASSERT(clone->get_input_string(1) == "0");
  ASSERT(clone->get_description(1) == "");
  ASSERT(clone->get_value(2) == 2064.15);
  ASSERT(clone->get_input_string(2) == "2064.2");
  ASSERT(clone->get_description(2) == "");
}

void test_updates()
{
  nvbench::float64_axis axis("Basic");
  axis.set_inputs({-100.3, 0., 2064.15});

  ASSERT(axis.get_name() == "Basic");
  ASSERT(axis.get_size() == 3);
  ASSERT(axis.get_value(0) == -100.3);
  ASSERT(axis.get_input_string(0) == "-100.3");
  ASSERT(axis.get_description(0) == "");
  ASSERT(axis.get_value(1) == 0.);
  ASSERT(axis.get_input_string(1) == "0");
  ASSERT(axis.get_description(1) == "");
  ASSERT(axis.get_value(2) == 2064.15);
  ASSERT(axis.get_input_string(2) == "2064.2");
  ASSERT(axis.get_description(2) == "");

  auto clone_base = axis.clone();
  ASSERT(clone_base.get() != nullptr);
  auto *clone = dynamic_cast<nvbench::float64_axis *>(clone_base.get());
  ASSERT(clone != nullptr);

  ASSERT(clone->get_name() == "Basic");
  ASSERT(clone->get_size() == 3);
  ASSERT(clone->get_value(0) == -100.3);
  ASSERT(clone->get_input_string(0) == "-100.3");
  ASSERT(clone->get_description(0) == "");
  ASSERT(clone->get_value(1) == 0.);
  ASSERT(clone->get_input_string(1) == "0");
  ASSERT(clone->get_description(1) == "");
  ASSERT(clone->get_value(2) == 2064.15);
  ASSERT(clone->get_input_string(2) == "2064.2");
  ASSERT(clone->get_description(2) == "");

  // Clear the clone:
  clone->set_inputs({});
  ASSERT(clone->get_name() == "Basic");
  ASSERT(clone->get_size() == 0);

  // Original axis should be unaffected:
  ASSERT(axis.get_name() == "Basic");
  ASSERT(axis.get_size() == 3);
  ASSERT(axis.get_value(0) == -100.3);
  ASSERT(axis.get_input_string(0) == "-100.3");
  ASSERT(axis.get_description(0) == "");
  ASSERT(axis.get_value(1) == 0.);
  ASSERT(axis.get_input_string(1) == "0");
  ASSERT(axis.get_description(1) == "");
  ASSERT(axis.get_value(2) == 2064.15);
  ASSERT(axis.get_input_string(2) == "2064.2");
  ASSERT(axis.get_description(2) == "");

  // Insert new data:
  clone->set_inputs({3.14, 6.28});
  ASSERT(clone->get_name() == "Basic");
  ASSERT(clone->get_size() == 2);
  ASSERT(clone->get_value(0) == 3.14);
  ASSERT(clone->get_input_string(0) == "3.14");
  ASSERT(clone->get_description(0) == "");
  ASSERT(clone->get_value(1) == 6.28);
  ASSERT(clone->get_input_string(1) == "6.28");
  ASSERT(clone->get_description(1) == "");

  // Original axis should be unaffected:
  ASSERT(axis.get_name() == "Basic");
  ASSERT(axis.get_size() == 3);
  ASSERT(axis.get_value(0) == -100.3);
  ASSERT(axis.get_input_string(0) == "-100.3");
  ASSERT(axis.get_description(0) == "");
  ASSERT(axis.get_value(1) == 0.);
  ASSERT(axis.get_input_string(1) == "0");
  ASSERT(axis.get_description(1) == "");
  ASSERT(axis.get_value(2) == 2064.15);
  ASSERT(axis.get_input_string(2) == "2064.2");
  ASSERT(axis.get_description(2) == "");
}

int main()
{
  test_empty();
  test_basic();
  test_updates();
}
