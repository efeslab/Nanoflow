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

#include <nvbench/string_axis.cuh>

#include "test_asserts.cuh"

void test_empty()
{
  nvbench::string_axis axis("Empty");
  axis.set_inputs({});

  ASSERT(axis.get_name() == "Empty");
  ASSERT(axis.get_type() == nvbench::axis_type::string);
  ASSERT(axis.get_size() == 0);
  ASSERT(axis.get_size() == 0);

  const auto clone_base = axis.clone();
  ASSERT(clone_base.get() != nullptr);
  const auto *clone =
    dynamic_cast<const nvbench::string_axis *>(clone_base.get());
  ASSERT(clone != nullptr);

  ASSERT(clone->get_name() == "Empty");
  ASSERT(clone->get_type() == nvbench::axis_type::string);
  ASSERT(clone->get_size() == 0);
  ASSERT(clone->get_size() == 0);
}

void test_basic()
{
  nvbench::string_axis axis("Basic");
  axis.set_inputs({"String 1", "String 2", "String 3"});

  ASSERT(axis.get_name() == "Basic");
  ASSERT(axis.get_size() == 3);
  ASSERT(axis.get_value(0) == "String 1");
  ASSERT(axis.get_input_string(0) == "String 1");
  ASSERT(axis.get_description(0) == "");
  ASSERT(axis.get_value(1) == "String 2");
  ASSERT(axis.get_input_string(1) == "String 2");
  ASSERT(axis.get_description(1) == "");
  ASSERT(axis.get_value(2) == "String 3");
  ASSERT(axis.get_input_string(2) == "String 3");
  ASSERT(axis.get_description(2) == "");

  const auto clone_base = axis.clone();
  ASSERT(clone_base.get() != nullptr);
  const auto *clone =
    dynamic_cast<const nvbench::string_axis *>(clone_base.get());
  ASSERT(clone != nullptr);

  ASSERT(clone->get_name() == "Basic");
  ASSERT(clone->get_size() == 3);
  ASSERT(clone->get_value(0) == "String 1");
  ASSERT(clone->get_input_string(0) == "String 1");
  ASSERT(clone->get_description(0) == "");
  ASSERT(clone->get_value(1) == "String 2");
  ASSERT(clone->get_input_string(1) == "String 2");
  ASSERT(clone->get_description(1) == "");
  ASSERT(clone->get_value(2) == "String 3");
  ASSERT(clone->get_input_string(2) == "String 3");
  ASSERT(clone->get_description(2) == "");
}

void test_update()
{
  nvbench::string_axis axis("Basic");
  axis.set_inputs({"String 1", "String 2", "String 3"});

  auto clone_base = axis.clone();
  ASSERT(clone_base.get() != nullptr);
  auto *clone = dynamic_cast<nvbench::string_axis *>(clone_base.get());
  ASSERT(clone != nullptr);

  // Test that the axis is valid after emptying:
  clone->set_inputs({});
  ASSERT(clone->get_name() == "Basic");
  ASSERT(clone->get_size() == 0);

  // Populate with new values:
  clone->set_inputs({"New String", "Newer String"});
  ASSERT(clone->get_name() == "Basic");
  ASSERT(clone->get_size() == 2);
  ASSERT(clone->get_value(0) == "New String");
  ASSERT(clone->get_input_string(0) == "New String");
  ASSERT(clone->get_description(0) == "");
  ASSERT(clone->get_value(1) == "Newer String");
  ASSERT(clone->get_input_string(1) == "Newer String");
  ASSERT(clone->get_description(1) == "");

  // Check that the original axis didn't change
  ASSERT(axis.get_name() == "Basic");
  ASSERT(axis.get_size() == 3);
  ASSERT(axis.get_value(0) == "String 1");
  ASSERT(axis.get_input_string(0) == "String 1");
  ASSERT(axis.get_description(0) == "");
  ASSERT(axis.get_value(1) == "String 2");
  ASSERT(axis.get_input_string(1) == "String 2");
  ASSERT(axis.get_description(1) == "");
  ASSERT(axis.get_value(2) == "String 3");
  ASSERT(axis.get_input_string(2) == "String 3");
  ASSERT(axis.get_description(2) == "");
}

int main()
{
  test_empty();
  test_basic();
  test_update();
}
