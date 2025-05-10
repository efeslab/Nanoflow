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

#include <nvbench/named_values.cuh>

#include "test_asserts.cuh"

#include <algorithm>

void test_empty()
{
  nvbench::named_values vals;
  ASSERT(vals.get_size() == 0);
  ASSERT(vals.get_names().size() == 0);
  ASSERT(vals.has_value("Nope") == false);
  ASSERT_THROWS_ANY([[maybe_unused]] auto val = vals.get_value("Nope"));
  ASSERT_THROWS_ANY([[maybe_unused]] auto type = vals.get_type("Nope"));
  // Removing non-existent entries shouldn't cause a problem:
  vals.remove_value("Nope");
}

void test_basic()
{
  auto sort = [](auto &&vec) {
    std::sort(vec.begin(), vec.end());
    return std::forward<decltype(vec)>(vec);
  };

  nvbench::named_values vals;
  vals.set_int64("Int", 32);
  vals.set_float64("Float", 34.5);
  vals.set_string("String", "string!");
  vals.set_value("IntVar", {nvbench::int64_t{36}});

  std::vector<std::string> names{"Float", "Int", "IntVar", "String"};

  ASSERT(vals.get_size() == 4);
  ASSERT(sort(vals.get_names()) == names);

  ASSERT(vals.has_value("Float"));
  ASSERT(vals.has_value("Int"));
  ASSERT(vals.has_value("IntVar"));
  ASSERT(vals.has_value("String"));

  ASSERT(std::get<nvbench::float64_t>(vals.get_value("Float")) == 34.5);
  ASSERT(std::get<nvbench::int64_t>(vals.get_value("Int")) == 32);
  ASSERT(std::get<nvbench::int64_t>(vals.get_value("IntVar")) == 36);
  ASSERT(std::get<std::string>(vals.get_value("String")) == "string!");

  ASSERT(vals.get_type("Float") == nvbench::named_values::type::float64);
  ASSERT(vals.get_type("Int") == nvbench::named_values::type::int64);
  ASSERT(vals.get_type("IntVar") == nvbench::named_values::type::int64);
  ASSERT(vals.get_type("String") == nvbench::named_values::type::string);

  ASSERT(vals.get_int64("Int") == 32);
  ASSERT(vals.get_int64("IntVar") == 36);
  ASSERT_THROWS_ANY([[maybe_unused]] auto v = vals.get_int64("Float"));
  ASSERT_THROWS_ANY([[maybe_unused]] auto v = vals.get_int64("String"));

  ASSERT(vals.get_float64("Float") == 34.5);
  ASSERT_THROWS_ANY([[maybe_unused]] auto v = vals.get_float64("Int"));
  ASSERT_THROWS_ANY([[maybe_unused]] auto v = vals.get_float64("IntVar"));
  ASSERT_THROWS_ANY([[maybe_unused]] auto v = vals.get_float64("String"));

  ASSERT(vals.get_string("String") == "string!");
  ASSERT_THROWS_ANY([[maybe_unused]] auto v = vals.get_string("Int"));
  ASSERT_THROWS_ANY([[maybe_unused]] auto v = vals.get_string("IntVar"));
  ASSERT_THROWS_ANY([[maybe_unused]] auto v = vals.get_string("Float"));

  vals.remove_value("IntVar");
  names = {"Float", "Int", "String"};

  ASSERT(vals.get_size() == 3);
  ASSERT(sort(vals.get_names()) == names);

  ASSERT(!vals.has_value("IntVar"));
  ASSERT(vals.has_value("Float"));
  ASSERT(vals.has_value("Int"));
  ASSERT(vals.has_value("String"));

  vals.clear();
  names = {};

  ASSERT(vals.get_size() == 0);
  ASSERT(sort(vals.get_names()) == names);

  ASSERT(!vals.has_value("IntVar"));
  ASSERT(!vals.has_value("Float"));
  ASSERT(!vals.has_value("Int"));
  ASSERT(!vals.has_value("String"));
}

void test_append()
{
  nvbench::named_values vals1;
  vals1.set_int64("Int1", 32);
  vals1.set_float64("Float1", 34.5);
  vals1.set_string("String1", "string1!");
  vals1.set_value("IntVar1", {nvbench::int64_t{36}});

  nvbench::named_values vals2;
  vals2.set_int64("Int2", 42);
  vals2.set_float64("Float2", 3.14);
  vals2.set_string("String2", "string2!");
  vals2.set_value("IntVar2", {nvbench::int64_t{55}});

  vals1.append(vals2);

  // Order should be preserved:
  const auto &names = vals1.get_names();
  ASSERT(names.size() == 8);
  ASSERT(names[0] == "Int1");
  ASSERT(names[1] == "Float1");
  ASSERT(names[2] == "String1");
  ASSERT(names[3] == "IntVar1");
  ASSERT(names[4] == "Int2");
  ASSERT(names[5] == "Float2");
  ASSERT(names[6] == "String2");
  ASSERT(names[7] == "IntVar2");

  ASSERT(vals1.get_int64("Int1") == 32);
  ASSERT(vals1.get_float64("Float1") == 34.5);
  ASSERT(vals1.get_string("String1") == "string1!");
  ASSERT(vals1.get_int64("IntVar1") == 36);
  ASSERT(vals1.get_int64("Int2") == 42);
  ASSERT(vals1.get_float64("Float2") == 3.14);
  ASSERT(vals1.get_string("String2") == "string2!");
  ASSERT(vals1.get_int64("IntVar2") == 55);
}

int main()
{
  test_empty();
  test_basic();
  test_append();
}
