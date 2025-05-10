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

#include <nvbench/enum_type_list.cuh>

#include "test_asserts.cuh"

#include <fmt/format.h>

#include <type_traits>

// If using gcc version < 7, disable some tests to WAR a compiler bug. See NVIDIA/nvbench#39.
#if defined(__GNUC__) && __GNUC__ == 7
#define USING_GCC_7
#endif

enum class scoped_enum
{
  val_1,
  val_2,
  val_3
};
NVBENCH_DECLARE_ENUM_TYPE_STRINGS(
  scoped_enum,
  [](scoped_enum value) {
    switch (value)
    {
      case scoped_enum::val_1:
        return fmt::format("1");
      case scoped_enum::val_2:
        return fmt::format("2");
      case scoped_enum::val_3:
        return fmt::format("3");
      default:
        return std::string{"Unknown"};
    }
  },
  [](scoped_enum value) {
    switch (value)
    {
      case scoped_enum::val_1:
        return fmt::format("scoped_enum::val_1");
      case scoped_enum::val_2:
        return fmt::format("scoped_enum::val_2");
      case scoped_enum::val_3:
        return fmt::format("scoped_enum::val_3");
      default:
        return std::string{"Unknown"};
    }
  })

enum unscoped_enum
{
  unscoped_val_1,
  unscoped_val_2,
  unscoped_val_3
};
NVBENCH_DECLARE_ENUM_TYPE_STRINGS(
  unscoped_enum,
  [](unscoped_enum value) {
    switch (value)
    {
      case unscoped_val_1:
        return fmt::format("1");
      case unscoped_val_2:
        return fmt::format("2");
      case unscoped_val_3:
        return fmt::format("3");
      default:
        return std::string{"Unknown"};
    }
  },
  [](unscoped_enum value) {
    switch (value)
    {
      case unscoped_val_1:
        return fmt::format("unscoped_val_1");
      case unscoped_val_2:
        return fmt::format("unscoped_val_2");
      case unscoped_val_3:
        return fmt::format("unscoped_val_3");
      default:
        return std::string{"Unknown"};
    }
  })

void test_int()
{
  ASSERT((std::is_same_v<nvbench::enum_type_list<>, nvbench::type_list<>>));
  ASSERT((std::is_same_v<nvbench::enum_type_list<0>,
                         nvbench::type_list<nvbench::enum_type<0>>>));
  ASSERT((std::is_same_v<nvbench::enum_type_list<0, 1, 2, 3, 4>,
                         nvbench::type_list<nvbench::enum_type<0>,
                                            nvbench::enum_type<1>,
                                            nvbench::enum_type<2>,
                                            nvbench::enum_type<3>,
                                            nvbench::enum_type<4>>>));
}

void test_scoped_enum()
{
#ifndef USING_GCC_7
  ASSERT((
    std::is_same_v<nvbench::enum_type_list<scoped_enum::val_1>,
                   nvbench::type_list<nvbench::enum_type<scoped_enum::val_1>>>));
#endif
  ASSERT((
    std::is_same_v<nvbench::enum_type_list<scoped_enum::val_1,
                                           scoped_enum::val_2,
                                           scoped_enum::val_3>,
                   nvbench::type_list<nvbench::enum_type<scoped_enum::val_1>,
                                      nvbench::enum_type<scoped_enum::val_2>,
                                      nvbench::enum_type<scoped_enum::val_3>>>));
}

void test_unscoped_enum()
{
#ifndef USING_GCC_7
  ASSERT(
    (std::is_same_v<nvbench::enum_type_list<unscoped_val_1>,
                    nvbench::type_list<nvbench::enum_type<unscoped_val_1>>>));
  ASSERT(
    (std::is_same_v<
      nvbench::enum_type_list<unscoped_val_1, unscoped_val_2, unscoped_val_3>,
      nvbench::type_list<nvbench::enum_type<unscoped_val_1>,
                         nvbench::enum_type<unscoped_val_2>,
                         nvbench::enum_type<unscoped_val_3>>>));
#endif
}

void test_scoped_enum_type_strings()
{
  using values = nvbench::enum_type_list<scoped_enum::val_1,
                                         scoped_enum::val_2,
                                         scoped_enum::val_3>;
  using val_1  = nvbench::tl::get<0, values>;
  using val_2  = nvbench::tl::get<1, values>;
  using val_3  = nvbench::tl::get<2, values>;
  ASSERT((nvbench::type_strings<val_1>::input_string() == "1"));
  ASSERT((nvbench::type_strings<val_1>::description() == "scoped_enum::val_1"));
  ASSERT((nvbench::type_strings<val_2>::input_string() == "2"));
  ASSERT((nvbench::type_strings<val_2>::description() == "scoped_enum::val_2"));
  ASSERT((nvbench::type_strings<val_3>::input_string() == "3"));
  ASSERT((nvbench::type_strings<val_3>::description() == "scoped_enum::val_3"));
}

void test_unscoped_enum_type_strings()
{
  using values = nvbench::enum_type_list<unscoped_enum::unscoped_val_1,
                                         unscoped_enum::unscoped_val_2,
                                         unscoped_enum::unscoped_val_3>;
  using val_1  = nvbench::tl::get<0, values>;
  using val_2  = nvbench::tl::get<1, values>;
  using val_3  = nvbench::tl::get<2, values>;
  ASSERT((nvbench::type_strings<val_1>::input_string() == "1"));
  ASSERT((nvbench::type_strings<val_2>::input_string() == "2"));
  ASSERT((nvbench::type_strings<val_3>::input_string() == "3"));

  // There's a gcc bug (fixed in gcc8) that causes these templates to resolve
  // incorrectly. See NVIDIA/nvbench#39 and NVBug 3404609 for details.
#if defined(__GNUC__) && __GNUC_ > 7
  ASSERT((nvbench::type_strings<val_1>::description() == "unscoped_val_1"));
  ASSERT((nvbench::type_strings<val_2>::description() == "unscoped_val_2"));
  ASSERT((nvbench::type_strings<val_3>::description() == "unscoped_val_3"));
#endif
}

int main()
{
  test_int();
  test_scoped_enum();
  test_unscoped_enum();
  test_scoped_enum_type_strings();
  test_unscoped_enum_type_strings();
}
