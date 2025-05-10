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

#include <nvbench/type_list.cuh>

#include <nvbench/type_strings.cuh>

#include "test_asserts.cuh"

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

// Unique, numbered types for testing type_list functionality.
using T0 = std::integral_constant<std::size_t, 0>;
using T1 = std::integral_constant<std::size_t, 1>;
using T2 = std::integral_constant<std::size_t, 2>;
using T3 = std::integral_constant<std::size_t, 3>;
using T4 = std::integral_constant<std::size_t, 4>;
using T5 = std::integral_constant<std::size_t, 5>;
using T6 = std::integral_constant<std::size_t, 6>;
using T7 = std::integral_constant<std::size_t, 7>;

NVBENCH_DECLARE_TYPE_STRINGS(T0, "T0", "T0");
NVBENCH_DECLARE_TYPE_STRINGS(T1, "T1", "T1");
NVBENCH_DECLARE_TYPE_STRINGS(T2, "T2", "T2");
NVBENCH_DECLARE_TYPE_STRINGS(T3, "T3", "T3");
NVBENCH_DECLARE_TYPE_STRINGS(T4, "T4", "T4");
NVBENCH_DECLARE_TYPE_STRINGS(T5, "T5", "T5");
NVBENCH_DECLARE_TYPE_STRINGS(T6, "T6", "T6");
NVBENCH_DECLARE_TYPE_STRINGS(T7, "T7", "T7");

struct test_size
{
  using TL0 = nvbench::type_list<>;
  using TL1 = nvbench::type_list<T0>;
  using TL2 = nvbench::type_list<T0, T1>;
  using TL3 = nvbench::type_list<T0, T1, T2>;
  static_assert(nvbench::tl::size<TL0>{} == 0);
  static_assert(nvbench::tl::size<TL1>{} == 1);
  static_assert(nvbench::tl::size<TL2>{} == 2);
  static_assert(nvbench::tl::size<TL3>{} == 3);
};

struct test_get
{
  using TL = nvbench::type_list<T0, T1, T2, T3, T4, T5>;
  static_assert(std::is_same_v<T0, nvbench::tl::get<0, TL>>);
  static_assert(std::is_same_v<T1, nvbench::tl::get<1, TL>>);
  static_assert(std::is_same_v<T2, nvbench::tl::get<2, TL>>);
  static_assert(std::is_same_v<T3, nvbench::tl::get<3, TL>>);
  static_assert(std::is_same_v<T4, nvbench::tl::get<4, TL>>);
  static_assert(std::is_same_v<T5, nvbench::tl::get<5, TL>>);
};

struct test_concat
{
  using TLEmpty = nvbench::type_list<>;
  using TL012   = nvbench::type_list<T0, T1, T2>;
  using TL765   = nvbench::type_list<T7, T6, T5>;

  struct empty_tests
  {
    static_assert(
      std::is_same_v<nvbench::tl::concat<TLEmpty, TLEmpty>, TLEmpty>);
    static_assert(std::is_same_v<nvbench::tl::concat<TLEmpty, TL012>, TL012>);
    static_assert(std::is_same_v<nvbench::tl::concat<TL012, TLEmpty>, TL012>);
  };

  static_assert(std::is_same_v<nvbench::tl::concat<TL012, TL765>,
                               nvbench::type_list<T0, T1, T2, T7, T6, T5>>);
};

struct test_prepend_each
{
  using T   = void;
  using T01 = nvbench::type_list<T0, T1>;
  using T23 = nvbench::type_list<T2, T3>;
  using TLs = nvbench::type_list<T01, T23>;

  using Expected = nvbench::type_list<nvbench::type_list<T, T0, T1>,
                                      nvbench::type_list<T, T2, T3>>;
  static_assert(std::is_same_v<nvbench::tl::prepend_each<T, TLs>, Expected>);
};

struct test_empty_cartesian_product
{
  using prod = nvbench::tl::cartesian_product<nvbench::type_list<>>;
  static_assert(std::is_same_v<prod, nvbench::type_list<nvbench::type_list<>>>);
};

struct test_single_cartesian_product
{
  using prod_1 =
    nvbench::tl::cartesian_product<nvbench::type_list<nvbench::type_list<T0>>>;
  static_assert(
    std::is_same_v<prod_1, nvbench::type_list<nvbench::type_list<T0>>>);

  using prod_2 = nvbench::tl::cartesian_product<
    nvbench::type_list<nvbench::type_list<T0, T1>>>;
  static_assert(std::is_same_v<prod_2,
                               nvbench::type_list<nvbench::type_list<T0>,
                                                  nvbench::type_list<T1>>>);
};

struct test_cartesian_product
{
  using U0       = T2;
  using U1       = T3;
  using U2       = T4;
  using V0       = T5;
  using V1       = T6;
  using T01      = nvbench::type_list<T0, T1>;
  using U012     = nvbench::type_list<U0, U1, U2>;
  using V01      = nvbench::type_list<V0, V1>;
  using TLs      = nvbench::type_list<T01, U012, V01>;
  using CartProd = nvbench::type_list<nvbench::type_list<T0, U0, V0>,
                                      nvbench::type_list<T0, U0, V1>,
                                      nvbench::type_list<T0, U1, V0>,
                                      nvbench::type_list<T0, U1, V1>,
                                      nvbench::type_list<T0, U2, V0>,
                                      nvbench::type_list<T0, U2, V1>,
                                      nvbench::type_list<T1, U0, V0>,
                                      nvbench::type_list<T1, U0, V1>,
                                      nvbench::type_list<T1, U1, V0>,
                                      nvbench::type_list<T1, U1, V1>,
                                      nvbench::type_list<T1, U2, V0>,
                                      nvbench::type_list<T1, U2, V1>>;
  static_assert(std::is_same_v<nvbench::tl::cartesian_product<TLs>, CartProd>);
};

struct test_foreach
{
  using TL0 = nvbench::type_list<>;
  using TL1 = nvbench::type_list<T0>;
  using TL2 = nvbench::type_list<T0, T1>;
  using TL3 = nvbench::type_list<T0, T1, T2>;

  template <typename TypeList>
  static void test(std::vector<std::string> ref_vals)
  {
    std::vector<std::string> test_vals;
    nvbench::tl::foreach<TypeList>([&test_vals](auto wrapped_type) {
      using T = typename decltype(wrapped_type)::type;
      test_vals.push_back(nvbench::type_strings<T>::input_string());
    });
    ASSERT_MSG(test_vals == ref_vals, "{} != {}", test_vals, ref_vals);
  }

  static void run()
  {
    test<TL0>({});
    test<TL1>({"T0"});
    test<TL2>({"T0", "T1"});
    test<TL3>({"T0", "T1", "T2"});
  }
};

int main()
{
  // Note that most tests in this file are just static asserts. Only those with
  // runtime components are listed here.
  test_foreach::run();
}
