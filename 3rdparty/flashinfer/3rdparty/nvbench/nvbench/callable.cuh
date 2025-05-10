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

#pragma once

#include <nvbench/type_list.cuh>

#include <type_traits>

namespace nvbench
{
struct state;
}

// Define a simple callable wrapper around a function. This allows the function
// to be used as a class template parameter. Intended for use with kernel
// generators and `NVBENCH_BENCH` macros.
#define NVBENCH_DEFINE_UNIQUE_CALLABLE(function)                                                   \
  NVBENCH_DEFINE_CALLABLE(function, NVBENCH_UNIQUE_IDENTIFIER(function))

#define NVBENCH_DEFINE_CALLABLE(function, callable_name)                                           \
  struct callable_name                                                                             \
  {                                                                                                \
    void operator()(nvbench::state &state, nvbench::type_list<>) { function(state); }              \
  }

#define NVBENCH_DEFINE_UNIQUE_CALLABLE_TEMPLATE(function)                                          \
  NVBENCH_DEFINE_CALLABLE_TEMPLATE(function, NVBENCH_UNIQUE_IDENTIFIER(function))

#define NVBENCH_DEFINE_CALLABLE_TEMPLATE(function, callable_name)                                  \
  struct callable_name                                                                             \
  {                                                                                                \
    template <typename... Ts>                                                                      \
    void operator()(nvbench::state &state, nvbench::type_list<Ts...>)                              \
    {                                                                                              \
      function(state, nvbench::type_list<Ts...>{});                                                \
    }                                                                                              \
  }

#define NVBENCH_UNIQUE_IDENTIFIER(prefix) NVBENCH_UNIQUE_IDENTIFIER_IMPL1(prefix, __LINE__)
#define NVBENCH_UNIQUE_IDENTIFIER_IMPL1(prefix, unique_id)                                         \
  NVBENCH_UNIQUE_IDENTIFIER_IMPL2(prefix, unique_id)
#define NVBENCH_UNIQUE_IDENTIFIER_IMPL2(prefix, unique_id) prefix##_line_##unique_id
