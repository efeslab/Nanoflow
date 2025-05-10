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

#include <nvbench/types.cuh>

#include <string>
#include <type_traits>
#include <typeinfo>

namespace nvbench
{

std::string demangle(const std::string &str);

template <typename T>
std::string demangle()
{
  return demangle(typeid(T).name());
}

template <typename T>
struct type_strings
{
  // The string used to identify the type in shorthand (e.g. output tables and
  // CLI options):
  static std::string input_string() { return nvbench::demangle<T>(); }

  // A more descriptive identifier for the type, if input_string is not a common
  // identifier. May be blank if `input_string` is obvious.
  static std::string description() { return {}; }
};

template <typename T, T Value>
struct type_strings<std::integral_constant<T, Value>>
{
  // The string used to identify the type in shorthand (e.g. output tables and
  // CLI options):
  static std::string input_string() { return std::to_string(Value); }

  // A more descriptive identifier for the type, if input_string is not a common
  // identifier. May be blank if `input_string` is obvious.
  static std::string description() { return nvbench::demangle<std::integral_constant<T, Value>>(); }
};

} // namespace nvbench

/*!
 * Declare an `input_string` and `description` to use with a specific `type`.
 */
#define NVBENCH_DECLARE_TYPE_STRINGS(Type, InputString, Description)                               \
  namespace nvbench                                                                                \
  {                                                                                                \
  template <>                                                                                      \
  struct type_strings<Type>                                                                        \
  {                                                                                                \
    static std::string input_string() { return {InputString}; }                                    \
    static std::string description() { return {Description}; }                                     \
  };                                                                                               \
  }

NVBENCH_DECLARE_TYPE_STRINGS(nvbench::int8_t, "I8", "int8_t");
NVBENCH_DECLARE_TYPE_STRINGS(nvbench::int16_t, "I16", "int16_t");
NVBENCH_DECLARE_TYPE_STRINGS(nvbench::int32_t, "I32", "int32_t");
NVBENCH_DECLARE_TYPE_STRINGS(nvbench::int64_t, "I64", "int64_t");
NVBENCH_DECLARE_TYPE_STRINGS(nvbench::uint8_t, "U8", "uint8_t");
NVBENCH_DECLARE_TYPE_STRINGS(nvbench::uint16_t, "U16", "uint16_t");
NVBENCH_DECLARE_TYPE_STRINGS(nvbench::uint32_t, "U32", "uint32_t");
NVBENCH_DECLARE_TYPE_STRINGS(nvbench::uint64_t, "U64", "uint64_t");
NVBENCH_DECLARE_TYPE_STRINGS(nvbench::float32_t, "F32", "float");
NVBENCH_DECLARE_TYPE_STRINGS(nvbench::float64_t, "F64", "double");
