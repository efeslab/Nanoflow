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

#include <fmt/format.h>
#include <stdexcept>

#define NVBENCH_THROW(exception_type, format_str, ...)                                             \
  throw exception_type(                                                                            \
    fmt::format("{}:{}: {}", __FILE__, __LINE__, fmt::format(format_str, __VA_ARGS__)))

#define NVBENCH_THROW_IF(condition, exception_type, format_str, ...)                               \
  do                                                                                               \
  {                                                                                                \
    if (condition)                                                                                 \
    {                                                                                              \
      NVBENCH_THROW(exception_type, format_str, __VA_ARGS__);                                      \
    }                                                                                              \
  } while (false)
