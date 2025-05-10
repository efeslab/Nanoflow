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

#include <type_traits>

#define NVBENCH_DECLARE_FLAGS(T)                                                                   \
  constexpr inline T operator|(T v1, T v2)                                                         \
  {                                                                                                \
    using UT = std::underlying_type_t<T>;                                                          \
    return static_cast<T>(static_cast<UT>(v1) | static_cast<UT>(v2));                              \
  }                                                                                                \
  constexpr inline T operator&(T v1, T v2)                                                         \
  {                                                                                                \
    using UT = std::underlying_type_t<T>;                                                          \
    return static_cast<T>(static_cast<UT>(v1) & static_cast<UT>(v2));                              \
  }                                                                                                \
  constexpr inline T operator^(T v1, T v2)                                                         \
  {                                                                                                \
    using UT = std::underlying_type_t<T>;                                                          \
    return static_cast<T>(static_cast<UT>(v1) ^ static_cast<UT>(v2));                              \
  }                                                                                                \
  constexpr inline T operator~(T v1)                                                               \
  {                                                                                                \
    using UT = std::underlying_type_t<T>;                                                          \
    return static_cast<T>(~static_cast<UT>(v1));                                                   \
  }
