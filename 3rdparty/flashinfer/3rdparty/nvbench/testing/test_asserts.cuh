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

#include <cstdio>
#include <stdexcept>

#define ASSERT(cond)                                                            \
  do                                                                            \
  {                                                                             \
    if (cond)                                                                   \
    {}                                                                          \
    else                                                                        \
    {                                                                           \
      fmt::print("{}:{}: Assertion failed ({}).\n", __FILE__, __LINE__, #cond); \
      std::fflush(stdout);                                                      \
      throw std::runtime_error("Unit test failure.");                           \
    }                                                                           \
  } while (false)

#define ASSERT_MSG(cond, fmtstr, ...)                                          \
  do                                                                           \
  {                                                                            \
    if (cond)                                                                  \
    {}                                                                         \
    else                                                                       \
    {                                                                          \
      fmt::print("{}:{}: Test assertion failed ({}) {}\n",                     \
                 __FILE__,                                                     \
                 __LINE__,                                                     \
                 #cond,                                                        \
                 fmt::format(fmtstr, __VA_ARGS__));                            \
      std::fflush(stdout);                                                     \
      throw std::runtime_error("Unit test failure.");                          \
    }                                                                          \
  } while (false)

#define ASSERT_THROWS_ANY(expr)                                                \
  do                                                                           \
  {                                                                            \
    bool threw = false;                                                        \
    try                                                                        \
    {                                                                          \
      expr;                                                                    \
    }                                                                          \
    catch (...)                                                                \
    {                                                                          \
      threw = true;                                                            \
    }                                                                          \
    if (!threw)                                                                \
    {                                                                          \
      fmt::print("{}:{}: Expression expected exception: '{}'.",                \
                 __FILE__,                                                     \
                 __LINE__,                                                     \
                 #expr);                                                       \
      std::fflush(stdout);                                                     \
      throw std::runtime_error("Unit test failure.");                          \
    }                                                                          \
  } while (false)
