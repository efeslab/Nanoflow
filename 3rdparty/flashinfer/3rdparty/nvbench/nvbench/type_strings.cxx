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

#include <nvbench/type_strings.cuh>

#include <fmt/format.h>

#include <string>

#if defined(__GNUC__) || defined(__clang__)
#define NVBENCH_CXXABI_DEMANGLE
#endif

#ifdef NVBENCH_CXXABI_DEMANGLE
#include <cxxabi.h>

#include <cstdlib>
#include <memory>

namespace
{
struct free_wrapper
{
  void operator()(void *ptr) { std::free(ptr); }
};
} // end namespace

#endif // NVBENCH_CXXABI_DEMANGLE

namespace nvbench
{

std::string demangle(const std::string &str)
{
#ifdef NVBENCH_CXXABI_DEMANGLE
  std::unique_ptr<char, free_wrapper> demangled{
    abi::__cxa_demangle(str.c_str(), nullptr, nullptr, nullptr)};
  return std::string(demangled.get());
#else
  return str;
#endif
};

} // namespace nvbench
