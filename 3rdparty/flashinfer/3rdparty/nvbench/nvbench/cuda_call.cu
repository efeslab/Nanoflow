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

#include <nvbench/config.cuh>
#include <nvbench/cuda_call.cuh>

#include <fmt/format.h>

#include <cstdlib>
#include <stdexcept>
#include <string>

namespace nvbench
{

namespace cuda_call
{

void throw_error(const std::string &filename,
                 std::size_t lineno,
                 const std::string &command,
                 cudaError_t error_code)
{
  throw std::runtime_error(fmt::format("{}:{}: Cuda API call returned error: "
                                       "{}: {}\nCommand: '{}'",
                                       filename,
                                       lineno,
                                       cudaGetErrorName(error_code),
                                       cudaGetErrorString(error_code),
                                       command));
}

#ifdef NVBENCH_HAS_CUPTI
void throw_error(const std::string &filename,
                 std::size_t lineno,
                 const std::string &command,
                 CUresult error_code)
{
  const char *name = nullptr;
  cuGetErrorName(error_code, &name);

  const char *string = nullptr;
  cuGetErrorString(error_code, &string);

  throw std::runtime_error(fmt::format("{}:{}: Driver API call returned error: "
                                       "{}: {}\nCommand: '{}'",
                                       filename,
                                       lineno,
                                       name,
                                       string,
                                       command));
}
#else
void throw_error(const std::string &, std::size_t, const std::string &, CUresult) {}
#endif

void exit_error(const std::string &filename,
                std::size_t lineno,
                const std::string &command,
                cudaError_t error_code)
{
  fmt::print(stderr,
             "{}:{}: Cuda API call returned error: {}: {}\nCommand: '{}'",
             filename,
             lineno,
             cudaGetErrorName(error_code),
             cudaGetErrorString(error_code),
             command);
  std::exit(EXIT_FAILURE);
}

} // namespace cuda_call

} // namespace nvbench
