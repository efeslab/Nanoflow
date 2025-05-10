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

#include <nvbench/config.cuh>
#include <nvbench/detail/throw.cuh>

#include <fmt/format.h>

#ifdef NVBENCH_HAS_NVML
#include <nvml.h>
#endif // NVBENCH_HAS_NVML

#include <stdexcept>

namespace nvbench::nvml
{

// RAII struct that initializes and shuts down NVML
// Needs to be constructed and kept alive while using nvml
struct NVMLLifetimeManager
{
  NVMLLifetimeManager();
  ~NVMLLifetimeManager();
private:
  bool m_inited{false};
};

/// Base class for NVML-specific exceptions
struct error : std::runtime_error
{
  using runtime_error::runtime_error;
};

/// Thrown when NVML support is disabled.
struct not_enabled : error
{
  not_enabled()
      : error{"NVML not available. Reconfigure NVBench with the CMake option "
              "`-DNVBench_ENABLE_NVML=ON`."}
  {}
};

// Only `error` and `not_enabled` are defined when NVML is disabled.
// Other exceptions may hold types defined by NVML.
#ifdef NVBENCH_HAS_NVML

/// Thrown when a generic NVML call inside NVBENCH_NVML_CALL fails
struct call_failed : error
{
  call_failed(const std::string &filename,
              std::size_t lineno,
              const std::string &call,
              nvmlReturn_t error_code,
              std::string error_string)
      : error(fmt::format("{}:{}:\n"
                          "\tNVML call failed:\n"
                          "\t\tCall: {}\n"
                          "\t\tError: ({}) {}",
                          filename,
                          lineno,
                          call,
                          static_cast<int>(error_code),
                          error_string))
      , m_error_code(error_code)
      , m_error_string(error_string)
  {}

  [[nodiscard]] nvmlReturn_t get_error_code() const { return m_error_code; }

  [[nodiscard]] const std::string &get_error_string() const { return m_error_string; }

private:
  nvmlReturn_t m_error_code;
  std::string m_error_string;
};

#endif // NVBENCH_HAS_NVML

} // namespace nvbench::nvml

#ifdef NVBENCH_HAS_NVML

#define NVBENCH_NVML_CALL(call)                                                                    \
  do                                                                                               \
  {                                                                                                \
    const auto _rr = call;                                                                         \
    if (_rr != NVML_SUCCESS)                                                                       \
    {                                                                                              \
      throw nvbench::nvml::call_failed(__FILE__, __LINE__, #call, _rr, nvmlErrorString(_rr));      \
    }                                                                                              \
  } while (false)

// Same as above, but used for nvmlInit(), where a failure means that
// nvmlErrorString is not available.
#define NVBENCH_NVML_CALL_NO_API(call)                                                             \
  do                                                                                               \
  {                                                                                                \
    const auto _rr = call;                                                                         \
    if (_rr != NVML_SUCCESS)                                                                       \
    {                                                                                              \
      throw nvbench::nvml::call_failed(__FILE__, __LINE__, #call, _rr, "");                        \
    }                                                                                              \
  } while (false)

#endif // NVBENCH_HAS_NVML
