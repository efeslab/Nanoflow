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

#include <chrono>

namespace nvbench
{

struct cpu_timer
{
  __forceinline__ cpu_timer() = default;

  // move-only
  cpu_timer(const cpu_timer &)            = delete;
  cpu_timer(cpu_timer &&)                 = default;
  cpu_timer &operator=(const cpu_timer &) = delete;
  cpu_timer &operator=(cpu_timer &&)      = default;

  __forceinline__ void start() { m_start = std::chrono::high_resolution_clock::now(); }

  __forceinline__ void stop() { m_stop = std::chrono::high_resolution_clock::now(); }

  // In seconds:
  [[nodiscard]] __forceinline__ nvbench::float64_t get_duration()
  {
    const auto duration = m_stop - m_start;
    const auto ns       = std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
    return static_cast<nvbench::float64_t>(ns) * (1e-9);
  }

private:
  std::chrono::high_resolution_clock::time_point m_start;
  std::chrono::high_resolution_clock::time_point m_stop;
};

} // namespace nvbench
