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

#include <nvbench/benchmark_base.cuh>

#include <memory>
#include <vector>

namespace nvbench
{

/**
 * Singleton class that owns reference copies of all known benchmarks.
 */
struct benchmark_manager
{
  using benchmark_vector = std::vector<std::unique_ptr<nvbench::benchmark_base>>;

  /**
   * @return The singleton benchmark_manager instance.
   */
  [[nodiscard]] static benchmark_manager &get();

  /**
   * Setup any default values for the benchmarks. Invoked from `main`.
   *
   * Specifically, any CUDA calls (e.g. cudaGetDeviceProperties, etc) needed to initialize the
   * benchmarks should be done here to avoid creating a CUDA context before we configure the CUDA
   * environment in `main`.
   */
   void initialize();

  /**
   * Register a new benchmark.
   */
  benchmark_base &add(std::unique_ptr<benchmark_base> bench);

  /**
   * Clone all benchmarks in the manager into the returned vector.
   */
  [[nodiscard]] benchmark_vector clone_benchmarks() const;

  /**
   * Get a non-mutable reference to benchmark with the specified name/index.
   * @{
   */
  [[nodiscard]] const benchmark_base &get_benchmark(const std::string &name) const;
  [[nodiscard]] const benchmark_base &get_benchmark(std::size_t idx) const
  {
    return *m_benchmarks.at(idx);
  }
  /**@}*/

  [[nodiscard]] const benchmark_vector &get_benchmarks() const { return m_benchmarks; };

private:
  benchmark_manager()                                     = default;
  benchmark_manager(const benchmark_manager &)            = delete;
  benchmark_manager(benchmark_manager &&)                 = delete;
  benchmark_manager &operator=(const benchmark_manager &) = delete;
  benchmark_manager &operator=(benchmark_manager &&)      = delete;

  benchmark_vector m_benchmarks;
};

} // namespace nvbench
