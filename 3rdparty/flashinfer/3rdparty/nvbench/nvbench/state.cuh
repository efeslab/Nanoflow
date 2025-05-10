/*
 *  Copyright 2021-2022 NVIDIA Corporation
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

#include <nvbench/cuda_stream.cuh>
#include <nvbench/device_info.cuh>
#include <nvbench/exec_tag.cuh>
#include <nvbench/named_values.cuh>
#include <nvbench/summary.cuh>
#include <nvbench/types.cuh>
#include <nvbench/stopping_criterion.cuh>

#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace nvbench
{

struct benchmark_base;

namespace detail
{
struct state_generator;
struct state_tester;
} // namespace detail

/**
 * Stores all information about a particular benchmark configuration.
 *
 * One state object exists for every combination of a benchmark's parameter
 * axes. It provides access to:
 * - Parameter values (get_int64, get_float64, get_string)
 *   - The names of parameters from type axes are stored as strings.
 * - Skip information (skip, is_skipped, get_skip_reason)
 *   - If the benchmark fails or is invalid, it may be skipped with an
 *     informative message.
 * - Summaries (add_summary, get_summary, get_summaries)
 *   - Summaries store measurement information as key/value pairs.
 *     See nvbench::summary for details.
 */
struct state
{
  // move-only
  state(const state &)            = delete;
  state(state &&)                 = default;
  state &operator=(const state &) = delete;
  state &operator=(state &&)      = default;

  [[nodiscard]] const nvbench::cuda_stream &get_cuda_stream() const { return m_cuda_stream; }
  void set_cuda_stream(nvbench::cuda_stream &&stream) { m_cuda_stream = std::move(stream); }

  /// The CUDA device associated with with this benchmark state. May be
  /// nullopt for CPU-only benchmarks.
  [[nodiscard]] const std::optional<nvbench::device_info> &get_device() const { return m_device; }

  /// An index into a benchmark::type_configs type_list. Returns 0 if no type
  /// axes in the associated benchmark.
  [[nodiscard]] std::size_t get_type_config_index() const { return m_type_config_index; }

  [[nodiscard]] nvbench::int64_t get_int64(const std::string &axis_name) const;
  [[nodiscard]] nvbench::int64_t get_int64_or_default(const std::string &axis_name,
                                                      nvbench::int64_t default_value) const;

  [[nodiscard]] nvbench::float64_t get_float64(const std::string &axis_name) const;
  [[nodiscard]] nvbench::float64_t get_float64_or_default(const std::string &axis_name,
                                                          nvbench::float64_t default_value) const;

  [[nodiscard]] const std::string &get_string(const std::string &axis_name) const;
  [[nodiscard]] const std::string &get_string_or_default(const std::string &axis_name,
                                                         const std::string &default_value) const;

  void add_element_count(std::size_t elements, std::string column_name = {});

  void set_element_count(std::size_t elements) { m_element_count = elements; }
  [[nodiscard]] std::size_t get_element_count() const { return m_element_count; }

  template <typename ElementType>
  void add_global_memory_reads(std::size_t count, std::string column_name = {})
  {
    this->add_global_memory_reads(count * sizeof(ElementType), std::move(column_name));
  }
  void add_global_memory_reads(std::size_t bytes, std::string column_name = {});

  template <typename ElementType>
  void add_global_memory_writes(std::size_t count, std::string column_name = {})
  {
    this->add_global_memory_writes(count * sizeof(ElementType), std::move(column_name));
  }
  void add_global_memory_writes(std::size_t bytes, std::string column_name = {});

  void add_buffer_size(std::size_t num_bytes,
                       std::string summary_tag,
                       std::string column_name = {},
                       std::string description = {});

  void set_global_memory_rw_bytes(std::size_t bytes) { m_global_memory_rw_bytes = bytes; }
  [[nodiscard]] std::size_t get_global_memory_rw_bytes() const { return m_global_memory_rw_bytes; }

  void skip(std::string reason) { m_skip_reason = std::move(reason); }
  [[nodiscard]] bool is_skipped() const { return !m_skip_reason.empty(); }
  [[nodiscard]] const std::string &get_skip_reason() const { return m_skip_reason; }

  /// Execute at least this many trials per measurement. @{
  [[nodiscard]] nvbench::int64_t get_min_samples() const { return m_min_samples; }
  void set_min_samples(nvbench::int64_t min_samples) { m_min_samples = min_samples; }
  /// @}

  [[nodiscard]] const nvbench::criterion_params &get_criterion_params() const
  {
    return m_criterion_params;
  }

  /// Control the stopping criterion for the measurement loop.
  /// @{
  [[nodiscard]] const std::string& get_stopping_criterion() const { return m_stopping_criterion; }
  void set_stopping_criterion(std::string criterion) { m_stopping_criterion = std::move(criterion); }
  /// @}

  /// If true, the benchmark is only run once, skipping all warmup runs and only
  /// executing a single non-batched measurement. This is intended for use with
  /// external profiling tools. @{
  [[nodiscard]] bool get_run_once() const { return m_run_once; }
  void set_run_once(bool v) { m_run_once = v; }
  /// @}

  /// If true, the benchmark does not use the blocking_kernel. This is intended
  /// for use with external profiling tools. @{
  [[nodiscard]] bool get_disable_blocking_kernel() const { return m_disable_blocking_kernel; }
  void set_disable_blocking_kernel(bool v) { m_disable_blocking_kernel = v; }
  /// @}

  /// Accumulate at least this many seconds of timing data per measurement. 
  /// Only applies to `stdrel` stopping criterion. @{
  [[nodiscard]] nvbench::float64_t get_min_time() const
  {
    return m_criterion_params.get_float64("min-time");
  }
  void set_min_time(nvbench::float64_t min_time)
  {
    m_criterion_params.set_float64("min-time", min_time);
  }
  /// @}

  /// Specify the maximum amount of noise if a measurement supports noise.
  /// Noise is the relative standard deviation:
  /// `noise = stdev / mean_time`.
  /// Only applies to `stdrel` stopping criterion. @{
  [[nodiscard]] nvbench::float64_t get_max_noise() const
  {
    return m_criterion_params.get_float64("max-noise");
  }
  void set_max_noise(nvbench::float64_t max_noise)
  {
    m_criterion_params.set_float64("max-noise", max_noise);
  }
  /// @}

  /// If a warmup run finishes in less than `skip_time`, the measurement will
  /// be skipped.
  /// Extremely fast kernels (< 5000 ns) often timeout before they can
  /// accumulate `min_time` measurements, and are often uninteresting. Setting
  /// this value can help improve performance by skipping time consuming
  /// measurement that don't provide much information.
  /// Default value is -1., which disables the feature.
  /// @{
  [[nodiscard]] nvbench::float64_t get_skip_time() const { return m_skip_time; }
  void set_skip_time(nvbench::float64_t skip_time) { m_skip_time = skip_time; }
  /// @}

  /// If a measurement take more than `timeout` seconds to complete, stop the
  /// measurement early. A warning should be printed if this happens.
  /// This setting overrides all other termination criteria.
  /// Note that this is measured in CPU walltime, not sample time.
  /// @{
  [[nodiscard]] nvbench::float64_t get_timeout() const { return m_timeout; }
  void set_timeout(nvbench::float64_t timeout) { m_timeout = timeout; }
  /// @}

  /// If a `KernelLauncher` syncs and `nvbench::exec_tag::sync` is not passed
  /// to `state.exec(...)`, a deadlock may occur. If a `blocking_kernel` blocks
  /// for more than `blocking_kernel_timeout` seconds, an error will be printed
  /// and the kernel will unblock to prevent deadlocks.
  /// A negative value disables the timeout.
  /// @{
  [[nodiscard]] nvbench::float64_t get_blocking_kernel_timeout() const
  {
    return m_blocking_kernel_timeout;
  }
  void set_blocking_kernel_timeout(nvbench::float64_t timeout)
  {
    m_blocking_kernel_timeout = timeout;
  }
  ///@}

  [[nodiscard]] const named_values &get_axis_values() const { return m_axis_values; }

  /*!
   * Return a string of "axis_name1=input_string1 axis_name2=input_string2 ..."
   */
  [[nodiscard]] std::string get_axis_values_as_string(bool color = false) const;

  [[nodiscard]] const benchmark_base &get_benchmark() const { return m_benchmark; }

  void collect_l1_hit_rates() { m_collect_l1_hit_rates = true; }
  void collect_l2_hit_rates() { m_collect_l2_hit_rates = true; }
  void collect_stores_efficiency() { m_collect_stores_efficiency = true; }
  void collect_loads_efficiency() { m_collect_loads_efficiency = true; }
  void collect_dram_throughput() { m_collect_dram_throughput = true; }

  void collect_cupti_metrics()
  {
    collect_l1_hit_rates();
    collect_l2_hit_rates();
    collect_stores_efficiency();
    collect_loads_efficiency();
    collect_dram_throughput();
  }

  [[nodiscard]] bool is_l1_hit_rate_collected() const { return m_collect_l1_hit_rates; }
  [[nodiscard]] bool is_l2_hit_rate_collected() const { return m_collect_l2_hit_rates; }
  [[nodiscard]] bool is_stores_efficiency_collected() const { return m_collect_stores_efficiency; }
  [[nodiscard]] bool is_loads_efficiency_collected() const { return m_collect_loads_efficiency; }
  [[nodiscard]] bool is_dram_throughput_collected() const { return m_collect_dram_throughput; }

  [[nodiscard]] bool is_cupti_required() const
  {
    // clang-format off
    return is_l2_hit_rate_collected() ||
           is_l1_hit_rate_collected() ||
           is_stores_efficiency_collected() ||
           is_loads_efficiency_collected() ||
           is_dram_throughput_collected();
    // clang-format on
  }

  summary &add_summary(std::string summary_tag);
  summary &add_summary(summary s);
  [[nodiscard]] const summary &get_summary(std::string_view tag) const;
  [[nodiscard]] summary &get_summary(std::string_view tag);
  [[nodiscard]] const std::vector<summary> &get_summaries() const;
  [[nodiscard]] std::vector<summary> &get_summaries();

  /// A single line description of the state:
  ///
  /// ```
  /// <bench_name> [<parameters>]
  /// ```
  [[nodiscard]] std::string get_short_description(bool color = false) const;

  // TODO This will need detailed docs and include a reference to an appropriate
  // section of the user's guide
  template <typename ExecTags, typename KernelLauncher>
  void exec(ExecTags, KernelLauncher &&kernel_launcher);

  template <typename KernelLauncher>
  void exec(KernelLauncher &&kernel_launcher)
  {
    this->exec(nvbench::exec_tag::none, std::forward<KernelLauncher>(kernel_launcher));
  }

private:
  friend struct nvbench::detail::state_generator;
  friend struct nvbench::detail::state_tester;

  explicit state(const benchmark_base &bench);

  state(const benchmark_base &bench,
        nvbench::named_values values,
        std::optional<nvbench::device_info> device,
        std::size_t type_config_index);

  nvbench::cuda_stream m_cuda_stream;
  std::reference_wrapper<const nvbench::benchmark_base> m_benchmark;
  nvbench::named_values m_axis_values;
  std::optional<nvbench::device_info> m_device;
  std::size_t m_type_config_index{};

  bool m_run_once{false};
  bool m_disable_blocking_kernel{false};


  nvbench::criterion_params m_criterion_params;
  std::string m_stopping_criterion;

  nvbench::int64_t m_min_samples;

  nvbench::float64_t m_skip_time;
  nvbench::float64_t m_timeout;

  // Deadlock protection. See blocking_kernel's class doc for details.
  nvbench::float64_t m_blocking_kernel_timeout{30.0};

  std::vector<nvbench::summary> m_summaries;
  std::string m_skip_reason;
  std::size_t m_element_count{};
  std::size_t m_global_memory_rw_bytes{};

  bool m_collect_l1_hit_rates{};
  bool m_collect_l2_hit_rates{};
  bool m_collect_stores_efficiency{};
  bool m_collect_loads_efficiency{};
  bool m_collect_dram_throughput{};
};

} // namespace nvbench

#define NVBENCH_STATE_EXEC_GUARD
#include <nvbench/detail/state_exec.cuh>
#undef NVBENCH_STATE_EXEC_GUARD
