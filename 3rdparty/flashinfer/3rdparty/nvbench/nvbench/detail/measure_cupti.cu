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

#include <nvbench/benchmark_base.cuh>
#include <nvbench/detail/measure_cupti.cuh>
#include <nvbench/detail/throw.cuh>
#include <nvbench/printer_base.cuh>
#include <nvbench/state.cuh>

#include <fmt/format.h>

#include <algorithm>
#include <stdexcept>

namespace nvbench::detail
{

namespace
{

enum class metric_id : int
{
  dram_peak_sustained_throughput = 0,
  global_load_efficiency,
  global_store_efficiency,
  l1_hit_rate,
  l2_hit_rate,

  count
};

template <metric_id id>
struct metric_traits;

template <>
struct metric_traits<metric_id::dram_peak_sustained_throughput>
{
  static constexpr const char *metric_name = "dram__throughput.avg.pct_of_peak_sustained_elapsed";

  static constexpr const char *name = "HBWPeak";
  static constexpr const char *hint = "percentage";

  static constexpr const char *description =
    "The utilization level of the device memory relative to the peak "
    "utilization.";

  static constexpr double divider = 100.0;

  static bool is_collected(nvbench::state &m_state)
  {
    return m_state.is_dram_throughput_collected();
  };
};

template <>
struct metric_traits<metric_id::global_load_efficiency>
{
  static constexpr const char *metric_name =
    "smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct";

  static constexpr const char *name = "LoadEff";
  static constexpr const char *hint = "percentage";

  static constexpr const char *description =
    "Ratio of requested global memory load throughput to required global "
    "memory load throughput expressed as percentage.";

  static constexpr double divider = 100.0;

  static bool is_collected(nvbench::state &m_state)
  {
    return m_state.is_loads_efficiency_collected();
  };
};

template <>
struct metric_traits<metric_id::global_store_efficiency>
{
  static constexpr const char *metric_name =
    "smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct";

  static constexpr const char *name = "StoreEff";
  static constexpr const char *hint = "percentage";

  static constexpr const char *description =
    "Ratio of requested global memory store throughput to required global "
    "memory store throughput expressed as percentage.";

  static constexpr double divider = 100.0;

  static bool is_collected(nvbench::state &m_state)
  {
    return m_state.is_stores_efficiency_collected();
  };
};

template <>
struct metric_traits<metric_id::l1_hit_rate>
{
  static constexpr const char *metric_name = "l1tex__t_sector_hit_rate.pct";
  static constexpr const char *name        = "L1HitRate";
  static constexpr const char *hint        = "percentage";
  static constexpr const char *description = "Hit rate at L1 cache.";
  static constexpr double divider          = 100.0;

  static bool is_collected(nvbench::state &m_state) { return m_state.is_l1_hit_rate_collected(); };
};

template <>
struct metric_traits<metric_id::l2_hit_rate>
{
  static constexpr const char *metric_name = "lts__t_sector_hit_rate.pct";
  static constexpr const char *name        = "L2HitRate";
  static constexpr const char *hint        = "percentage";
  static constexpr const char *description = "Hit rate at L2 cache.";
  static constexpr double divider          = 100.0;

  static bool is_collected(nvbench::state &m_state) { return m_state.is_l2_hit_rate_collected(); };
};

template <metric_id id = metric_id::dram_peak_sustained_throughput>
void add_metrics_impl(nvbench::state &state, std::vector<std::string> &metrics)
{
  if (metric_traits<id>::is_collected(state))
  {
    metrics.emplace_back(metric_traits<id>::metric_name);
  }

  constexpr auto next_id = static_cast<metric_id>(static_cast<int>(id) + 1);
  add_metrics_impl<next_id>(state, metrics);
}

template <>
void add_metrics_impl<metric_id::count>(nvbench::state &, std::vector<std::string> &)
{}

std::vector<std::string> add_metrics(nvbench::state &state)
{
  std::vector<std::string> metrics;
  metrics.reserve(static_cast<int>(metric_id::count));

  add_metrics_impl(state, metrics);
  return metrics;
}

} // namespace

measure_cupti_base::measure_cupti_base(state &exec_state)
// clang-format off
// (formatter doesn't handle `try :` very well...)
try
  : m_state{exec_state}
  , m_launch{m_state.get_cuda_stream()}
  , m_cupti{*m_state.get_device(), add_metrics(m_state)}
{}
// clang-format on
catch (const std::exception &ex)
{
  if (auto printer_opt_ref = exec_state.get_benchmark().get_printer(); printer_opt_ref)
  {
    auto &printer = printer_opt_ref.value().get();
    printer.log(nvbench::log_level::warn,
                fmt::format("CUPTI failed to construct profiler: {}", ex.what()));
  }
}

void measure_cupti_base::check()
{
  const auto device = m_state.get_device();
  if (!device)
  {
    NVBENCH_THROW(std::runtime_error, "{}", "Device required for `cupti` measurement.");
  }
  if (!device->is_active())
  { // This means something went wrong higher up. Throw an error.
    NVBENCH_THROW(std::runtime_error, "{}", "Internal error: Current device is not active.");
  }
}

namespace
{

template <metric_id id = metric_id::dram_peak_sustained_throughput>
void gen_summary(std::size_t result_id, nvbench::state &m_state, const std::vector<double> &result)
{
  using metric = metric_traits<id>;

  if (metric::is_collected(m_state))
  {
    auto &summ = m_state.add_summary(fmt::format("nv/cupti/{}", metric::metric_name));
    summ.set_string("name", metric::name);
    summ.set_string("hint", metric::hint);
    summ.set_string("description", metric::description);
    summ.set_float64("value", result[result_id++] / metric::divider);
  }

  constexpr auto next_id = static_cast<metric_id>(static_cast<int>(id) + 1);
  gen_summary<next_id>(result_id, m_state, result);
}

template <>
void gen_summary<metric_id::count>(std::size_t, nvbench::state &, const std::vector<double> &)
{}

void gen_summaries(nvbench::state &state, const std::vector<double> &result)
{
  gen_summary(0, state, result);
}

} // namespace

void measure_cupti_base::generate_summaries()
try
{
  gen_summaries(m_state, m_cupti.get_counter_values());

  {
    auto &summ = m_state.add_summary("nv/cupti/sample_size");
    summ.set_string("name", "Samples");
    summ.set_string("hint", "sample_size");
    summ.set_string("description", "Number of CUPTI kernel executions");
    summ.set_int64("value", m_total_samples);
  }

  {
    auto &summ = m_state.add_summary("nv/cupti/walltime");
    summ.set_string("name", "Walltime");
    summ.set_string("hint", "duration");
    summ.set_string("description", "Walltime used for CUPTI measurements");
    summ.set_float64("value", m_walltime_timer.get_duration());
    summ.set_string("hide", "Hidden by default.");
  }

  // Log if a printer exists:
  if (auto printer_opt_ref = m_state.get_benchmark().get_printer(); printer_opt_ref.has_value())
  {
    auto &printer = printer_opt_ref.value().get();
    printer.log(nvbench::log_level::pass,
                fmt::format("CUPTI: {:0.2f}s total wall, {}x",
                            m_walltime_timer.get_duration(),
                            m_total_samples));
  }
}
catch (const std::exception &ex)
{
  if (auto printer_opt_ref = m_state.get_benchmark().get_printer(); printer_opt_ref)
  {
    auto &printer = printer_opt_ref.value().get();
    printer.log(nvbench::log_level::warn,
                fmt::format("CUPTI failed to generate the summary: {}", ex.what()));
  }
}

} // namespace nvbench::detail
