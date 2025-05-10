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
#include <nvbench/criterion_manager.cuh>
#include <nvbench/detail/measure_cold.cuh>
#include <nvbench/detail/throw.cuh>
#include <nvbench/device_info.cuh>
#include <nvbench/printer_base.cuh>
#include <nvbench/state.cuh>
#include <nvbench/summary.cuh>

#include <fmt/format.h>

namespace nvbench::detail
{

measure_cold_base::measure_cold_base(state &exec_state)
    : m_state{exec_state}
    , m_launch{m_state.get_cuda_stream()}
    , m_criterion_params{exec_state.get_criterion_params()}
    , m_stopping_criterion{nvbench::criterion_manager::get().get_criterion(exec_state.get_stopping_criterion())}
    , m_run_once{exec_state.get_run_once()}
    , m_no_block{exec_state.get_disable_blocking_kernel()}
    , m_min_samples{exec_state.get_min_samples()}
    , m_skip_time{exec_state.get_skip_time()}
    , m_timeout{exec_state.get_timeout()}
{
  if (m_min_samples > 0)
  {
    m_cuda_times.reserve(static_cast<std::size_t>(m_min_samples));
    m_cpu_times.reserve(static_cast<std::size_t>(m_min_samples));
  }
}

void measure_cold_base::check()
{
  const auto device = m_state.get_device();
  if (!device)
  {
    NVBENCH_THROW(std::runtime_error, "{}", "Device required for `cold` measurement.");
  }
  if (!device->is_active())
  { // This means something went wrong higher up. Throw an error.
    NVBENCH_THROW(std::runtime_error, "{}", "Internal error: Current device is not active.");
  }
}

void measure_cold_base::initialize()
{
  m_total_cuda_time = 0.;
  m_total_cpu_time  = 0.;
  m_cpu_noise       = 0.;
  m_total_samples   = 0;
  m_cuda_times.clear();
  m_cpu_times.clear();
  m_max_time_exceeded = false;

  m_stopping_criterion.initialize(m_criterion_params);
}

void measure_cold_base::run_trials_prologue() { m_walltime_timer.start(); }

void measure_cold_base::record_measurements()
{
  // Update and record timers and counters:
  const auto cur_cuda_time = m_cuda_timer.get_duration();
  const auto cur_cpu_time  = m_cpu_timer.get_duration();
  m_cuda_times.push_back(cur_cuda_time);
  m_cpu_times.push_back(cur_cpu_time);
  m_total_cuda_time += cur_cuda_time;
  m_total_cpu_time += cur_cpu_time;
  ++m_total_samples;

  m_stopping_criterion.add_measurement(cur_cuda_time);
}

bool measure_cold_base::is_finished()
{
  if (m_run_once)
  {
    return true;
  }

  // Check that we've gathered enough samples:
  if (m_total_samples > m_min_samples)
  {
    if (m_stopping_criterion.is_finished())
    {
      return true;
    }
  }

  // Check for timeouts:
  m_walltime_timer.stop();
  if (m_walltime_timer.get_duration() > m_timeout)
  {
    m_max_time_exceeded = true;
    return true;
  }

  return false;
}

void measure_cold_base::run_trials_epilogue()
{
  // Only need to compute this at the end, not per iteration.
  const auto cpu_mean  = m_total_cuda_time / static_cast<nvbench::float64_t>(m_total_samples);
  const auto cpu_stdev = nvbench::detail::statistics::standard_deviation(m_cpu_times.cbegin(),
                                                                         m_cpu_times.cend(),
                                                                         cpu_mean);
  m_cpu_noise          = cpu_stdev / cpu_mean;

  m_walltime_timer.stop();
}

void measure_cold_base::generate_summaries()
{
  const auto d_samples = static_cast<double>(m_total_samples);
  {
    auto &summ = m_state.add_summary("nv/cold/sample_size");
    summ.set_string("name", "Samples");
    summ.set_string("hint", "sample_size");
    summ.set_string("description", "Number of isolated kernel executions");
    summ.set_int64("value", m_total_samples);
  }

  const auto avg_cpu_time = m_total_cpu_time / d_samples;
  {
    auto &summ = m_state.add_summary("nv/cold/time/cpu/mean");
    summ.set_string("name", "CPU Time");
    summ.set_string("hint", "duration");
    summ.set_string("description",
                    "Mean isolated kernel execution time "
                    "(measured on host CPU)");
    summ.set_float64("value", avg_cpu_time);
  }

  {
    auto &summ = m_state.add_summary("nv/cold/time/cpu/stdev/relative");
    summ.set_string("name", "Noise");
    summ.set_string("hint", "percentage");
    summ.set_string("description", "Relative standard deviation of isolated CPU times");
    summ.set_float64("value", m_cpu_noise);
  }

  const auto avg_cuda_time = m_total_cuda_time / d_samples;
  {
    auto &summ = m_state.add_summary("nv/cold/time/gpu/mean");
    summ.set_string("name", "GPU Time");
    summ.set_string("hint", "duration");
    summ.set_string("description",
                    "Mean isolated kernel execution time "
                    "(measured with CUDA events)");
    summ.set_float64("value", avg_cuda_time);
  }

  const auto mean_cuda_time = m_total_cuda_time / static_cast<nvbench::float64_t>(m_total_samples);
  const auto cuda_stdev     = nvbench::detail::statistics::standard_deviation(m_cuda_times.cbegin(),
                                                                          m_cuda_times.cend(),
                                                                          mean_cuda_time);
  const auto cuda_rel_stdev = cuda_stdev / mean_cuda_time;
  const auto noise = cuda_rel_stdev;
  const auto max_noise = m_criterion_params.get_float64("max-noise");
  const auto min_time = m_criterion_params.get_float64("min-time");

  {
    auto &summ = m_state.add_summary("nv/cold/time/gpu/stdev/relative");
    summ.set_string("name", "Noise");
    summ.set_string("hint", "percentage");
    summ.set_string("description", "Relative standard deviation of isolated GPU times");
    summ.set_float64("value", noise);
  }

  if (const auto items = m_state.get_element_count(); items != 0)
  {
    auto &summ = m_state.add_summary("nv/cold/bw/item_rate");
    summ.set_string("name", "Elem/s");
    summ.set_string("hint", "item_rate");
    summ.set_string("description", "Number of input elements processed per second");
    summ.set_float64("value", static_cast<double>(items) / avg_cuda_time);
  }

  if (const auto bytes = m_state.get_global_memory_rw_bytes(); bytes != 0)
  {
    const auto avg_used_gmem_bw = static_cast<double>(bytes) / avg_cuda_time;
    {
      auto &summ = m_state.add_summary("nv/cold/bw/global/bytes_per_second");
      summ.set_string("name", "GlobalMem BW");
      summ.set_string("hint", "byte_rate");
      summ.set_string("description",
                      "Number of bytes read/written per second to the CUDA "
                      "device's global memory");
      summ.set_float64("value", avg_used_gmem_bw);
    }

    {
      const auto peak_gmem_bw =
        static_cast<double>(m_state.get_device()->get_global_memory_bus_bandwidth());

      auto &summ = m_state.add_summary("nv/cold/bw/global/utilization");
      summ.set_string("name", "BWUtil");
      summ.set_string("hint", "percentage");
      summ.set_string("description",
                      "Global device memory utilization as a percentage of the "
                      "device's peak bandwidth");
      summ.set_float64("value", avg_used_gmem_bw / peak_gmem_bw);
    }
  } // bandwidth

  {
    auto &summ = m_state.add_summary("nv/cold/walltime");
    summ.set_string("name", "Walltime");
    summ.set_string("hint", "duration");
    summ.set_string("description", "Walltime used for isolated measurements");
    summ.set_float64("value", m_walltime_timer.get_duration());
    summ.set_string("hide", "Hidden by default.");
  }

  // Log if a printer exists:
  if (auto printer_opt_ref = m_state.get_benchmark().get_printer(); printer_opt_ref.has_value())
  {
    auto &printer = printer_opt_ref.value().get();

    if (m_max_time_exceeded)
    {
      const auto timeout = m_walltime_timer.get_duration();

      if (noise > max_noise)
      {
        printer.log(nvbench::log_level::warn,
                    fmt::format("Current measurement timed out ({:0.2f}s) "
                                "while over noise threshold ({:0.2f}% > "
                                "{:0.2f}%)",
                                timeout,
                                noise * 100,
                                max_noise * 100));
      }
      if (m_total_samples < m_min_samples)
      {
        printer.log(nvbench::log_level::warn,
                    fmt::format("Current measurement timed out ({:0.2f}s) "
                                "before accumulating min_samples ({} < {})",
                                timeout,
                                m_total_samples,
                                m_min_samples));
      }
      if (m_total_cuda_time < min_time)
      {
        printer.log(nvbench::log_level::warn,
                    fmt::format("Current measurement timed out ({:0.2f}s) "
                                "before accumulating min_time ({:0.2f}s < "
                                "{:0.2f}s)",
                                timeout,
                                m_total_cuda_time,
                                min_time));
      }
    }

    // Log to stdout:
    printer.log(nvbench::log_level::pass,
                fmt::format("Cold: {:0.6f}ms GPU, {:0.6f}ms CPU, {:0.2f}s "
                            "total GPU, {:0.2f}s total wall, {}x ",
                            avg_cuda_time * 1e3,
                            avg_cpu_time * 1e3,
                            m_total_cuda_time,
                            m_walltime_timer.get_duration(),
                            m_total_samples));

    printer.process_bulk_data(m_state, "nv/cold/sample_times", "sample_times", m_cuda_times);
  }
}

void measure_cold_base::check_skip_time(nvbench::float64_t warmup_time)
{
  if (m_skip_time > 0. && warmup_time < m_skip_time)
  {
    auto reason = fmt::format("Warmup time did not meet skip_time limit: "
                              "{:0.3f}us < {:0.3f}us.",
                              warmup_time * 1e6,
                              m_skip_time * 1e6);

    m_state.skip(reason);
    NVBENCH_THROW(std::runtime_error, "{}", std::move(reason));
  }
}

void measure_cold_base::block_stream()
{
  m_blocker.block(m_launch.get_stream(), m_state.get_blocking_kernel_timeout());
}

} // namespace nvbench::detail
