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

#include <nvbench/blocking_kernel.cuh>
#include <nvbench/cpu_timer.cuh>
#include <nvbench/cuda_call.cuh>
#include <nvbench/cuda_timer.cuh>
#include <nvbench/exec_tag.cuh>
#include <nvbench/launch.cuh>

#include <cuda_runtime.h>

#include <algorithm>

namespace nvbench
{

struct state;

namespace detail
{

// non-templated code goes here to keep instantiation cost down:
struct measure_hot_base
{
  explicit measure_hot_base(nvbench::state &exec_state);
  measure_hot_base(const measure_hot_base &)            = delete;
  measure_hot_base(measure_hot_base &&)                 = delete;
  measure_hot_base &operator=(const measure_hot_base &) = delete;
  measure_hot_base &operator=(measure_hot_base &&)      = delete;

protected:
  void check();

  void initialize()
  {
    m_total_cuda_time   = 0.;
    m_total_samples     = 0;
    m_max_time_exceeded = false;
  }

  void generate_summaries();

  void check_skip_time(nvbench::float64_t warmup_time);

  void block_stream();

  __forceinline__ void unblock_stream() { m_blocker.unblock(); }

  nvbench::state &m_state;

  nvbench::launch m_launch;
  nvbench::cuda_timer m_cuda_timer;
  nvbench::cpu_timer m_walltime_timer;
  nvbench::blocking_kernel m_blocker;

  nvbench::int64_t m_min_samples{};
  nvbench::float64_t m_min_time{};

  nvbench::float64_t m_skip_time{};
  nvbench::float64_t m_timeout{};

  nvbench::int64_t m_total_samples{};
  nvbench::float64_t m_total_cuda_time{};

  bool m_max_time_exceeded{false};
};

template <typename KernelLauncher, bool use_blocking_kernel>
struct measure_hot : public measure_hot_base
{
  measure_hot(nvbench::state &state, KernelLauncher &kernel_launcher)
      : measure_hot_base(state)
      , m_kernel_launcher{kernel_launcher}
  {}

  void operator()()
  {
    this->check();
    this->initialize();
    this->run_warmup();
    this->run_trials();
    this->generate_summaries();
  }

private:
  // Run the kernel once, measuring the GPU time. If under skip_time, skip the
  // measurement.
  void run_warmup()
  {
    if constexpr (use_blocking_kernel)
    {
      this->block_stream();
    }

    m_cuda_timer.start(m_launch.get_stream());
    this->launch_kernel();
    m_cuda_timer.stop(m_launch.get_stream());

    if constexpr (use_blocking_kernel)
    {
      this->unblock_stream();
    }
    this->sync_stream();

    this->check_skip_time(m_cuda_timer.get_duration());
  }

  void run_trials()
  {
    m_walltime_timer.start();

    // Use warmup results to estimate the number of iterations to run.
    // The .95 factor here pads the batch_size a bit to avoid needing a second
    // batch due to noise.
    const auto time_estimate = m_cuda_timer.get_duration() * 0.95;
    auto batch_size          = static_cast<nvbench::int64_t>(m_min_time / time_estimate);

    do
    {
      batch_size = std::max(batch_size, nvbench::int64_t{1});

      if constexpr (use_blocking_kernel)
      {
        // Block stream until some work is queued.
        // Limit the number of kernel executions while blocked to prevent
        // deadlocks. See warnings on blocking_kernel.
        const auto blocked_launches   = std::min(batch_size, nvbench::int64_t{2});
        const auto unblocked_launches = batch_size - blocked_launches;

        this->block_stream();
        m_cuda_timer.start(m_launch.get_stream());

        for (nvbench::int64_t i = 0; i < blocked_launches; ++i)
        {
          // If your benchmark deadlocks in the next launch, reduce the size of
          // blocked_launches. See note above.
          this->launch_kernel();
        }

        this->unblock_stream(); // Start executing earlier launches

        for (nvbench::int64_t i = 0; i < unblocked_launches; ++i)
        {
          this->launch_kernel();
        }
      }
      else
      {
        m_cuda_timer.start(m_launch.get_stream());

        for (nvbench::int64_t i = 0; i < batch_size; ++i)
        {
          this->launch_kernel();
        }
      }

      m_cuda_timer.stop(m_launch.get_stream());
      this->sync_stream();

      m_total_cuda_time += m_cuda_timer.get_duration();
      m_total_samples += batch_size;

      // Predict number of remaining iterations:
      batch_size = static_cast<nvbench::int64_t>(
        (m_min_time - m_total_cuda_time) /
        (m_total_cuda_time / static_cast<nvbench::float64_t>(m_total_samples)));

      if (m_total_cuda_time > m_min_time && // min time okay
          m_total_samples > m_min_samples)  // min samples okay
      {
        break; // Stop iterating
      }

      m_walltime_timer.stop();
      if (m_walltime_timer.get_duration() > m_timeout)
      {
        m_max_time_exceeded = true;
        break;
      }
    } while (true);

    m_walltime_timer.stop();
  }

  __forceinline__ void launch_kernel() { m_kernel_launcher(m_launch); }

  __forceinline__ void sync_stream() const
  {
    NVBENCH_CUDA_CALL(cudaStreamSynchronize(m_launch.get_stream()));
  }

  KernelLauncher &m_kernel_launcher;
};

} // namespace detail
} // namespace nvbench
