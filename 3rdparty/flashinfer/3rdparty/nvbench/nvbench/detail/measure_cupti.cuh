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
#include <nvbench/config.cuh>
#include <nvbench/cpu_timer.cuh>
#include <nvbench/cuda_call.cuh>
#include <nvbench/cuda_timer.cuh>
#include <nvbench/cupti_profiler.cuh>
#include <nvbench/device_info.cuh>
#include <nvbench/exec_tag.cuh>
#include <nvbench/launch.cuh>

#include <nvbench/detail/kernel_launcher_timer_wrapper.cuh>
#include <nvbench/detail/l2flush.cuh>
#include <nvbench/detail/statistics.cuh>

#include <cuda_runtime.h>

#include <algorithm>
#include <utility>
#include <vector>

namespace nvbench
{

struct state;

namespace detail
{

// non-templated code goes here:
struct measure_cupti_base
{
  explicit measure_cupti_base(nvbench::state &exec_state);
  measure_cupti_base(const measure_cupti_base &)            = delete;
  measure_cupti_base(measure_cupti_base &&)                 = delete;
  measure_cupti_base &operator=(const measure_cupti_base &) = delete;
  measure_cupti_base &operator=(measure_cupti_base &&)      = delete;

protected:
  struct kernel_launch_timer;

  void check();
  void generate_summaries();

  __forceinline__ void flush_device_l2() { m_l2flush.flush(m_launch.get_stream()); }

  __forceinline__ void sync_stream() const
  {
    NVBENCH_CUDA_CALL(cudaStreamSynchronize(m_launch.get_stream()));
  }

  nvbench::state &m_state;

  nvbench::launch m_launch;
  nvbench::detail::l2flush m_l2flush;
  nvbench::cpu_timer m_walltime_timer;

  cupti_profiler m_cupti;

  nvbench::int64_t m_total_samples{};
};

struct measure_cupti_base::kernel_launch_timer
{
  explicit kernel_launch_timer(measure_cupti_base &measure)
      : m_measure{measure}
  {}

  __forceinline__ void start()
  {
    m_measure.flush_device_l2();
    m_measure.sync_stream();

    if (m_measure.m_cupti.is_initialized())
    {
      m_measure.m_cupti.start_user_loop();
    }
  }

  __forceinline__ void stop()
  {
    if (m_measure.m_cupti.is_initialized())
    {
      m_measure.m_cupti.stop_user_loop();
    }

    m_measure.sync_stream();
  }

private:
  measure_cupti_base &m_measure;
};

template <typename KernelLauncher>
struct measure_cupti : public measure_cupti_base
{
  measure_cupti(nvbench::state &state, KernelLauncher &kernel_launcher)
      : measure_cupti_base(state)
      , m_kernel_launcher{kernel_launcher}
  {}

  void operator()()
  {
    this->check();
    this->run();
    this->generate_summaries();
  }

private:
  // Run the kernel as many times as CUPTI requires.
  void run()
  {
    m_walltime_timer.start();
    m_total_samples = 0;

    kernel_launch_timer timer(*this);

    m_cupti.prepare_user_loop();

    do
    {
      m_kernel_launcher(m_launch, timer);
      ++m_total_samples;
    } while (m_cupti.is_replay_required());

    m_cupti.process_user_loop();

    m_walltime_timer.stop();
  }

  KernelLauncher &m_kernel_launcher;
};

} // namespace detail
} // namespace nvbench
