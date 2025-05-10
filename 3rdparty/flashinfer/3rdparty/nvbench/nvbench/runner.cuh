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

#include <nvbench/detail/state_generator.cuh>

#include <stdexcept>
#include <vector>

namespace nvbench
{

// Non-templated code goes here to reduce instantiation costs:
struct runner_base
{
  explicit runner_base(nvbench::benchmark_base &bench)
      : m_benchmark{bench}
  {}

  void generate_states();

  void handle_sampling_exception(const std::exception &e, nvbench::state &exec_state) const;

  void run_state_prologue(state &exec_state) const;
  void run_state_epilogue(state &exec_state) const;

  void print_skip_notification(nvbench::state &exec_state) const;

  nvbench::benchmark_base &m_benchmark;
};

template <typename BenchmarkType>
struct runner : public runner_base
{
  using benchmark_type                          = BenchmarkType;
  using kernel_generator                        = typename benchmark_type::kernel_generator;
  using type_configs                            = typename benchmark_type::type_configs;
  static constexpr std::size_t num_type_configs = benchmark_type::num_type_configs;

  explicit runner(benchmark_type &bench)
      : runner_base{bench}
  {}

  void run()
  {
    if (m_benchmark.m_devices.empty())
    {
      this->run_device(std::nullopt);
    }
    else
    {
      for (const auto &device : m_benchmark.m_devices)
      {
        this->run_device(device);
      }
    }
  }

private:
  void run_device(const std::optional<nvbench::device_info> &device)
  {
    if (device)
    {
      device->set_active();
    }

    // Iterate through type_configs:
    std::size_t type_config_index = 0;
    nvbench::tl::foreach<type_configs>(
      [&self = *this, &states = m_benchmark.m_states, &type_config_index, &device](
        auto type_config_wrapper) {
        // Get current type_config:
        using type_config = typename decltype(type_config_wrapper)::type;

        // Find states with the current device / type_config
        for (nvbench::state &cur_state : states)
        {
          if (cur_state.get_device() == device &&
              cur_state.get_type_config_index() == type_config_index)
          {
            self.run_state_prologue(cur_state);
            try
            {
              kernel_generator{}(cur_state, type_config{});
              if (cur_state.is_skipped())
              {
                self.print_skip_notification(cur_state);
              }
            }
            catch (std::exception &e)
            {
              self.handle_sampling_exception(e, cur_state);
            }
            self.run_state_epilogue(cur_state);
          }
        }

        ++type_config_index;
      });
  }
};

} // namespace nvbench
