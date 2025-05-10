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

#include <nvbench/runner.cuh>

#include <nvbench/benchmark_base.cuh>
#include <nvbench/printer_base.cuh>
#include <nvbench/state.cuh>

#include <fmt/format.h>

#include <cstdio>
#include <stdexcept>

namespace nvbench
{

void runner_base::generate_states()
{
  m_benchmark.m_states = nvbench::detail::state_generator::create(m_benchmark);
}

void runner_base::handle_sampling_exception(const std::exception &e, state &exec_state) const
{
  // If the state is skipped, that means the execution framework class handled
  // the error already.
  if (exec_state.is_skipped())
  {
    this->print_skip_notification(exec_state);
  }
  else
  {
    const auto reason = fmt::format("Unexpected error: {}", e.what());

    if (auto printer_opt_ref = exec_state.get_benchmark().get_printer();
        printer_opt_ref.has_value())
    {
      auto &printer = printer_opt_ref.value().get();
      printer.log(nvbench::log_level::fail, reason);
    }

    exec_state.skip(reason);
  }
}

void runner_base::run_state_prologue(nvbench::state &exec_state) const
{
  // Log if a printer exists:
  if (auto printer_opt_ref = exec_state.get_benchmark().get_printer(); printer_opt_ref.has_value())
  {
    auto &printer = printer_opt_ref.value().get();
    printer.log_run_state(exec_state);
  }
}

void runner_base::run_state_epilogue(state &exec_state) const
{
  // Notify the printer that the state has completed::
  if (auto printer_opt_ref = exec_state.get_benchmark().get_printer(); printer_opt_ref.has_value())
  {
    auto &printer = printer_opt_ref.value().get();
    printer.add_completed_state();
  }
}

void runner_base::print_skip_notification(state &exec_state) const
{
  if (auto printer_opt_ref = exec_state.get_benchmark().get_printer(); printer_opt_ref.has_value())
  {
    auto &printer = printer_opt_ref.value().get();
    printer.log(nvbench::log_level::skip, exec_state.get_skip_reason());
  }
}

} // namespace nvbench
