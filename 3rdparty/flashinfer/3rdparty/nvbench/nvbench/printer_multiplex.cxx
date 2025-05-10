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

#include <nvbench/printer_multiplex.cuh>

#include <iostream>

namespace nvbench
{

printer_multiplex::printer_multiplex()
    : printer_base(std::cerr) // Nothing should write to this.
{}

void printer_multiplex::do_print_device_info()
{
  for (auto &format_ptr : m_printers)
  {
    format_ptr->print_device_info();
  }
}

void printer_multiplex::do_print_log_preamble()
{
  for (auto &format_ptr : m_printers)
  {
    format_ptr->print_log_preamble();
  }
}

void printer_multiplex::do_print_log_epilogue()
{
  for (auto &format_ptr : m_printers)
  {
    format_ptr->print_log_epilogue();
  }
}

void printer_multiplex::do_log(nvbench::log_level level, const std::string &str)
{
  for (auto &format_ptr : m_printers)
  {
    format_ptr->log(level, str);
  }
}

void printer_multiplex::do_log_run_state(const nvbench::state &exec_state)
{
  for (auto &format_ptr : m_printers)
  {
    format_ptr->log_run_state(exec_state);
  }
}

void printer_multiplex::do_process_bulk_data_float64(state &state,
                                                     const std::string &tag,
                                                     const std::string &hint,
                                                     const std::vector<nvbench::float64_t> &data)
{
  for (auto &format_ptr : m_printers)
  {
    format_ptr->process_bulk_data(state, tag, hint, data);
  }
}

void printer_multiplex::do_print_benchmark_list(const benchmark_vector &benches)
{
  for (auto &format_ptr : m_printers)
  {
    format_ptr->print_benchmark_list(benches);
  }
}

void printer_multiplex::do_print_benchmark_results(const benchmark_vector &benches)
{
  for (auto &format_ptr : m_printers)
  {
    format_ptr->print_benchmark_results(benches);
  }
}
void printer_multiplex::do_set_completed_state_count(std::size_t states)
{
  printer_base::do_set_completed_state_count(states);
  for (auto &format_ptr : m_printers)
  {
    format_ptr->set_completed_state_count(states);
  }
}

void printer_multiplex::do_add_completed_state()
{
  printer_base::do_add_completed_state();
  for (auto &format_ptr : m_printers)
  {
    format_ptr->add_completed_state();
  }
}

void printer_multiplex::do_set_total_state_count(std::size_t states)
{
  printer_base::do_set_total_state_count(states);
  for (auto &format_ptr : m_printers)
  {
    format_ptr->set_total_state_count(states);
  }
}
void printer_multiplex::do_log_argv(const std::vector<std::string> &argv)
{
  printer_base::do_log_argv(argv);
  for (auto &format_ptr : m_printers)
  {
    format_ptr->log_argv(argv);
  }
}

} // namespace nvbench
