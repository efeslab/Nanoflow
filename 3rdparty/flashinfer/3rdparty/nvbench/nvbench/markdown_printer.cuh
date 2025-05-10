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

#include <nvbench/printer_base.cuh>

#include <string>

namespace nvbench
{

struct state;
struct summary;

/*!
 * Markdown output format.
 *
 * Includes customization points to modify numeric formatting.
 */
struct markdown_printer : nvbench::printer_base
{
  using printer_base::printer_base;

  /*!
   * Enable / disable color in the output.
   *
   * Turn off for file outputs. May not work on some interactive terminals.
   * Off by default. Enable for stdout markdown printers by passing `--color` on
   * the command line.
   *
   * @{
   */
  void set_color(bool enabled) { m_color = enabled; }
  [[nodiscard]] bool get_color() const { return m_color; }
  /*!@}*/

protected:
  // Virtual API from printer_base:
  void do_print_device_info() override;
  void do_print_log_preamble() override;
  void do_print_log_epilogue() override;
  void do_log(nvbench::log_level level, const std::string &msg) override;
  void do_log_run_state(const nvbench::state &exec_state) override;
  void do_print_benchmark_list(const benchmark_vector &benches) override;
  void do_print_benchmark_results(const benchmark_vector &benches) override;

  // Customization points for formatting:
  virtual std::string do_format_default(const nvbench::summary &data);
  virtual std::string do_format_duration(const nvbench::summary &seconds);
  virtual std::string do_format_item_rate(const nvbench::summary &items_per_sec);
  virtual std::string do_format_bytes(const nvbench::summary &bytes);
  virtual std::string do_format_byte_rate(const nvbench::summary &bytes_per_sec);
  virtual std::string do_format_sample_size(const nvbench::summary &count);
  virtual std::string do_format_percentage(const nvbench::summary &percentage);

  bool m_color{false};
};

} // namespace nvbench
