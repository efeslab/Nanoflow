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

#include <nvbench/device_info.cuh>
#include <nvbench/printer_multiplex.cuh>
#include <nvbench/stopping_criterion.cuh>

#include <iosfwd>
#include <memory>
#include <string>
#include <vector>

namespace nvbench
{

struct benchmark_base;
struct float64_axis;
struct int64_axis;
struct printer_base;
struct string_axis;
struct type_axis;

/**
 * Parses command-line args into a set of benchmarks.
 */
struct option_parser
{
  using benchmark_vector = std::vector<std::unique_ptr<nvbench::benchmark_base>>;

  option_parser();
  ~option_parser();

  void parse(int argc, char const *const argv[]);
  void parse(std::vector<std::string> args);

  [[nodiscard]] benchmark_vector &get_benchmarks() { return m_benchmarks; };
  [[nodiscard]] const benchmark_vector &get_benchmarks() const { return m_benchmarks; };

  [[nodiscard]] const std::vector<std::string> &get_args() const { return m_args; }

  /*!
   * Returns the output format requested by the parse options.
   *
   * If no output format requested, markdown + stdout are used.
   *
   * If multiple formats requested, an output_multiple is used.
   *
   * The returned object is only valid for the lifetime of this option_parser.
   */
  // printer_base has no useful const API, so no const overload.
  [[nodiscard]] nvbench::printer_base &get_printer();

private:
  void parse_impl();

  using arg_iterator_t = std::vector<std::string>::const_iterator;
  void parse_range(arg_iterator_t first, arg_iterator_t last);

  void add_markdown_printer(const std::string &spec);
  void add_csv_printer(const std::string &spec);
  void add_json_printer(const std::string &spec, bool enable_binary);

  std::ostream &printer_spec_to_ostream(const std::string &spec);

  void print_version() const;
  void print_list(printer_base& printer) const;
  void print_help() const;
  void print_help_axis() const;

  void set_persistence_mode(const std::string &state);
  void lock_gpu_clocks(const std::string &rate);

  void set_stopping_criterion(const std::string &criterion);
  void enable_run_once();
  void disable_blocking_kernel();

  void add_benchmark(const std::string &name);
  void replay_global_args();

  void update_devices(const std::string &devices);

  void update_axis(const std::string &spec);
  static void update_int64_axis(int64_axis &axis,
                                std::string_view value_spec,
                                std::string_view flag_spec);
  static void update_float64_axis(float64_axis &axis,
                                  std::string_view value_spec,
                                  std::string_view flag_spec);
  static void update_string_axis(string_axis &axis,
                                 std::string_view value_spec,
                                 std::string_view flag_spec);
  static void update_type_axis(type_axis &axis,
                               std::string_view value_spec,
                               std::string_view flag_spec);

  void update_int64_prop(const std::string &prop_arg, const std::string &prop_val);
  void update_float64_prop(const std::string &prop_arg, const std::string &prop_val);

  void update_criterion_prop(const std::string &prop_arg,
                             const std::string &prop_val,
                             const nvbench::named_values::type type);

  void update_used_device_state() const;

  // Command line args
  std::vector<std::string> m_args;

  // Store benchmark modifiers passed in before any benchmarks are requested as
  // "global args". Replay them after every benchmark.
  std::vector<std::string> m_global_benchmark_args;

  // List of devices specified by the most recent --devices option, or all
  // devices if --devices has not been used.
  std::vector<nvbench::device_info> m_recent_devices;

  benchmark_vector m_benchmarks;

  // Manages lifetimes of any ofstreams opened for m_printer.
  std::vector<std::unique_ptr<std::ofstream>> m_ofstream_storage;

  // The main printer to use:
  nvbench::printer_multiplex m_printer;

  // Use color on any stdout markdown printers.
  bool m_color_md_stdout_printer{false};

  // True if any stdout printers have been added to m_printer.
  bool m_have_stdout_printer{false};

  // Used for device modification commands like --log-gpu-clocks
  bool m_exit_after_parsing{false};
};

} // namespace nvbench
