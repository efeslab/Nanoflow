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

#include <nvbench/types.cuh>

#include <iosfwd>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace nvbench
{

struct benchmark_base;
struct state;

enum class log_level
{
  run,
  pass,
  fail,
  skip,
  warn,
  info
};

/*!
 * Base class for all output formats.
 *
 * Simple file writers probably just want to override `print_benchmark_results`
 * and do everything in there.
 *
 * Interactive terminal output formats should implement the logging functions so
 * users have some feedback.
 */
struct printer_base
{
  using benchmark_vector = std::vector<std::unique_ptr<benchmark_base>>;

  /*!
   * Construct a new printer_base that will write to ostream.
   */
  explicit printer_base(std::ostream &ostream)
      : printer_base(ostream, {})
  {}

  /*!
   * Construct a new print_base that will write to an ostream, described by
   * stream_name.
   *
   * `stream_name` is used to open any additional files needed by the printer.
   * If `ostream` is a file stream, use the filename. Stream name may be
   * "stdout" / "stderr" or empty.
   * @param ostream
   * @param stream_name
   */
  explicit printer_base(std::ostream &ostream, std::string stream_name);

  virtual ~printer_base();

  // move-only
  printer_base(const printer_base &)            = delete;
  printer_base(printer_base &&)                 = default;
  printer_base &operator=(const printer_base &) = delete;
  printer_base &operator=(printer_base &&)      = delete;

  /*!
   * Called once with the command line arguments used to invoke the current
   * executable.
   */
  void log_argv(const std::vector<std::string> &argv) { this->do_log_argv(argv); }

  /*!
   * Print a summary of all detected devices, if supported.
   *
   * Called before running benchmarks for active terminal output.
   */
  void print_device_info() { this->do_print_device_info(); }

  /*!
   * Called before/after starting benchmarks and submitting logs.
   * @{
   */
  void print_log_preamble() { this->do_print_log_preamble(); }
  void print_log_epilogue() { this->do_print_log_epilogue(); }
  /*!@}*/

  /*!
   * Print a log message at the specified log level.
   */
  void log(nvbench::log_level level, const std::string &msg) { this->do_log(level, msg); }

  /*!
   * Called before running the measurements associated with state.
   * Implementations are expected to call `log(log_level::run, ...)`.
   */
  void log_run_state(const nvbench::state &exec_state) { this->do_log_run_state(exec_state); }

  /*!
   * Measurements may call this to allow a printer to perform extra processing
   * on large sets of data.
   *
   * @param state The `nvbench::state` associated with this measurement.
   *
   * @param tag A tag identifying the data. Tags must be unique within a state,
   *            but the same tag may be reused in multiple states. Data produced
   *            by NVBench will be prefixed with "nv/", for example, isolated
   *            sample time measurements are tagged "nv/cold/sample_times".
   *
   * @param hint A hint describing the type of data. Subclasses may use these
   *             to determine how to handle the data, and should ignore any
   *             hints they don't understand. Common hints are:
   *             - "sample_times": `data` contains all sample times for a
   *               measurement (in seconds).
   */
  void process_bulk_data(nvbench::state &state,
                         const std::string &tag,
                         const std::string &hint,
                         const std::vector<nvbench::float64_t> &data)
  {
    this->do_process_bulk_data_float64(state, tag, hint, data);
  }

  /*!
   * Print details of the unexecuted benchmarks in `benches`. This is used for
   * `--list`.
   */
  void print_benchmark_list(const benchmark_vector &benches)
  {
    this->do_print_benchmark_list(benches);
  }

  /*!
   * Print the results of the benchmarks in `benches`.
   */
  void print_benchmark_results(const benchmark_vector &benches)
  {
    this->do_print_benchmark_results(benches);
  }

  /*!
   * Used to track progress for interactive progress display:
   *
   * - `completed_state_count`: Number of states with completed measurements.
   * - `total_state_count`: Total number of states.
   * @{
   */
  virtual void set_completed_state_count(std::size_t states)
  {
    this->do_set_completed_state_count(states);
  }
  virtual void add_completed_state() { this->do_add_completed_state(); }
  [[nodiscard]] virtual std::size_t get_completed_state_count() const
  {
    return this->do_get_completed_state_count();
  }

  virtual void set_total_state_count(std::size_t states) { this->do_set_total_state_count(states); }
  [[nodiscard]] virtual std::size_t get_total_state_count() const
  {
    return this->do_get_total_state_count();
  }
  /*!@}*/

protected:
  // Implementation hooks for subclasses:
  virtual void do_log_argv(const std::vector<std::string> &) {}
  virtual void do_print_device_info() {}
  virtual void do_print_log_preamble() {}
  virtual void do_print_log_epilogue() {}
  virtual void do_log(nvbench::log_level, const std::string &) {}
  virtual void do_log_run_state(const nvbench::state &) {}
  virtual void do_process_bulk_data_float64(nvbench::state &,
                                            const std::string &,
                                            const std::string &,
                                            const std::vector<nvbench::float64_t> &){};

  virtual void do_print_benchmark_list(const benchmark_vector &) 
  {
    throw std::runtime_error{"nvbench::do_print_benchmark_list is not supported by this printer."};
  }

  virtual void do_print_benchmark_results(const benchmark_vector &) {}

  virtual void do_set_completed_state_count(std::size_t states);
  virtual void do_add_completed_state();
  [[nodiscard]] virtual std::size_t do_get_completed_state_count() const;

  virtual void do_set_total_state_count(std::size_t states);
  [[nodiscard]] virtual std::size_t do_get_total_state_count() const;

  std::ostream &m_ostream;

  // May be empty, a filename,  or "stdout" / "stderr" depending on the type of
  // stream in m_stream.
  std::string m_stream_name;

  std::size_t m_completed_state_count{};
  std::size_t m_total_state_count{};
};

} // namespace nvbench
