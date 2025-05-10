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

#include <nvbench/config.cuh>
#include <nvbench/device_info.cuh>

#include <optional>
#include <string>
#include <vector>

namespace nvbench::detail
{

#ifdef NVBENCH_HAS_CUPTI
/**
 * Pass required metrics in the constructor and organize your code as follows
 * to get counters back:
 *
 * ```
 * cupti_profiler cupti(
 *      nvbench::device_info{0},
 *      { "l1tex__t_sector_hit_rate.pct" });
 *
 * cupti->prepare_user_loop();
 *
 * do
 * {
 *   m_measure.m_cupti->start_user_loop();
 *
 *   kernel_1<<<1, 1>>>();
 *   // ...
 *   kernel_n<<<1, 1>>>();
 *
 *   m_measure.m_cupti->stop_user_loop();
 * } while(cupti->is_replay_required());
 *
 * cupti->process_user_loop();
 *
 * auto result = m_cupti->get_counter_values();
 * ```
 *
 * Check Perfworks Metric table here for the full list of metrics:
 * https://docs.nvidia.com/cupti/r_main.html#metrics-reference-7x
 */
class cupti_profiler
{
  bool m_available{};
  std::string m_chip_name;

  // Counter data
  std::vector<std::string> m_metric_names;
  std::vector<std::uint8_t> m_data_image_prefix;
  std::vector<std::uint8_t> m_config_image;
  std::vector<std::uint8_t> m_data_image;
  std::vector<std::uint8_t> m_data_scratch_buffer;
  std::vector<std::uint8_t> m_availability_image;
  nvbench::device_info m_device;

  // CUPTI runs a series of replay passes, where each pass contains a sequence
  // of ranges. Every metric enabled in the configuration is collected
  // separately per unique range in the pass. CUPTI supports auto and
  // user-defined ranges. With auto range mode, ranges are defined around each
  // kernel automatically. In the user range mode, ranges are defined manually.
  // We define a single user range for the whole measurement.
  static const int m_num_ranges = 1;

public:
  // Move only
  cupti_profiler(cupti_profiler &&) noexcept;
  cupti_profiler &operator=(cupti_profiler &&) noexcept;

  cupti_profiler(const cupti_profiler &)            = delete;
  cupti_profiler &operator=(const cupti_profiler &) = delete;

  cupti_profiler(nvbench::device_info device, std::vector<std::string> &&metric_names);
  ~cupti_profiler();

  [[nodiscard]] bool is_initialized() const;

  /// Should be called before replay loop
  void prepare_user_loop();

  /// Should be called before any kernel calls in the replay loop
  void start_user_loop();

  /// Should be called after all kernel calls in the replay loop
  void stop_user_loop();

  /// Should be called after the replay loop
  void process_user_loop();

  /// Indicates whether another iteration of the replay loop is required
  [[nodiscard]] bool is_replay_required();

  /// Returns counters for metrics requested in the constructor
  [[nodiscard]] std::vector<double> get_counter_values();

private:
  void initialize_profiler();
  void initialize_chip_name();
  void initialize_availability_image();
  static void initialize_nvpw();
  void initialize_config_image();
  void initialize_counter_data_prefix_image();
  void initialize_counter_data_image();
};
#endif

} // namespace nvbench::detail
