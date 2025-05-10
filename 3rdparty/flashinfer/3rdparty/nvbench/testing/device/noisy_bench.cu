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

#include <nvbench/nvbench.cuh>
#include <nvbench/test_kernels.cuh>

#include <fmt/format.h>

#include <algorithm>
#include <limits>
#include <random>
#include <stdexcept>

void noisy_bench(nvbench::state &state)
{
  // time, convert ms -> s
  const auto mean = static_cast<nvbench::float32_t>(state.get_float64("Mean")) /
                    1000.f;
  // rel stdev
  const auto noise_pct =
    static_cast<nvbench::float32_t>(state.get_float64("Noise"));
  const auto noise = noise_pct / 100.f;
  // abs stdev
  const auto stdev = noise * mean;

  std::minstd_rand rng{};
  std::normal_distribution<nvbench::float32_t> dist(mean, stdev);

  // cold tag will save time by disabling batch measurements
  state.exec(nvbench::exec_tag::impl::cold, [&](nvbench::launch &launch) {
    const auto seconds = dist(rng);
    nvbench::sleep_kernel<<<1, 1, 0, launch.get_stream()>>>(seconds);
  });

  const auto measured_mean = static_cast<nvbench::float32_t>(
    state.get_summary("nv/cold/time/gpu/mean").get_float64("value"));
  const auto measured_noise = [&]() {
    try
    {
      return static_cast<nvbench::float32_t>(
        state.get_summary("nv/cold/time/gpu/stdev/relative")
          .get_float64("value"));
    }
    catch (std::invalid_argument &)
    {
      return std::numeric_limits<nvbench::float32_t>::infinity();
    }
  }();
  const auto measured_stdev = measured_noise * measured_mean;

  const auto mean_error  = std::fabs(measured_mean - mean);
  const auto stdev_error = std::fabs(measured_stdev - stdev);
  const auto noise_error = std::fabs(measured_noise - noise);

  const auto mean_threshold  = std::max(0.025f * mean, 8e-6f); // 2.5% or 8us
  const auto stdev_threshold = std::max(0.05f * stdev, 5e-6f); // 5% or 5us

  const auto mean_pass  = mean_error < mean_threshold;
  const auto stdev_pass = stdev_error < stdev_threshold;

  fmt::print("| {:^5} "
             "| {:^12} | {:^12} "
             "| {:^12} | {:^12} | {:^4} |\n",
             "",
             "Expected",
             "Measured",
             "Error",
             "Threshold",
             "Flag");
  fmt::print("|{:-^7}"
             "|{:-^14}|{:-^14}"
             "|{:-^14}|{:-^14}|{:-^6}|\n",
             "",
             "",
             "",
             "",
             "",
             "");
  fmt::print("| Mean  "
             "| {:>9.6f} ms | {:>9.6f} ms "
             "| {:>9.6f} ms | {:>9.6f} ms | {:4} |\n"
             "| Stdev "
             "| {:>9.6f} ms | {:>9.6f} ms "
             "| {:>9.6f} ms | {:>9.6f} ms | {:4} |\n"
             "| Noise "
             "| {:>9.6f}%   | {:>9.6f}%   "
             "| {:>9.6f}%   | {:5}        | {:4} |\n",
             mean * 1000,
             measured_mean * 1000,
             mean_error * 1000,
             mean_threshold * 1000,
             mean_pass ? "" : "!!!!",

             stdev * 1000,
             measured_stdev * 1000,
             stdev_error * 1000,
             stdev_threshold * 1000,
             stdev_pass ? "" : "!!!!",

             noise * 100,
             measured_noise * 100,
             noise_error * 100,
             "",
             "");

  if (!mean_pass)
  {
    // This isn't actually logged, it just tells ctest to mark the test as
    // skipped as a soft-failure.
    fmt::print("Warn: Mean error exceeds threshold: ({:.3} ms > {:.3} ms)\n",
               mean_error * 1000,
               mean_threshold * 1000);
  }

  if (!stdev_pass)
  {
    // This isn't actually logged, it just tells ctest to mark the test as
    // skipped as a soft-failure.
    fmt::print("Warn: Stdev error exceeds threshold: "
               "({:.6} ms > {:.6} ms, noise: {:.3}%)\n",
               stdev_error * 1000,
               stdev_threshold * 1000,
               measured_noise * 100);
  }
}
NVBENCH_BENCH(noisy_bench)
  .add_float64_axis("Mean", {0.05, 0.1, 0.5, 1.0, 10.0}) // ms
  .add_float64_axis("Noise", {0.1, 5., 25.})             // %
  // disable this; we want to test that the benchmarking loop will still exit
  // when max_noise is never reached:
  .set_max_noise(0.0000001);
