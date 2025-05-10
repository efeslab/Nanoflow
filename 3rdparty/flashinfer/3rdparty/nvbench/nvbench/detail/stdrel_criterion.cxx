/*
 *  Copyright 2023 NVIDIA Corporation
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

#include <nvbench/detail/stdrel_criterion.cuh>

namespace nvbench::detail
{

stdrel_criterion::stdrel_criterion()
    : stopping_criterion_base{"stdrel",
                              {{"max-noise", nvbench::detail::compat_max_noise()},
                               {"min-time", nvbench::detail::compat_min_time()}}}
{}

void stdrel_criterion::do_initialize()
{
  m_total_samples = 0;
  m_total_cuda_time = 0.0;
  m_cuda_times.clear();
  m_noise_tracker.clear();
}

void stdrel_criterion::do_add_measurement(nvbench::float64_t measurement)
{
  m_total_samples++;
  m_total_cuda_time += measurement;
  m_cuda_times.push_back(measurement);

  // Compute convergence statistics using CUDA timings:
  const auto mean_cuda_time = m_total_cuda_time / static_cast<nvbench::float64_t>(m_total_samples);
  const auto cuda_stdev     = nvbench::detail::statistics::standard_deviation(m_cuda_times.cbegin(),
                                                                          m_cuda_times.cend(),
                                                                          mean_cuda_time);
  const auto cuda_rel_stdev       = cuda_stdev / mean_cuda_time;
  if (std::isfinite(cuda_rel_stdev))
  {
    m_noise_tracker.push_back(cuda_rel_stdev);
  }
}

bool stdrel_criterion::do_is_finished()
{
  if (m_total_cuda_time <= m_params.get_float64("min-time"))
  {
    return false;
  }

  // Noise has dropped below threshold
  if (m_noise_tracker.back() < m_params.get_float64("max-noise"))
  {
    return true;
  }

  // Check if the noise (cuda rel stdev) has converged by inspecting a
  // trailing window of recorded noise measurements.
  // This helps identify benchmarks that are inherently noisy and would
  // never converge to the target stdev threshold. This check ensures that the
  // benchmark will end if the stdev stabilizes above the target threshold.
  // Gather some iterations before checking noise, and limit how often we
  // check this.
  if (m_noise_tracker.size() > 64 && (m_total_samples % 16 == 0))
  {
    // Use the current noise as the stdev reference.
    const auto current_noise = m_noise_tracker.back();
    const auto noise_stdev =
      nvbench::detail::statistics::standard_deviation(m_noise_tracker.cbegin(),
                                                      m_noise_tracker.cend(),
                                                      current_noise);
    const auto noise_rel_stdev = noise_stdev / current_noise;

    // If the rel stdev of the last N cuda noise measurements is less than
    // 5%, consider the result stable.
    const auto noise_threshold = 0.05;
    if (noise_rel_stdev < noise_threshold)
    {
      return true;
    }
  }

  return false;
}

} // namespace nvbench::detail
