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

#include <nvbench/detail/entropy_criterion.cuh>
#include <nvbench/types.cuh>

#include <cmath>


namespace nvbench::detail
{

entropy_criterion::entropy_criterion()
    : stopping_criterion_base{"entropy", {{"max-angle", 0.048}, {"min-r2", 0.36}}}
{
  m_freq_tracker.reserve(m_entropy_tracker.capacity() * 2);
  m_probabilities.reserve(m_entropy_tracker.capacity() * 2);
}

void entropy_criterion::do_initialize()
{
  m_total_samples   = 0;
  m_total_cuda_time = 0.0;
  m_entropy_tracker.clear();
  m_freq_tracker.clear();
}

nvbench::float64_t entropy_criterion::compute_entropy() 
{
  const std::size_t n = m_freq_tracker.size();
  if (n == 0)
  {
    return 0.0;
  }

  m_probabilities.resize(n);
  for (std::size_t i = 0; i < n; i++)
  {
    m_probabilities[i] = static_cast<nvbench::float64_t>(m_freq_tracker[i].second) /
                         static_cast<nvbench::float64_t>(m_total_samples);
  }

  nvbench::float64_t entropy{};
  for (nvbench::float64_t p : m_probabilities)
  {
    entropy -= p * std::log2(p);
  }

  return entropy;
}

void entropy_criterion::do_add_measurement(nvbench::float64_t measurement)
{
  m_total_samples++;
  m_total_cuda_time += measurement;

  {
    auto key = measurement;
    constexpr bool bin_keys = false;

    if (bin_keys) 
    {
      const auto resolution_us = 0.5;
      const auto resulution_s = resolution_us / 1'000'000;
      const auto epsilon = resulution_s * 2;
      key = std::round(key / epsilon) * epsilon;
    }

    // This approach is about 3x faster than `std::{unordered_,}map`
    // Up to 100k samples, only about 20% slower than corresponding stdrel method
    auto it = std::lower_bound(m_freq_tracker.begin(),
                               m_freq_tracker.end(),
                               std::make_pair(key, nvbench::int64_t{}));

    if (it != m_freq_tracker.end() && it->first == key)
    {
      it->second += 1;
    }
    else
    {
      m_freq_tracker.insert(it, std::make_pair(key, nvbench::int64_t{1}));
    }
  }

  m_entropy_tracker.push_back(compute_entropy());
}

bool entropy_criterion::do_is_finished()
{
  if (m_entropy_tracker.size() < 2)
  {
    return false;
  }

  // Even number of samples is used to reduce the overhead and not required to compute entropy.
  // This makes `is_finished()` about 20% faster than corresponding stdrel method.
  if (m_total_samples % 2 != 0)
  {
    return false;
  }

  auto begin = m_entropy_tracker.cbegin();
  auto end   = m_entropy_tracker.cend();
  auto mean  = statistics::compute_mean(begin, end);

  const auto [slope, intercept] = statistics::compute_linear_regression(begin, end, mean);

  if (statistics::slope2deg(slope) > m_params.get_float64("max-angle")) 
  {
    return false;
  }

  const auto r2 = statistics::compute_r2(begin, end, mean, slope, intercept);
  if (r2 < m_params.get_float64("min-r2"))
  {
    return false;
  }

  return true;
}

} // namespace nvbench::detail
