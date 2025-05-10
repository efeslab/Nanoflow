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

#include <nvbench/detail/transform_reduce.cuh>
#include <nvbench/types.cuh>

#include <cmath>
#include <functional>
#include <iterator>
#include <limits>
#include <numeric>
#include <cmath>

#include <type_traits>

#ifndef M_PI
  #define M_PI 3.14159265358979323846
#endif

namespace nvbench::detail::statistics
{

/**
 * Computes and returns the unbiased sample standard deviation.
 *
 * If the input has fewer than 5 sample, infinity is returned.
 */
template <typename Iter, typename ValueType = typename std::iterator_traits<Iter>::value_type>
ValueType standard_deviation(Iter first, Iter last, ValueType mean)
{
  static_assert(std::is_floating_point_v<ValueType>);

  const auto num = std::distance(first, last);

  if (num < 5) // don't bother with low sample sizes.
  {
    return std::numeric_limits<ValueType>::infinity();
  }

  const auto variance = nvbench::detail::transform_reduce(first,
                                                          last,
                                                          ValueType{},
                                                          std::plus<>{},
                                                          [mean](auto val) {
                                                            val -= mean;
                                                            val *= val;
                                                            return val;
                                                          }) /
                        static_cast<ValueType>((num - 1)); // Bessel’s correction
  return std::sqrt(variance);
}

/**
 * Computes and returns the mean.
 *
 * If the input has fewer than 1 sample, infinity is returned.
 */
template <class It>
nvbench::float64_t compute_mean(It first, It last)
{
  const auto num = std::distance(first, last);

  if (num < 1)
  {
    return std::numeric_limits<nvbench::float64_t>::infinity();
  }

  return std::accumulate(first, last, 0.0) / static_cast<nvbench::float64_t>(num);
}

/**
 * Computes linear regression and returns the slope and intercept
 *
 * This version takes precomputed mean of [first, last).
 * If the input has fewer than 2 samples, infinity is returned for both slope and intercept.
 */
template <class It>
std::pair<nvbench::float64_t, nvbench::float64_t>
compute_linear_regression(It first, It last, nvbench::float64_t mean_y)
{
  const std::size_t n = static_cast<std::size_t>(std::distance(first, last));

  if (n < 2)
  {
    return std::make_pair(std::numeric_limits<nvbench::float64_t>::infinity(),
                          std::numeric_limits<nvbench::float64_t>::infinity());
  }

  // Assuming x starts from 0
  const nvbench::float64_t mean_x = (static_cast<nvbench::float64_t>(n) - 1.0) / 2.0;

  // Calculate the numerator and denominator for the slope
  nvbench::float64_t numerator   = 0.0;
  nvbench::float64_t denominator = 0.0;

  for (std::size_t i = 0; i < n; ++i, ++first)
  {
    const nvbench::float64_t x_diff = static_cast<nvbench::float64_t>(i) - mean_x;
    numerator += x_diff * (*first - mean_y);
    denominator += x_diff * x_diff;
  }

  // Calculate the slope and intercept
  const nvbench::float64_t slope     = numerator / denominator;
  const nvbench::float64_t intercept = mean_y - slope * mean_x;

  return std::make_pair(slope, intercept);
}

/**
 * Computes linear regression and returns the slope and intercept
 *
 * If the input has fewer than 2 samples, infinity is returned for both slope and intercept.
 */
template <class It>
std::pair<nvbench::float64_t, nvbench::float64_t> compute_linear_regression(It first, It last)
{
  return compute_linear_regression(first, last, compute_mean(first, last));
}

/**
 * Computes and returns the R^2 (coefficient of determination)
 *
 * This version takes precomputed mean of [first, last).
 */
template <class It>
nvbench::float64_t compute_r2(It first,
                              It last,
                              nvbench::float64_t mean_y,
                              nvbench::float64_t slope,
                              nvbench::float64_t intercept)
{
  const std::size_t n = static_cast<std::size_t>(std::distance(first, last));

  nvbench::float64_t ss_tot = 0.0;
  nvbench::float64_t ss_res = 0.0;

  for (std::size_t i = 0; i < n; ++i, ++first)
  {
    const nvbench::float64_t y = *first;
    const nvbench::float64_t y_pred = slope * static_cast<nvbench::float64_t>(i) + intercept;

    ss_tot += (y - mean_y) * (y - mean_y);
    ss_res += (y - y_pred) * (y - y_pred);
  }

  if (ss_tot == 0.0)
  {
    return 1.0;
  }

  return 1.0 - ss_res / ss_tot;
}

/**
 * Computes and returns the R^2 (coefficient of determination)
 */
template <class It>
nvbench::float64_t
compute_r2(It first, It last, nvbench::float64_t slope, nvbench::float64_t intercept)
{
  return compute_r2(first, last, compute_mean(first, last), slope, intercept);
}

inline nvbench::float64_t rad2deg(nvbench::float64_t rad)
{
  return rad * 180.0 / M_PI;
}

inline nvbench::float64_t slope2rad(nvbench::float64_t slope)
{
  return std::atan2(slope, 1.0);
}

inline nvbench::float64_t slope2deg(nvbench::float64_t slope)
{
  return rad2deg(slope2rad(slope));
}

} // namespace nvbench::detail::statistics
