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

#include <nvbench/detail/statistics.cuh>
#include <nvbench/types.cuh>

#include "test_asserts.cuh"

#include <algorithm>
#include <vector>

namespace statistics = nvbench::detail::statistics;

void test_mean()
{
  {
    std::vector<nvbench::float64_t> data{1.0, 2.0, 3.0, 4.0, 5.0};
    const nvbench::float64_t actual = statistics::compute_mean(std::begin(data), std::end(data));
    const nvbench::float64_t expected = 3.0;
    ASSERT(std::abs(actual - expected) < 0.001);
  }

  {
    std::vector<nvbench::float64_t> data;
    const bool finite = std::isfinite(statistics::compute_mean(std::begin(data), std::end(data)));
    ASSERT(!finite);
  }
}

void test_std()
{
  std::vector<nvbench::float64_t> data{1.0, 2.0, 3.0, 4.0, 5.0};
  const nvbench::float64_t mean = 3.0;
  const nvbench::float64_t actual = statistics::standard_deviation(std::begin(data), std::end(data), mean);
  const nvbench::float64_t expected = 1.581;
  ASSERT(std::abs(actual - expected) < 0.001);
}

void test_lin_regression()
{
  {
    std::vector<nvbench::float64_t> ys{1.0, 2.0, 3.0, 4.0, 5.0};
    auto [slope, intercept] = statistics::compute_linear_regression(std::begin(ys), std::end(ys));
    ASSERT(slope == 1.0);
    ASSERT(intercept == 1.0);
  }
  {
    std::vector<nvbench::float64_t> ys{42.0, 42.0, 42.0};
    auto [slope, intercept] = statistics::compute_linear_regression(std::begin(ys), std::end(ys));
    ASSERT(slope == 0.0);
    ASSERT(intercept == 42.0);
  }
  {
    std::vector<nvbench::float64_t> ys{8.0, 4.0, 0.0};
    auto [slope, intercept] = statistics::compute_linear_regression(std::begin(ys), std::end(ys));
    ASSERT(slope == -4.0);
    ASSERT(intercept == 8.0);
  }
}

void test_r2()
{
  {
    std::vector<nvbench::float64_t> ys{1.0, 2.0, 3.0, 4.0, 5.0};
    auto [slope, intercept] = statistics::compute_linear_regression(std::begin(ys), std::end(ys));
    const nvbench::float64_t actual = statistics::compute_r2(std::begin(ys), std::end(ys), slope, intercept);
    const nvbench::float64_t expected = 1.0;
    ASSERT(std::abs(actual - expected) < 0.001);
  }
  {
    std::vector<nvbench::float64_t> signal{1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<nvbench::float64_t> noise{-1.0, 1.0, -1.0, 1.0, -1.0};
    std::vector<nvbench::float64_t> ys(signal.size());

    std::transform(std::begin(signal),
                   std::end(signal),
                   std::begin(noise),
                   std::begin(ys),
                   std::plus<nvbench::float64_t>());

    auto [slope, intercept] = statistics::compute_linear_regression(std::begin(ys), std::end(ys));
    const nvbench::float64_t expected = 0.675;
    const nvbench::float64_t actual = statistics::compute_r2(std::begin(ys), std::end(ys), slope, intercept);
    ASSERT(std::abs(actual - expected) < 0.001);
  }
}

void test_slope_conversion()
{
  {
    const nvbench::float64_t actual = statistics::slope2deg(0.0);
    const nvbench::float64_t expected = 0.0;
    ASSERT(std::abs(actual - expected) < 0.001);
  }
  {
    const nvbench::float64_t actual = statistics::slope2deg(1.0);
    const nvbench::float64_t expected = 45.0;
    ASSERT(std::abs(actual - expected) < 0.001);
  }
  {
    const nvbench::float64_t actual = statistics::slope2deg(5.0);
    const nvbench::float64_t expected = 78.69;
    ASSERT(std::abs(actual - expected) < 0.001);
  }
}

int main()
{
  test_mean();
  test_std();
  test_lin_regression();
  test_r2();
  test_slope_conversion();
}
