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
#include <nvbench/stopping_criterion.cuh>
#include <nvbench/types.cuh>

#include "test_asserts.cuh"

#include <vector>
#include <random>
#include <numeric>

void test_const()
{
  nvbench::criterion_params params;
  nvbench::detail::stdrel_criterion criterion;

  criterion.initialize(params);
  for (int i = 0; i < 5; i++)
  { // nvbench wants at least 5 to compute the standard deviation
    criterion.add_measurement(42.0);
  }
  ASSERT(criterion.is_finished());
}

std::vector<double> generate(double mean, double rel_std_dev, int size)
{
  static std::mt19937::result_type seed = 0;
  std::mt19937 gen(seed++);
  std::vector<nvbench::float64_t> v(static_cast<std::size_t>(size));
  std::normal_distribution<nvbench::float64_t> dist(mean, mean * rel_std_dev);
  std::generate(v.begin(), v.end(), [&]{ return dist(gen); });
  return v;
}

void test_stdrel()
{
  const nvbench::int64_t size = 10;
  const nvbench::float64_t mean = 42.0;
  const nvbench::float64_t max_noise = 0.1;

  nvbench::criterion_params params;
  params.set_float64("max-noise", max_noise);

  nvbench::detail::stdrel_criterion criterion;
  criterion.initialize(params);

  for (nvbench::float64_t measurement: generate(mean, max_noise / 2, size))
  {
    criterion.add_measurement(measurement);
  }
  ASSERT(criterion.is_finished());

  params.set_float64("max-noise", max_noise);
  criterion.initialize(params);

  for (nvbench::float64_t measurement: generate(mean, max_noise * 2, size))
  {
    criterion.add_measurement(measurement);
  }
  ASSERT(!criterion.is_finished());
}

int main()
{
  test_const();
  test_stdrel();
}
