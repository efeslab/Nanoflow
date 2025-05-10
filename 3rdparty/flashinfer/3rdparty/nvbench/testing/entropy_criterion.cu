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
#include <nvbench/stopping_criterion.cuh>
#include <nvbench/types.cuh>

#include "test_asserts.cuh"

#include <vector>
#include <random>
#include <numeric>

void test_const()
{
  nvbench::criterion_params params;
  nvbench::detail::entropy_criterion criterion;

  criterion.initialize(params);
  for (int i = 0; i < 6; i++) 
  { // nvbench wants at least 5 to compute the standard deviation
    criterion.add_measurement(42.0);
  }
  ASSERT(criterion.is_finished());
}

void produce_entropy_arch(nvbench::detail::entropy_criterion &criterion)
{
  /*
   * This pattern is designed to simulate the entropy:
   *
   *   0.0, 1.0, 1.5, 2.0, 2.3, 2.5 <---- no unexpected measurement after this point
   *   2.5, 2.4, 2.2, 2.1, 2.0, 1.9 <-+
   *   1.8, 1.7, 1.6, 1.6, 1.5, 1.4   |
   *   1.4, 1.3, 1.3, 1.3, 1.2, 1.2   |
   *   1.1, 1.1, 1.1, 1.0, 1.0, 1.0   +-- entropy only decreases after 5-th sample, 
   *   1.0, 0.9, 0.9, 0.9, 0.9, 0.9   |   so the slope should be negative
   *   0.8, 0.8, 0.8, 0.8, 0.8, 0.8   |
   *   0.7, 0.7, 0.7, 0.7, 0.7, 0.7 <-+
   */
  for (nvbench::float64_t x = 0.0; x < 50.0; x += 1.0)
  {
    criterion.add_measurement(x > 5.0 ? 5.0 : x);
  }
}

void test_entropy_arch()
{
  nvbench::detail::entropy_criterion criterion;

  // The R2 should be around 0.5
  // The angle should be around -1.83
  nvbench::criterion_params params;
  params.set_float64("min-r2", 0.3);
  params.set_float64("max-angle", -1.0);
  criterion.initialize(params);
  produce_entropy_arch(criterion);
  ASSERT(criterion.is_finished());

  params.set_float64("min-r2", 0.7);
  criterion.initialize(params);
  produce_entropy_arch(criterion);
  ASSERT(!criterion.is_finished());

  params.set_float64("min-r2", 0.3);
  params.set_float64("max-angle", -2.0);
  criterion.initialize(params);
  produce_entropy_arch(criterion);
  ASSERT(!criterion.is_finished());
}

int main()
{
  test_const();
  test_entropy_arch();
}
