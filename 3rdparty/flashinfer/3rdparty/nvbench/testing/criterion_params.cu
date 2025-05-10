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

#include <nvbench/criterion_manager.cuh>
#include <nvbench/types.cuh>

#include "test_asserts.cuh"

void test_compat_parameters()
{
  nvbench::criterion_params params;

  ASSERT(params.has_value("max-noise"));
  ASSERT(params.has_value("min-time"));

  ASSERT(params.get_float64("max-noise") == nvbench::detail::compat_max_noise());
  ASSERT(params.get_float64("min-time") == nvbench::detail::compat_min_time());
}

void test_compat_overwrite()
{
  nvbench::criterion_params params;
  params.set_float64("max-noise", 40000.0);
  params.set_float64("min-time", 42000.0);

  ASSERT(params.get_float64("max-noise") == 40000.0);
  ASSERT(params.get_float64("min-time") == 42000.0);
}

void test_overwrite()
{
  nvbench::criterion_params params;
  ASSERT(!params.has_value("custom"));

  params.set_float64("custom", 42.0);
  ASSERT(params.get_float64("custom") == 42.0);

  params.set_float64("custom", 4.2);
  ASSERT(params.get_float64("custom") == 4.2);
}

int main()
{
  test_compat_parameters();
  test_compat_overwrite();
  test_overwrite();
}

