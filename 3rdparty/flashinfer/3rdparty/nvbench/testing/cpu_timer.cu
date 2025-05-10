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

#include <nvbench/cpu_timer.cuh>

#include "test_asserts.cuh"

#include <chrono>
#include <thread>

void test_basic()
{
  using namespace std::literals::chrono_literals;

  nvbench::cpu_timer timer;

  timer.start();
  std::this_thread::sleep_for(250ms);
  timer.stop();

  ASSERT(timer.get_duration() > 0.25);
  ASSERT(timer.get_duration() < 0.50);
}

int main() { test_basic(); }
