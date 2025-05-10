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

#include <nvbench/benchmark.cuh>
#include <nvbench/benchmark_manager.cuh>
#include <nvbench/callable.cuh>
#include <nvbench/type_list.cuh>

#include <memory>

#define NVBENCH_TYPE_AXES(...) nvbench::type_list<__VA_ARGS__>

#define NVBENCH_BENCH(KernelGenerator)                                                             \
  NVBENCH_DEFINE_UNIQUE_CALLABLE(KernelGenerator);                                                 \
  nvbench::benchmark_base &NVBENCH_UNIQUE_IDENTIFIER(obj_##KernelGenerator) =                      \
    nvbench::benchmark_manager::get()                                                              \
      .add(std::make_unique<nvbench::benchmark<NVBENCH_UNIQUE_IDENTIFIER(KernelGenerator)>>())     \
      .set_name(#KernelGenerator)

#define NVBENCH_BENCH_TYPES(KernelGenerator, TypeAxes)                                             \
  NVBENCH_DEFINE_UNIQUE_CALLABLE_TEMPLATE(KernelGenerator);                                        \
  nvbench::benchmark_base &NVBENCH_UNIQUE_IDENTIFIER(obj_##KernelGenerator) =                      \
    nvbench::benchmark_manager::get()                                                              \
      .add(std::make_unique<                                                                       \
           nvbench::benchmark<NVBENCH_UNIQUE_IDENTIFIER(KernelGenerator), TypeAxes>>())            \
      .set_name(#KernelGenerator)
