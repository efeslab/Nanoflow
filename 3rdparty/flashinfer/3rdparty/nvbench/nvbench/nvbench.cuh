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
#include <nvbench/benchmark_base.cuh>
#include <nvbench/benchmark_manager.cuh>
#include <nvbench/callable.cuh>
#include <nvbench/config.cuh>
#include <nvbench/cpu_timer.cuh>
#include <nvbench/criterion_manager.cuh>
#include <nvbench/create.cuh>
#include <nvbench/cuda_call.cuh>
#include <nvbench/cuda_stream.cuh>
#include <nvbench/cuda_timer.cuh>
#include <nvbench/enum_type_list.cuh>
#include <nvbench/exec_tag.cuh>
#include <nvbench/launch.cuh>
#include <nvbench/main.cuh>
#include <nvbench/range.cuh>
#include <nvbench/state.cuh>
#include <nvbench/type_list.cuh>
#include <nvbench/types.cuh>
