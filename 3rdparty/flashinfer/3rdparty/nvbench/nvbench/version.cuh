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

#include <nvbench/detail/version.cuh>

// WAR issue rapidsai/rapids-cmake#99
#define NVBENCH_VERSION_MAJOR NVBench_VERSION_MAJOR
#define NVBENCH_VERSION_MINOR NVBench_VERSION_MINOR
#define NVBENCH_VERSION_PATCH NVBench_VERSION_PATCH

// clang-format off

/// Numeric version as "MMmmpp"
#define NVBENCH_VERSION \
  NVBENCH_VERSION_MAJOR * 10000 + \
  NVBENCH_VERSION_MINOR * 100+ \
  NVBENCH_VERSION_PATCH

// clang-format on
