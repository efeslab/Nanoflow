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

#include <nvbench/detail/git_revision.cuh>

// WAR issue rapidsai/rapids-cmake#99:
#define NVBENCH_GIT_BRANCH NVBench_GIT_BRANCH
#define NVBENCH_GIT_SHA1 NVBench_GIT_SHA1
#define NVBENCH_GIT_VERSION NVBench_GIT_VERSION
#ifdef NVBench_GIT_IS_DIRTY
#define NVBENCH_GIT_IS_DIRTY
#endif
