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

#include <nvbench/cuda_call.cuh>

#include <cuda_runtime_api.h>

namespace nvbench::detail
{

/**
 * RAII wrapper that sets the current CUDA device on construction and restores
 * the previous device on destruction.
 */
struct [[maybe_unused]] device_scope
{
  explicit device_scope(int dev_id)
      : m_old_device_id(get_current_device())
  {
    NVBENCH_CUDA_CALL(cudaSetDevice(dev_id));
  }
  ~device_scope() { NVBENCH_CUDA_CALL(cudaSetDevice(m_old_device_id)); }

  // move-only
  device_scope(device_scope &&)                 = default;
  device_scope &operator=(device_scope &&)      = default;
  device_scope(const device_scope &)            = delete;
  device_scope &operator=(const device_scope &) = delete;

private:
  static int get_current_device()
  {
    int dev_id{};
    NVBENCH_CUDA_CALL(cudaGetDevice(&dev_id));
    return dev_id;
  }

  int m_old_device_id;
};

} // namespace nvbench::detail
