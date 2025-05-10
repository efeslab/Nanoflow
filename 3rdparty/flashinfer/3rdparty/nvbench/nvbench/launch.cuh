/*
 *  Copyright 2021-2022 NVIDIA Corporation
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

#include <nvbench/cuda_stream.cuh>

namespace nvbench
{

/**
 * Configuration object used to communicate with a `KernelLauncher`.
 *
 * The `KernelLauncher` passed into `nvbench::state::exec` is required to
 * accept an `nvbench::launch` argument:
 *
 * ```cpp
 * state.exec([](nvbench::launch &launch) {
 *   kernel<<<M, N, 0, launch.get_stream()>>>();
 * }
 * ```
 */
struct launch
{
  explicit launch(const nvbench::cuda_stream &stream)
      : m_stream{stream}
  {}

  // move-only
  launch(const launch &)            = delete;
  launch(launch &&)                 = default;
  launch &operator=(const launch &) = delete;
  launch &operator=(launch &&)      = delete;

  /**
   * @return a CUDA stream that all kernels and other stream-ordered CUDA work
   * must use. This stream can be changed by the `KernelGenerator` using the
   * `nvbench::state::set_cuda_stream` method.
   */
  __forceinline__ const nvbench::cuda_stream &get_stream() const { return m_stream; };

private:
  // The stream is owned by the `nvbench::state` associated with this launch.
  const nvbench::cuda_stream &m_stream;
};

} // namespace nvbench
