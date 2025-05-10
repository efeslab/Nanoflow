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

namespace nvbench
{

struct launch;

namespace detail
{

/// Wrap a KernelLauncher call between start/stop calls.
///
/// This simplifies the implementation of measurements that support
/// `nvbench::exec_tag::timer`.
template <typename KernelLauncher>
struct kernel_launch_timer_wrapper
{
  explicit kernel_launch_timer_wrapper(KernelLauncher &launcher)
      : m_kernel_launcher{launcher}
  {}

  template <typename TimerT>
  __forceinline__ void operator()(nvbench::launch &launch, TimerT &timer)
  {
    timer.start();
    m_kernel_launcher(launch);
    timer.stop();
  }

  KernelLauncher &m_kernel_launcher;
};

} // namespace detail
} // namespace nvbench
