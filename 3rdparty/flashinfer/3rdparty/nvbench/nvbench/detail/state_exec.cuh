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

#ifndef NVBENCH_STATE_EXEC_GUARD
#error "This is a private implementation header for state.cuh. " \
       "Do not include it directly."
#endif // NVBENCH_STATE_EXEC_GUARD

#include <nvbench/config.cuh>
#include <nvbench/exec_tag.cuh>
#include <nvbench/state.cuh>

#include <nvbench/detail/kernel_launcher_timer_wrapper.cuh>
#ifdef NVBENCH_HAS_CUPTI
#include <nvbench/detail/measure_cupti.cuh>
#endif // NVBENCH_HAS_CUPTI
#include <nvbench/detail/measure_cold.cuh>
#include <nvbench/detail/measure_hot.cuh>

#include <type_traits>

namespace nvbench
{

// warning C4702: unreachable code
// Several spurious instances in this function. MSVC 2019 seems to forget that
// sometimes the constexpr branch /isn't/ taken.
NVBENCH_MSVC_PUSH_DISABLE_WARNING(4702)

template <typename ExecTags, typename KernelLauncher>
void state::exec(ExecTags tags, KernelLauncher &&kernel_launcher)
{
  using KL = typename std::remove_reference<KernelLauncher>::type;
  using namespace nvbench::exec_tag::impl;
  static_assert(is_exec_tag_v<ExecTags>,
                "`ExecTags` argument must be a member (or combination of "
                "members) from nvbench::exec_tag.");

  constexpr auto measure_tags  = tags & measure_mask;
  constexpr auto modifier_tags = tags & modifier_mask;

  // "run once" is handled by the cold measurement:
  if (!(modifier_tags & run_once) && this->get_run_once())
  {
    constexpr auto run_once_tags = modifier_tags | cold | run_once;
    this->exec(run_once_tags, std::forward<KernelLauncher>(kernel_launcher));
    return;
  }

  if (!(modifier_tags & no_block) && this->get_disable_blocking_kernel())
  {
    constexpr auto no_block_tags = tags | no_block;
    this->exec(no_block_tags, std::forward<KernelLauncher>(kernel_launcher));
    return;
  }

  // If no measurements selected, pick some defaults based on the modifiers:
  if constexpr (!measure_tags)
  {
    if constexpr (modifier_tags & (timer | sync))
    { // Can't do hot timings with manual timer or sync; whole point is to not
      // sync in between executions.
      this->exec(cold | tags, std::forward<KernelLauncher>(kernel_launcher));
    }
    else
    {
      this->exec(cold | hot | tags, std::forward<KernelLauncher>(kernel_launcher));
    }
    return;
  }

  if (this->is_skipped())
  {
    return;
  }

  // Each measurement is deliberately isolated in constexpr branches to
  // avoid instantiating unused measurements.
  if constexpr (tags & cold)
  {
    constexpr bool use_blocking_kernel = !(tags & no_block);
    if constexpr (tags & timer)
    {
// Estimate bandwidth here
#ifdef NVBENCH_HAS_CUPTI
      if constexpr (!(modifier_tags & run_once))
      {
        if (this->is_cupti_required())
        {
          using measure_t = nvbench::detail::measure_cupti<KL>;
          measure_t measure{*this, kernel_launcher};
          measure();
        }
      }
#endif

      using measure_t = nvbench::detail::measure_cold<KL, use_blocking_kernel>;
      measure_t measure{*this, kernel_launcher};
      measure();
    }
    else
    { // Need to wrap the kernel launcher with a timer wrapper:
      using wrapper_t = nvbench::detail::kernel_launch_timer_wrapper<KL>;
      wrapper_t wrapper{kernel_launcher};

// Estimate bandwidth here
#ifdef NVBENCH_HAS_CUPTI
      if constexpr (!(modifier_tags & run_once))
      {
        if (this->is_cupti_required())
        {
          using measure_t = nvbench::detail::measure_cupti<wrapper_t>;
          measure_t measure{*this, wrapper};
          measure();
        }
      }
#endif

      using measure_t = nvbench::detail::measure_cold<wrapper_t, use_blocking_kernel>;
      measure_t measure(*this, wrapper);
      measure();
    }
  }

  if constexpr (tags & hot)
  {
    static_assert(!(tags & sync), "Hot measurement doesn't support the `sync` exec_tag.");
    static_assert(!(tags & timer), "Hot measurement doesn't support the `timer` exec_tag.");
    constexpr bool use_blocking_kernel = !(tags & no_block);
    using measure_t                    = nvbench::detail::measure_hot<KL, use_blocking_kernel>;
    measure_t measure{*this, kernel_launcher};
    measure();
  }
}

NVBENCH_MSVC_POP_WARNING()

} // namespace nvbench
