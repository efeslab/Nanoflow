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

#include <nvbench/flags.cuh>

#include <type_traits>

namespace nvbench::detail
{

// See the similarly named tags in nvbench::exec_tag:: for documentation.
enum class exec_flag
{
  none = 0x0,

  // Modifiers:
  timer         = 0x01, // KernelLauncher uses manual timing
  no_block      = 0x02, // Disables use of `blocking_kernel`.
  sync          = 0x04, // KernelLauncher has indicated that it will sync
  run_once      = 0x08, // Only run the benchmark once (for profiling).
  modifier_mask = timer | no_block | sync | run_once,

  // Measurement types:
  cold         = 0x0100, // measure_cold
  hot          = 0x0200, // measure_hot
  measure_mask = cold | hot
};

} // namespace nvbench::detail

NVBENCH_DECLARE_FLAGS(nvbench::detail::exec_flag)

namespace nvbench::exec_tag
{

namespace impl
{

struct tag_base
{};

template <typename ExecTag>
constexpr inline bool is_exec_tag_v = std::is_base_of_v<tag_base, ExecTag>;

/// Base class for exec_tag functionality.
/// This exists so that the `exec_flag`s can be embedded in a type with flag
/// semantics. This allows state::exec to only instantiate the measurements
/// that are actually used.
template <nvbench::detail::exec_flag Flags>
struct tag
    : std::integral_constant<nvbench::detail::exec_flag, Flags>
    , tag_base
{
  static constexpr nvbench::detail::exec_flag flags = Flags;

  template <nvbench::detail::exec_flag OFlags>
  constexpr auto operator|(tag<OFlags>) const
  {
    return tag<Flags | OFlags>{};
  }

  template <nvbench::detail::exec_flag OFlags>
  constexpr auto operator&(tag<OFlags>) const
  {
    return tag<Flags & OFlags>{};
  }

  constexpr auto operator~() const { return tag<~Flags>{}; }

  constexpr operator bool() const // NOLINT(google-explicit-constructor)
  {
    return Flags != nvbench::detail::exec_flag::none;
  }
};

using none_t          = tag<nvbench::detail::exec_flag::none>;
using timer_t         = tag<nvbench::detail::exec_flag::timer>;
using no_block_t      = tag<nvbench::detail::exec_flag::no_block>;
using sync_t          = tag<nvbench::detail::exec_flag::sync>;
using run_once_t      = tag<nvbench::detail::exec_flag::run_once>;
using hot_t           = tag<nvbench::detail::exec_flag::hot>;
using cold_t          = tag<nvbench::detail::exec_flag::cold>;
using modifier_mask_t = tag<nvbench::detail::exec_flag::modifier_mask>;
using measure_mask_t  = tag<nvbench::detail::exec_flag::measure_mask>;

constexpr inline none_t none;
constexpr inline timer_t timer;
constexpr inline no_block_t no_block;
constexpr inline sync_t sync;
constexpr inline run_once_t run_once;
constexpr inline cold_t cold;
constexpr inline hot_t hot;
constexpr inline modifier_mask_t modifier_mask;
constexpr inline measure_mask_t measure_mask;

} // namespace impl

constexpr inline auto none = nvbench::exec_tag::impl::none;

/// Modifier used when only a portion of the KernelLauncher needs to be timed.
/// Useful for resetting state in-between timed kernel launches.
constexpr inline auto timer = nvbench::exec_tag::impl::timer;

/// Modifier used to indicate that the KernelGenerator will perform CUDA
/// synchronizations. Without this flag such benchmarks will deadlock.
constexpr inline auto sync = nvbench::exec_tag::impl::no_block | nvbench::exec_tag::impl::sync;

/// Modifier used to indicate that batched measurements should be disabled
constexpr inline auto no_batch = nvbench::exec_tag::impl::cold;

} // namespace nvbench::exec_tag
