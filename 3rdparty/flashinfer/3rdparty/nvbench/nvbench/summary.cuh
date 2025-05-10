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

#include <nvbench/named_values.cuh>

#include <string>
#include <utility>

namespace nvbench
{

/**
 * @brief A single value associated with a benchmark state.
 *
 * Each summary object contains a single value with associated metadata, such
 * as name, description, type, and formatting hints. Each summary object
 * corresponds to a cell in an output markdown table, with summaries grouped
 * into columns by their tag.
 *
 * The summary tag provided at construction should be a unique identifier that
 * will be convenient and unambiguous during lookups. For example, summaries
 * produced by NVBench will begin with `nv/` and contain a hierarchical
 * organization of descriptors, such as `nv/cold/time/gpu/mean`.
 *
 * The summary may contain an arbitrary number of key/value pairs. The keys
 * are `std::string` and the values may be `std::string`, `int64_t`, or
 * `float64_t`. These may be used to store arbitrary user data and will be
 * written into the json output.
 *
 * Some keys are reserved and have special meaning. These may be used by tooling
 * to help interpret data:
 *
 * - `"name": required [string]` Compact, used for table headings.
 * - `"description": optional [string]` Longer description.
 * - `"value": required [string|float64|int64]` Actual value.
 * - `"hint": optional [string]` Formatting hints (see below)
 * - `"hide": optional [string]` If present, the summary will not be included in
 *                               markdown output tables.
 *
 * Additionally, keys beginning with `nv/` are reserved for NVBench.
 *
 * Hints indicate the type of data stored in "value", but may be omitted.
 * NVBench uses the following hints:
 *
 * - "duration": "value" is a float64_t time duration in seconds.
 * - "item_rate": "value" is a float64_t item rate in elements / second.
 * - "bytes": "value" is an int64_t number of bytes.
 * - "byte_rate": "value" is a float64_t byte rate in bytes / second.
 * - "sample_size": "value" is an int64_t samples count.
 * - "percentage": "value" is a float64_t percentage (100% stored as 1.0).
 * - "file/sample_times":
 *   - "filename" is the path to a binary file that encodes all sample
 *     times (in seconds) as float32_t values.
 *   - "size" is an int64_t containing the number of float32_t values stored in
 *     the binary file.
 *
 *
 * Example: Adding a new summary to an nvbench::state object:
 *
 * ```
 * auto &summ = state.add_summary("nv/batch/gpu/time/mean");
 * summ.set_string("name", "Batch GPU");
 * summ.set_string("hint", "duration");
 * summ.set_string("description",
 *                 "Average batch execution time measured by CUDA event
 *                  timers.");
 * summ.set_float64("value", avg_batch_gpu_time);
 * ```
 */
struct summary : public nvbench::named_values
{
  summary() = default;
  explicit summary(std::string tag)
      : m_tag(std::move(tag))
  {}

  // move-only
  summary(const summary &)            = delete;
  summary(summary &&)                 = default;
  summary &operator=(const summary &) = delete;
  summary &operator=(summary &&)      = default;

  void set_tag(std::string tag) { m_tag = std::move(tag); }
  [[nodiscard]] const std::string &get_tag() const { return m_tag; }

private:
  std::string m_tag;
};

} // namespace nvbench
