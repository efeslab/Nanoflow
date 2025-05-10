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

#include <nvbench/state.cuh>

#include <nvbench/benchmark_base.cuh>
#include <nvbench/detail/throw.cuh>
#include <nvbench/types.cuh>

#include <fmt/color.h>
#include <fmt/format.h>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace nvbench
{

state::state(const benchmark_base &bench)
    : m_benchmark{bench}
    , m_run_once{bench.get_run_once()}
    , m_disable_blocking_kernel{bench.get_disable_blocking_kernel()}
    , m_criterion_params{bench.get_criterion_params()}
    , m_stopping_criterion(bench.get_stopping_criterion())
    , m_min_samples{bench.get_min_samples()}
    , m_skip_time{bench.get_skip_time()}
    , m_timeout{bench.get_timeout()}
{}

state::state(const benchmark_base &bench,
             nvbench::named_values values,
             std::optional<nvbench::device_info> device,
             std::size_t type_config_index)
    : m_benchmark{bench}
    , m_axis_values{std::move(values)}
    , m_device{std::move(device)}
    , m_type_config_index{type_config_index}
    , m_run_once{bench.get_run_once()}
    , m_disable_blocking_kernel{bench.get_disable_blocking_kernel()}
    , m_criterion_params{bench.get_criterion_params()}
    , m_stopping_criterion(bench.get_stopping_criterion())
    , m_min_samples{bench.get_min_samples()}
    , m_skip_time{bench.get_skip_time()}
    , m_timeout{bench.get_timeout()}
{}

nvbench::int64_t state::get_int64(const std::string &axis_name) const
{
  return m_axis_values.get_int64(axis_name);
}

nvbench::int64_t state::get_int64_or_default(const std::string &axis_name,
                                             nvbench::int64_t default_value) const
try
{
  return this->get_int64(axis_name);
}
catch (...)
{
  return default_value;
}

nvbench::float64_t state::get_float64(const std::string &axis_name) const
{
  return m_axis_values.get_float64(axis_name);
}

nvbench::float64_t state::get_float64_or_default(const std::string &axis_name,
                                                 nvbench::float64_t default_value) const
try
{
  return this->get_float64(axis_name);
}
catch (...)
{
  return default_value;
}

const std::string &state::get_string(const std::string &axis_name) const
{
  return m_axis_values.get_string(axis_name);
}

const std::string &state::get_string_or_default(const std::string &axis_name,
                                                const std::string &default_value) const
try
{
  return this->get_string(axis_name);
}
catch (...)
{
  return default_value;
}

summary &state::add_summary(std::string summary_tag)
{
  return m_summaries.emplace_back(std::move(summary_tag));
}

summary &state::add_summary(summary s)
{
  m_summaries.push_back(std::move(s));
  return m_summaries.back();
}

const summary &state::get_summary(std::string_view tag) const
{
  // Check tags first
  auto iter = std::find_if(m_summaries.cbegin(), m_summaries.cend(), [&tag](const auto &s) {
    return s.get_tag() == tag;
  });
  if (iter != m_summaries.cend())
  {
    return *iter;
  }

  // Then names:
  iter = std::find_if(m_summaries.cbegin(), m_summaries.cend(), [&tag](const auto &s) {
    return s.get_string("name") == tag;
  });
  if (iter != m_summaries.cend())
  {
    return *iter;
  }

  NVBENCH_THROW(std::invalid_argument, "No summary tagged '{}'.", tag);
}

summary &state::get_summary(std::string_view tag)
{
  // Check tags first
  auto iter = std::find_if(m_summaries.begin(), m_summaries.end(), [&tag](const auto &s) {
    return s.get_tag() == tag;
  });
  if (iter != m_summaries.end())
  {
    return *iter;
  }

  // Then names:
  iter = std::find_if(m_summaries.begin(), m_summaries.end(), [&tag](const auto &s) {
    return s.get_string("name") == tag;
  });
  if (iter != m_summaries.end())
  {
    return *iter;
  }

  NVBENCH_THROW(std::invalid_argument, "No summary tagged '{}'.", tag);
}

const std::vector<summary> &state::get_summaries() const { return m_summaries; }

std::vector<summary> &state::get_summaries() { return m_summaries; }

std::string state::get_axis_values_as_string(bool color) const
{
  // Returns fmt_style if color is true, otherwise no style flags.
  auto style = [&color](auto fmt_style) {
    const fmt::text_style no_style;
    return color ? fmt_style : no_style;
  };

  // Create a Key=Value list of all parameters:
  fmt::memory_buffer buffer;

  auto append_key_value =
    [&buffer, &style](const std::string &key, const auto &value, std::string value_fmtstr = "{}") {
      constexpr auto key_format   = fmt::emphasis::italic;
      constexpr auto value_format = fmt::emphasis::bold;

      fmt::format_to(std::back_inserter(buffer),
                     "{}{}={}",
                     buffer.size() == 0 ? "" : " ",
                     fmt::format(style(key_format), "{}", key),
                     fmt::format(style(value_format), value_fmtstr, value));
    };

  if (m_device)
  {
    append_key_value("Device", m_device->get_id());
  }

  const axes_metadata &axes = m_benchmark.get().get_axes();
  for (const auto &name : m_axis_values.get_names())
  {
    const auto axis_type = m_axis_values.get_type(name);

    // Handle power-of-two int64 axes differently:
    if (axis_type == named_values::type::int64 && axes.get_int64_axis(name).is_power_of_two())
    {
      const nvbench::int64_t value    = m_axis_values.get_int64(name);
      const nvbench::int64_t exponent = int64_axis::compute_log2(value);
      append_key_value(name, exponent, "2^{}");
    }
    else if (axis_type == named_values::type::float64)
    {
      append_key_value(name, m_axis_values.get_float64(name), "{:.5g}");
    }
    else
    {
      auto visitor = [&name, &append_key_value](const auto &value) {
        append_key_value(name, value);
      };
      std::visit(visitor, m_axis_values.get_value(name));
    }
  }

  return fmt::to_string(buffer);
}

std::string state::get_short_description(bool color) const
{
  // Returns fmt_style if color is true, otherwise no style flags.
  auto style = [&color](auto fmt_style) {
    const fmt::text_style no_style;
    return color ? fmt_style : no_style;
  };

  return fmt::format("{} [{}]",
                     fmt::format(style(fmt::emphasis::bold), "{}", m_benchmark.get().get_name()),
                     this->get_axis_values_as_string(color));
}

void state::add_element_count(std::size_t elements, std::string column_name)
{
  m_element_count += elements;
  if (!column_name.empty())
  {
    auto &summ = this->add_summary("nv/element_count/" + column_name);
    summ.set_string("description", "Number of elements: " + column_name);
    summ.set_string("name", std::move(column_name));
    summ.set_int64("value", static_cast<nvbench::int64_t>(elements));
  }
}

void state::add_global_memory_reads(std::size_t bytes, std::string column_name)
{
  m_global_memory_rw_bytes += bytes;
  if (!column_name.empty())
  {
    std::string tag = fmt::format("nv/gmem/reads/{}", column_name);
    this->add_buffer_size(bytes, std::move(tag), std::move(column_name));
  }
}

void state::add_global_memory_writes(std::size_t bytes, std::string column_name)
{
  m_global_memory_rw_bytes += bytes;
  if (!column_name.empty())
  {
    const std::string tag = fmt::format("nv/gmem/writes/{}", column_name);
    this->add_buffer_size(bytes, std::move(tag), std::move(column_name));
  }
}

void state::add_buffer_size(std::size_t num_bytes,
                            std::string summary_tag,
                            std::string column_name,
                            std::string description)
{
  auto &summ = this->add_summary(std::move(summary_tag));
  summ.set_string("hint", "bytes");
  summ.set_int64("value", static_cast<nvbench::int64_t>(num_bytes));

  if (!column_name.empty())
  {
    summ.set_string("name", std::move(column_name));
  }
  else
  {
    summ.set_string("name", ("None"));
    summ.set_string("hide", "No column name provided.");
  }
  if (!description.empty())
  {
    summ.set_string("description", std::move(description));
  }
}

} // namespace nvbench
