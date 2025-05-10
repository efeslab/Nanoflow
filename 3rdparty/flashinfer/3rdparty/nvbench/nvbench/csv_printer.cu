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

#include <nvbench/csv_printer.cuh>

#include <nvbench/axes_metadata.cuh>
#include <nvbench/benchmark_base.cuh>
#include <nvbench/device_info.cuh>
#include <nvbench/summary.cuh>

#include <nvbench/internal/table_builder.cuh>

#include <fmt/format.h>

#include <cstdint>
#include <ostream>
#include <string>
#include <variant>
#include <vector>

namespace nvbench
{

void csv_printer::do_print_benchmark_results(const benchmark_vector &benches)
{
  auto format_visitor = [](const auto &v) {
    using T = std::decay_t<decltype(v)>;
    if constexpr (std::is_same_v<T, std::string>)
    {
      return v;
    }

    // warning C4702: unreachable code
    // This is a future-proofing fallback that's currently unused.
    NVBENCH_MSVC_PUSH_DISABLE_WARNING(4702)
    return fmt::format("{}", v);
  };
  NVBENCH_MSVC_POP_WARNING()

  // Prepare table:
  nvbench::internal::table_builder table;
  std::size_t row = 0;
  for (const auto &bench_ptr : benches)
  {
    const auto &bench = *bench_ptr;
    const auto &axes  = bench.get_axes();

    const auto &bench_name = bench.get_name();

    for (const auto &cur_state : bench.get_states())
    {
      std::optional<nvbench::device_info> device = cur_state.get_device();

      std::string device_id   = device ? fmt::to_string(device->get_id()) : std::string{};
      std::string device_name = device ? std::string{device->get_name()} : std::string{};

      table.add_cell(row, "_bench_name", "Benchmark", bench_name);
      table.add_cell(row, "_device_id", "Device", std::move(device_id));
      table.add_cell(row, "_device_name", "Device Name", std::move(device_name));

      const auto &axis_values = cur_state.get_axis_values();
      for (const auto &name : axis_values.get_names())
      {
        // Handle power-of-two int64 axes differently:
        if (axis_values.get_type(name) == named_values::type::int64 &&
            axes.get_int64_axis(name).is_power_of_two())
        {
          const nvbench::int64_t value    = axis_values.get_int64(name);
          const nvbench::int64_t exponent = int64_axis::compute_log2(value);
          table.add_cell(row,
                         name + "_axis_pow2_pretty",
                         name + " (pow2)",
                         fmt::format("2^{}", exponent));
          table.add_cell(row, name + "_axis_plain", fmt::format("{}", name), fmt::to_string(value));
        }
        else
        {
          std::string value = std::visit(format_visitor, axis_values.get_value(name));
          table.add_cell(row, name + "_axis", name, std::move(value));
        }
      }

      if (cur_state.is_skipped())
      {
        table.add_cell(row, "_skip_reason", "Skipped", "Yes");
        row++;
        continue;
      }

      table.add_cell(row, "_skip_reason", "Skipped", "No");

      for (const auto &summ : cur_state.get_summaries())
      {
        if (summ.has_value("hide"))
        {
          continue;
        }
        const std::string &tag    = summ.get_tag();
        const std::string &header = summ.has_value("name") ? summ.get_string("name") : tag;

        const std::string hint = summ.has_value("hint") ? summ.get_string("hint") : std::string{};
        std::string value      = std::visit(format_visitor, summ.get_value("value"));
        if (hint == "duration")
        {
          table.add_cell(row, tag, header + " (sec)", std::move(value));
        }
        else if (hint == "item_rate")
        {
          table.add_cell(row, tag, header + " (elem/sec)", std::move(value));
        }
        else if (hint == "bytes")
        {
          table.add_cell(row, tag, header + " (bytes)", std::move(value));
        }
        else if (hint == "byte_rate")
        {
          table.add_cell(row, tag, header + " (bytes/sec)", std::move(value));
        }
        else if (hint == "sample_size")
        {
          table.add_cell(row, tag, header, std::move(value));
        }
        else if (hint == "percentage")
        {
          table.add_cell(row, tag, header, std::move(value));
        }
        else
        {
          table.add_cell(row, tag, header, std::move(value));
        }
      }
      row++;
    }
  }

  if (table.m_columns.empty())
  { // No data.
    return;
  }

  // Pad with empty strings if needed.
  table.fix_row_lengths();

  fmt::memory_buffer buffer;
  { // Headers:
    std::size_t remaining = table.m_columns.size();
    for (const auto &col : table.m_columns)
    {
      fmt::format_to(std::back_inserter(buffer), "{}{}", col.header, (--remaining == 0) ? "" : ",");
    }
    fmt::format_to(std::back_inserter(buffer), "\n");
  }

  { // Rows
    for (std::size_t i = 0; i < table.m_num_rows; ++i)
    {
      std::size_t remaining = table.m_columns.size();
      for (const auto &col : table.m_columns)
      {
        fmt::format_to(std::back_inserter(buffer), "{}{}", col.rows[i], (--remaining == 0) ? "" : ",");
      }
      fmt::format_to(std::back_inserter(buffer), "\n");
    }
  }

  m_ostream << fmt::to_string(buffer);
}

} // namespace nvbench
