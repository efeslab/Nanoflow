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

#include <nvbench/json_printer.cuh>

#include <nvbench/axes_metadata.cuh>
#include <nvbench/benchmark_base.cuh>
#include <nvbench/config.cuh>
#include <nvbench/device_info.cuh>
#include <nvbench/device_manager.cuh>
#include <nvbench/git_revision.cuh>
#include <nvbench/state.cuh>
#include <nvbench/summary.cuh>
#include <nvbench/version.cuh>

#include <nvbench/detail/throw.cuh>

#include <fmt/format.h>

#include <nlohmann/json.hpp>

#include <cstdint>
#include <fstream>
#include <iterator>
#include <ostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if __has_include(<filesystem>)
#include <filesystem>
namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
static_assert(false, "No <filesystem> or <experimental/filesystem> found.");
#endif

#if NVBENCH_CPP_DIALECT >= 2020
#include <bit>
#endif

namespace
{

bool is_little_endian()
{
#if NVBENCH_CPP_DIALECT >= 2020
  return std::endian::native == std::endian::little;
#else
  const nvbench::uint32_t word = {0xBadDecaf};
  nvbench::uint8_t bytes[4];
  std::memcpy(bytes, &word, 4);
  return bytes[0] == 0xaf;
#endif
}

template <typename JsonNode>
void write_named_values(JsonNode &node, const nvbench::named_values &values)
{
  const auto value_names = values.get_names();
  for (const auto &value_name : value_names)
  {
    auto &value   = node.emplace_back();
    value["name"] = value_name;

    const auto type = values.get_type(value_name);
    switch (type)
    {
      case nvbench::named_values::type::int64:
        value["type"] = "int64";
        // Write as a string; JSON encodes all numbers as double-precision
        // floats, which would truncate int64s.
        value["value"] = fmt::to_string(values.get_int64(value_name));
        break;

      case nvbench::named_values::type::float64:
        value["type"] = "float64";
        // Write as a string for consistency with int64.
        value["value"] = fmt::to_string(values.get_float64(value_name));
        break;

      case nvbench::named_values::type::string:
        value["type"]  = "string";
        value["value"] = values.get_string(value_name);
        break;

      default:
        NVBENCH_THROW(std::runtime_error, "{}", "Unrecognized value type.");
    } // end switch (value type)
  }   // end foreach value name
}

} // end namespace

namespace nvbench
{

json_printer::version_t json_printer::get_json_file_version()
{
  // This version number should stay in sync with `file_version` in
  // scripts/nvbench_json/version.py.
  //
  // Use semantic versioning:
  // Major version: backwards incompatible changes
  // Minor version: backwards compatible additions
  // Patch version: backwards compatible bugfixes/patches
  return {1, 0, 0};
}

std::string json_printer::version_t::get_string() const
{
  return fmt::format("{}.{}.{}", this->major, this->minor, this->patch);
}

void json_printer::do_process_bulk_data_float64(state &state,
                                                const std::string &tag,
                                                const std::string &hint,
                                                const std::vector<nvbench::float64_t> &data)
{
  printer_base::do_process_bulk_data_float64(state, tag, hint, data);

  if (!m_enable_binary_output)
  {
    return;
  }

  if (hint == "sample_times")
  {
    nvbench::cpu_timer timer;
    timer.start();

    fs::path result_path{m_stream_name + "-bin/"};
    try
    {
      if (!fs::exists(result_path))
      {
        if (!fs::create_directory(result_path))
        {
          NVBENCH_THROW(std::runtime_error, "{}", "Failed to create result directory '{}'.");
        }
      }
      else if (!fs::is_directory(result_path))
      {
        NVBENCH_THROW(std::runtime_error, "{}", "'{}' exists and is not a directory.");
      }

      const auto file_id = m_num_jsonbin_files++;
      result_path /= fmt::format("{:d}.bin", file_id);

      std::ofstream out;
      out.exceptions(out.exceptions() | std::ios::failbit | std::ios::badbit);
      out.open(result_path, std::ios::binary | std::ios::out);

      // FIXME: SLOW -- Writing the binary file, 4 bytes at a time...
      // There are a lot of optimizations that could be done here if this ends
      // up being a noticeable bottleneck.
      for (auto value64 : data)
      {
        const auto value32 = static_cast<nvbench::float32_t>(value64);
        char buffer[4];
        std::memcpy(buffer, &value32, 4);
        // the c++17 implementation of is_little_endian isn't constexpr, but
        // all supported compilers optimize this branch as if it were.
        if (!is_little_endian())
        {
          using std::swap;
          swap(buffer[0], buffer[3]);
          swap(buffer[1], buffer[2]);
        }
        out.write(buffer, 4);
      }
    }
    catch (std::exception &e)
    {
      if (auto printer_opt_ref = state.get_benchmark().get_printer(); printer_opt_ref.has_value())
      {
        auto &printer = printer_opt_ref.value().get();
        printer.log(
          nvbench::log_level::warn,
          fmt::format("Error writing {} ({}) to {}: {}", tag, hint, result_path.string(), e.what()));
      }
    } // end catch

    auto &summ = state.add_summary(fmt::format("nv/json/bin:{}", tag));
    summ.set_string("name", "Samples Times File");
    summ.set_string("hint", "file/sample_times");
    summ.set_string("description",
                    "Binary file containing sample times as little-endian "
                    "float32.");
    summ.set_string("filename", result_path.string());
    summ.set_int64("size", static_cast<nvbench::int64_t>(data.size()));
    summ.set_string("hide", "Not needed in table.");

    timer.stop();
    if (auto printer_opt_ref = state.get_benchmark().get_printer(); printer_opt_ref.has_value())
    {
      auto &printer = printer_opt_ref.value().get();
      printer.log(
        nvbench::log_level::info,
        fmt::format("Wrote '{}' in {:>6.3f}ms", result_path.string(), timer.get_duration() * 1000));
    }
  } // end hint == sample_times
}

static void add_devices_section(nlohmann::ordered_json &root)
{
  auto &devices = root["devices"];
  for (const auto &dev_info : nvbench::device_manager::get().get_devices())
  {
    auto &device                    = devices.emplace_back();
    device["id"]                    = dev_info.get_id();
    device["name"]                  = dev_info.get_name();
    device["sm_version"]            = dev_info.get_sm_version();
    device["ptx_version"]           = dev_info.get_ptx_version();
    device["sm_default_clock_rate"] = dev_info.get_sm_default_clock_rate();
    device["number_of_sms"]         = dev_info.get_number_of_sms();
    device["max_blocks_per_sm"]     = dev_info.get_max_blocks_per_sm();
    device["max_threads_per_sm"]    = dev_info.get_max_threads_per_sm();
    device["max_threads_per_block"] = dev_info.get_max_threads_per_block();
    device["registers_per_sm"]      = dev_info.get_registers_per_sm();
    device["registers_per_block"]   = dev_info.get_registers_per_block();
    device["global_memory_size"]    = dev_info.get_global_memory_size();
    device["global_memory_bus_peak_clock_rate"] =
      dev_info.get_global_memory_bus_peak_clock_rate();
    device["global_memory_bus_width"]     = dev_info.get_global_memory_bus_width();
    device["global_memory_bus_bandwidth"] = dev_info.get_global_memory_bus_bandwidth();
    device["l2_cache_size"]               = dev_info.get_l2_cache_size();
    device["shared_memory_per_sm"]        = dev_info.get_shared_memory_per_sm();
    device["shared_memory_per_block"]     = dev_info.get_shared_memory_per_block();
    device["ecc_state"]                   = dev_info.get_ecc_state();
  }
}

void json_printer::do_print_benchmark_results(const benchmark_vector &benches)
{
  nlohmann::ordered_json root;

  {
    auto &metadata = root["meta"];

    {
      auto &argv = metadata["argv"];
      for (const auto &arg : m_argv)
      {
        argv.push_back(arg);
      }
    } // "argv"

    {
      auto &version = metadata["version"];

      {
        const auto version_info = json_printer::get_json_file_version();
        auto &json_version      = version["json"];

        json_version["major"]  = version_info.major;
        json_version["minor"]  = version_info.minor;
        json_version["patch"]  = version_info.patch;
        json_version["string"] = version_info.get_string();
      } // "json"

      {
        auto &nvb_version = version["nvbench"];

        nvb_version["major"]  = NVBENCH_VERSION_MAJOR;
        nvb_version["minor"]  = NVBENCH_VERSION_MINOR;
        nvb_version["patch"]  = NVBENCH_VERSION_PATCH;
        nvb_version["string"] = fmt::format("{}.{}.{}",
                                            NVBENCH_VERSION_MAJOR,
                                            NVBENCH_VERSION_MINOR,
                                            NVBENCH_VERSION_PATCH);

        nvb_version["git_branch"]  = NVBENCH_GIT_BRANCH;
        nvb_version["git_sha"]     = NVBENCH_GIT_SHA1;
        nvb_version["git_version"] = NVBENCH_GIT_VERSION;
        nvb_version["git_is_dirty"] =
#ifdef NVBENCH_GIT_IS_DIRTY
          true;
#else
          false;
#endif
      } // "nvbench"
    }   // "version"
  }     // "meta"

  add_devices_section(root);

  {
    auto &benchmarks = root["benchmarks"];
    for (const auto &bench_ptr : benches)
    {
      const auto bench_index = benchmarks.size();
      auto &bench            = benchmarks.emplace_back();

      bench["name"]  = bench_ptr->get_name();
      bench["index"] = bench_index;

      bench["min_samples"] = bench_ptr->get_min_samples();
      bench["min_time"]    = bench_ptr->get_min_time();
      bench["max_noise"]   = bench_ptr->get_max_noise();
      bench["skip_time"]   = bench_ptr->get_skip_time();
      bench["timeout"]     = bench_ptr->get_timeout();

      auto &devices = bench["devices"];
      for (const auto &dev_info : bench_ptr->get_devices())
      {
        devices.push_back(dev_info.get_id());
      }

      auto &axes = bench["axes"];
      for (const auto &axis_ptr : bench_ptr->get_axes().get_axes())
      {
        auto &axis = axes.emplace_back();

        axis["name"]  = axis_ptr->get_name();
        axis["type"]  = axis_ptr->get_type_as_string();
        axis["flags"] = axis_ptr->get_flags_as_string();

        auto &values         = axis["values"];
        const auto axis_size = axis_ptr->get_size();
        for (std::size_t i = 0; i < axis_size; ++i)
        {
          auto &value           = values.emplace_back();
          value["input_string"] = axis_ptr->get_input_string(i);
          value["description"]  = axis_ptr->get_description(i);

          switch (axis_ptr->get_type())
          {
            case nvbench::axis_type::type:
              value["is_active"] = static_cast<type_axis &>(*axis_ptr).get_is_active(i);
              break;

            case nvbench::axis_type::int64:
              value["value"] = static_cast<int64_axis &>(*axis_ptr).get_value(i);
              break;

            case nvbench::axis_type::float64:
              value["value"] = static_cast<float64_axis &>(*axis_ptr).get_value(i);
              break;

            case nvbench::axis_type::string:
              value["value"] = static_cast<string_axis &>(*axis_ptr).get_value(i);
              break;
            default:
              break;
          } // end switch (axis type)
        }   // end foreach axis value
      }     // end foreach axis

      auto &states = bench["states"];
      for (const auto &exec_state : bench_ptr->get_states())
      {
        auto &st = states.emplace_back();

        st["name"] = exec_state.get_axis_values_as_string();

        st["min_samples"] = exec_state.get_min_samples();
        st["min_time"]    = exec_state.get_min_time();
        st["max_noise"]   = exec_state.get_max_noise();
        st["skip_time"]   = exec_state.get_skip_time();
        st["timeout"]     = exec_state.get_timeout();

        st["device"]            = exec_state.get_device()->get_id();
        st["type_config_index"] = exec_state.get_type_config_index();

        // TODO I'd like to replace this with:
        //  [ {"name" : <axis name>, "index": <value_index>}, ...]
        // but it would take some refactoring in the data structures to get
        // that information through.
        ::write_named_values(st["axis_values"], exec_state.get_axis_values());

        auto &summaries = st["summaries"];
        for (const auto &exec_summ : exec_state.get_summaries())
        {
          auto &summ  = summaries.emplace_back();
          summ["tag"] = exec_summ.get_tag();

          // Write out the expected values as simple key/value pairs
          nvbench::named_values summary_values = exec_summ;
          if (summary_values.has_value("name"))
          {
            summ["name"] = summary_values.get_string("name");
            summary_values.remove_value("name");
          }
          if (summary_values.has_value("description"))
          {
            summ["description"] = summary_values.get_string("description");
            summary_values.remove_value("description");
          }
          if (summary_values.has_value("hint"))
          {
            summ["hint"] = summary_values.get_string("hint");
            summary_values.remove_value("hint");
          }
          if (summary_values.has_value("hide"))
          {
            summ["hide"] = summary_values.get_string("hide");
            summary_values.remove_value("hide");
          }

          // Write any additional values generically in
          // ["data"] = [{name,type,value}, ...]:
          if (summary_values.get_size() != 0)
          {
            ::write_named_values(summ["data"], summary_values);
          }
        }

        st["is_skipped"] = exec_state.is_skipped();
        if (exec_state.is_skipped())
        {
          st["skip_reason"] = exec_state.get_skip_reason();
          continue;
        }
      } // end foreach exec_state
    }   // end foreach benchmark
  }     // "benchmarks"

  m_ostream << root.dump(2) << "\n";
}

void json_printer::do_print_benchmark_list(const benchmark_vector &benches)
{
  if (benches.empty())
  {
    return;
  }

  nlohmann::ordered_json root;
  auto &benchmarks = root["benchmarks"];

  for (const auto &bench_ptr : benches)
  {
    const auto bench_index = benchmarks.size();
    auto &bench            = benchmarks.emplace_back();

    bench["name"]  = bench_ptr->get_name();
    bench["index"] = bench_index;

    // We have to ensure that the axes are represented as an array, not an
    // nil object when there are no axes.
    auto &axes = bench["axes"] = nlohmann::json::array();

    for (const auto &axis_ptr : bench_ptr->get_axes().get_axes())
    {
      auto &axis = axes.emplace_back();

      axis["name"]  = axis_ptr->get_name();
      axis["type"]  = axis_ptr->get_type_as_string();
      axis["flags"] = axis_ptr->get_flags_as_string();

      auto &values         = axis["values"];
      const auto axis_size = axis_ptr->get_size();
      for (std::size_t i = 0; i < axis_size; ++i)
      {
        auto &value           = values.emplace_back();
        value["input_string"] = axis_ptr->get_input_string(i);
        value["description"]  = axis_ptr->get_description(i);

        switch (axis_ptr->get_type())
        {
          case nvbench::axis_type::int64:
            value["value"] = static_cast<int64_axis &>(*axis_ptr).get_value(i);
            break;

          case nvbench::axis_type::float64:
            value["value"] = static_cast<float64_axis &>(*axis_ptr).get_value(i);
            break;

          case nvbench::axis_type::string:
            value["value"] = static_cast<string_axis &>(*axis_ptr).get_value(i);
            break;

          default:
            break;
        } // end switch (axis type)
      }   // end foreach axis value
    }
  } // end foreach bench

  m_ostream << root.dump(2) << "\n";
}

void json_printer::print_devices_json()
{
  nlohmann::ordered_json root;
  add_devices_section(root);
  m_ostream << root.dump(2) << "\n";
}

} // namespace nvbench
