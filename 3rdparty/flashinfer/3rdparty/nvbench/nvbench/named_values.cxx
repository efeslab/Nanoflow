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

#include <nvbench/named_values.cuh>

#include <nvbench/config.cuh>
#include <nvbench/detail/throw.cuh>

#include <fmt/format.h>

#include <algorithm>
#include <iterator>
#include <stdexcept>
#include <type_traits>

namespace nvbench
{

void named_values::append(const named_values &other)
{
  m_storage.insert(m_storage.end(), other.m_storage.cbegin(), other.m_storage.cend());
}

void named_values::clear() { m_storage.clear(); }

std::size_t named_values::get_size() const { return m_storage.size(); }

std::vector<std::string> named_values::get_names() const
{
  std::vector<std::string> names;
  names.reserve(m_storage.size());
  std::transform(m_storage.cbegin(),
                 m_storage.cend(),
                 std::back_inserter(names),
                 [](const auto &val) { return val.name; });
  return names;
}

bool named_values::has_value(const std::string &name) const
{
  auto iter = std::find_if(m_storage.cbegin(), m_storage.cend(), [&name](const auto &val) {
    return val.name == name;
  });
  return iter != m_storage.cend();
}

const named_values::value_type &named_values::get_value(const std::string &name) const
{
  auto iter = std::find_if(m_storage.cbegin(), m_storage.cend(), [&name](const auto &val) {
    return val.name == name;
  });
  if (iter == m_storage.cend())
  {
    NVBENCH_THROW(std::runtime_error, "No value with name '{}'.", name);
  }
  return iter->value;
}

named_values::type named_values::get_type(const std::string &name) const
{
  return std::visit(
    [&name]([[maybe_unused]] auto &&arg) {
      using T = std::decay_t<decltype(arg)>;
      if constexpr (std::is_same_v<T, nvbench::int64_t>)
      {
        return nvbench::named_values::type::int64;
      }
      else if constexpr (std::is_same_v<T, nvbench::float64_t>)
      {
        return nvbench::named_values::type::float64;
      }
      else if constexpr (std::is_same_v<T, std::string>)
      {
        return nvbench::named_values::type::string;
      }
      // warning C4702: unreachable code
      // This is a future-proofing check, it'll be reachable if something breaks
      NVBENCH_MSVC_PUSH_DISABLE_WARNING(4702)
      NVBENCH_THROW(std::runtime_error, "Unknown variant type for entry '{}'.", name);
    },
    this->get_value(name));
  NVBENCH_MSVC_POP_WARNING()
}

nvbench::int64_t named_values::get_int64(const std::string &name) const
try
{
  return std::get<nvbench::int64_t>(this->get_value(name));
}
catch (std::exception &err)
{
  NVBENCH_THROW(std::runtime_error, "Error looking up int64 value `{}`:\n{}", name, err.what());
}

nvbench::float64_t named_values::get_float64(const std::string &name) const
try
{
  return std::get<nvbench::float64_t>(this->get_value(name));
}
catch (std::exception &err)
{
  NVBENCH_THROW(std::runtime_error, "Error looking up float64 value `{}`:\n{}", name, err.what());
}

const std::string &named_values::get_string(const std::string &name) const
try
{
  return std::get<std::string>(this->get_value(name));
}
catch (std::exception &err)
{
  NVBENCH_THROW(std::runtime_error, "Error looking up string value `{}`:\n{}", name, err.what());
}

void named_values::set_int64(std::string name, nvbench::int64_t value)
{
  m_storage.push_back({std::move(name), value_type{value}});
}

void named_values::set_float64(std::string name, nvbench::float64_t value)
{
  m_storage.push_back({std::move(name), value_type{value}});
}

void named_values::set_string(std::string name, std::string value)
{
  m_storage.push_back({std::move(name), value_type{std::move(value)}});
}

void named_values::set_value(std::string name, named_values::value_type value)
{
  m_storage.push_back({std::move(name), std::move(value)});
}

void named_values::remove_value(const std::string &name)
{
  auto iter = std::find_if(m_storage.begin(), m_storage.end(), [&name](const auto &val) {
    return val.name == name;
  });
  if (iter != m_storage.end())
  {
    m_storage.erase(iter);
  }
}

} // namespace nvbench
