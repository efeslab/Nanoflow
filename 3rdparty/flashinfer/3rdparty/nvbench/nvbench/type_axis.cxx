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

#include <nvbench/type_axis.cuh>

#include <nvbench/detail/throw.cuh>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <stdexcept>

namespace nvbench
{

type_axis::~type_axis() = default;

void type_axis::set_active_inputs(const std::vector<std::string> &inputs)
{
  m_mask.clear();
  m_mask.resize(m_input_strings.size(), false);
  for (const auto &input : inputs)
  {
    const auto idx = this->get_type_index(input);
    m_mask[idx]    = true;
  }
}

bool type_axis::get_is_active(const std::string &input) const
{
  return this->get_is_active(this->get_type_index(input));
}

bool type_axis::get_is_active(std::size_t idx) const { return m_mask.at(idx); }

std::size_t type_axis::get_active_count() const
{
  return static_cast<std::size_t>(std::count(m_mask.cbegin(), m_mask.cend(), true));
}

std::size_t type_axis::get_type_index(const std::string &input_string) const
{
  auto it = std::find(m_input_strings.cbegin(), m_input_strings.cend(), input_string);
  if (it == m_input_strings.end())
  {
    NVBENCH_THROW(std::runtime_error,
                  "Invalid input string '{}' for type_axis `{}`.\n"
                  "Valid input strings: {}",
                  input_string,
                  this->get_name(),
                  m_input_strings);
  }

  return static_cast<std::size_t>(it - m_input_strings.cbegin());
}

} // namespace nvbench
