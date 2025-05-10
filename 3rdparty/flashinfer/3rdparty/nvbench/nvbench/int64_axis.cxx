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

#include <nvbench/int64_axis.cuh>

#include <nvbench/detail/throw.cuh>

#include <fmt/format.h>

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace nvbench
{

int64_axis::~int64_axis() = default;

void int64_axis::set_inputs(std::vector<int64_t> inputs, int64_axis_flags flags)
{
  m_inputs = std::move(inputs);
  m_flags  = flags;

  if (!this->is_power_of_two())
  {
    m_values = m_inputs;
  }
  else
  {
    m_values.resize(m_inputs.size());

    auto conv = [](int64_t in) -> int64_t {
      if (in < 0 || in >= 64)
      {
        NVBENCH_THROW(std::runtime_error,
                      "Input value exceeds valid range for power-of-two mode. "
                      "Input={} ValidRange=[0, 63]",
                      in);
      }
      return int64_axis::compute_pow2(in);
    };

    std::transform(m_inputs.cbegin(), m_inputs.cend(), m_values.begin(), conv);
  }
}

std::string int64_axis::do_get_input_string(std::size_t i) const
{
  return fmt::to_string(m_inputs[i]);
}

std::string int64_axis::do_get_description(std::size_t i) const
{
  return this->is_power_of_two() ? fmt::format("2^{} = {}", m_inputs[i], m_values[i])
                                 : std::string{};
}

std::string_view int64_axis::do_get_flags_as_string() const
{
  if (static_cast<bool>(m_flags & nvbench::int64_axis_flags::power_of_two))
  {
    return "pow2";
  }
  return {};
}

} // namespace nvbench
