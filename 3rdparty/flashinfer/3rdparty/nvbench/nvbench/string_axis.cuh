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

#include <nvbench/axis_base.cuh>

#include <nvbench/types.cuh>

#include <vector>

namespace nvbench
{

struct string_axis final : public axis_base
{
  explicit string_axis(std::string name)
      : axis_base{std::move(name), axis_type::string}
      , m_values{}
  {}

  ~string_axis() final;

  void set_inputs(std::vector<std::string> inputs) { m_values = std::move(inputs); }
  [[nodiscard]] const std::string &get_value(std::size_t i) const { return m_values[i]; }

private:
  std::unique_ptr<axis_base> do_clone() const final { return std::make_unique<string_axis>(*this); }
  std::size_t do_get_size() const final { return m_values.size(); }
  std::string do_get_input_string(std::size_t i) const final { return m_values[i]; }
  std::string do_get_description(std::size_t) const final { return {}; }

  std::vector<std::string> m_values;
};

} // namespace nvbench
