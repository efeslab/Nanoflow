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

#include <nvbench/type_list.cuh>
#include <nvbench/type_strings.cuh>

#include <string>
#include <vector>

namespace nvbench
{

struct type_axis final : public axis_base
{
  type_axis(std::string name, std::size_t axis_index)
      : axis_base{std::move(name), axis_type::type}
      , m_input_strings{}
      , m_descriptions{}
      , m_axis_index{axis_index}
  {}

  ~type_axis() final;

  template <typename TypeList>
  void set_inputs();

  void set_active_inputs(const std::vector<std::string> &inputs);

  [[nodiscard]] bool get_is_active(const std::string &input) const;
  [[nodiscard]] bool get_is_active(std::size_t index) const;
  [[nodiscard]] std::size_t get_active_count() const;

  /**
   * The index of this axis in the `benchmark`'s `type_axes` type list.
   */
  [[nodiscard]] std::size_t get_axis_index() const { return m_axis_index; }

  /**
   * The index in this axis of the type with the specified `input_string`.
   */
  [[nodiscard]] std::size_t get_type_index(const std::string &input_string) const;

private:
  std::unique_ptr<axis_base> do_clone() const final { return std::make_unique<type_axis>(*this); }
  std::size_t do_get_size() const final { return m_input_strings.size(); }
  std::string do_get_input_string(std::size_t i) const final { return m_input_strings[i]; }
  std::string do_get_description(std::size_t i) const final { return m_descriptions[i]; }

  std::vector<std::string> m_input_strings;
  std::vector<std::string> m_descriptions;
  // Use a mask to store active flags; inactive types must still be indexed
  // to keep things synced with the benchmark's type_axes typelist, which is
  // not mutable. Use `get_is_active` to determine whether an entry is active.
  std::vector<bool> m_mask;
  std::size_t m_axis_index;
};

template <typename TypeList>
void type_axis::set_inputs()
{
  // Need locals for lambda capture...
  auto &input_strings = m_input_strings;
  auto &descriptions  = m_descriptions;
  nvbench::tl::foreach<TypeList>(
    [&input_strings, &descriptions]([[maybe_unused]] auto wrapped_type) {
      using T       = typename decltype(wrapped_type)::type;
      using Strings = nvbench::type_strings<T>;
      input_strings.push_back(Strings::input_string());
      descriptions.push_back(Strings::description());
    });

  m_mask.clear();
  m_mask.resize(m_input_strings.size(), true);
}

} // namespace nvbench
