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

#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace nvbench
{

enum class axis_type
{
  type,
  int64,
  float64,
  string
};

std::string_view axis_type_to_string(axis_type);

struct axis_base
{
  virtual ~axis_base();

  [[nodiscard]] std::unique_ptr<axis_base> clone() const;

  [[nodiscard]] const std::string &get_name() const { return m_name; }
  void set_name(std::string name) { m_name = std::move(name); }

  [[nodiscard]] axis_type get_type() const { return m_type; }

  [[nodiscard]] std::string_view get_type_as_string() const { return axis_type_to_string(m_type); }

  [[nodiscard]] std::string_view get_flags_as_string() const
  {
    return this->do_get_flags_as_string();
  }

  [[nodiscard]] std::size_t get_size() const { return this->do_get_size(); }

  [[nodiscard]] std::string get_input_string(std::size_t i) const
  {
    return this->do_get_input_string(i);
  }

  [[nodiscard]] std::string get_description(std::size_t i) const
  {
    return this->do_get_description(i);
  }

protected:
  axis_base(std::string name, axis_type type)
      : m_name{std::move(name)}
      , m_type{type}
  {}

private:
  virtual std::unique_ptr<axis_base> do_clone() const          = 0;
  virtual std::size_t do_get_size() const                      = 0;
  virtual std::string do_get_input_string(std::size_t i) const = 0;
  virtual std::string do_get_description(std::size_t i) const  = 0;

  virtual std::string_view do_get_flags_as_string() const { return {}; };

  std::string m_name;
  axis_type m_type;
};

inline std::string_view axis_type_to_string(axis_type type)
{
  switch (type)
  {
    case axis_type::type:
      return "type";
      break;
    case axis_type::int64:
      return "int64";
      break;
    case axis_type::float64:
      return "float64";
      break;
    case axis_type::string:
      return "string";
      break;
  }
  throw std::runtime_error{"nvbench::axis_type_to_string Invalid axis_type."};
}

} // namespace nvbench
