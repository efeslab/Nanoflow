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

#include <nvbench/types.cuh>

#include <string>
#include <variant>
#include <vector>

namespace nvbench
{

/**
 * Maintains a map of key / value pairs where the keys are names and the
 * values may be int64s, float64s, or strings.
 */
struct named_values
{
  using value_type = std::variant<nvbench::int64_t, nvbench::float64_t, std::string>;

  enum class type
  {
    int64,
    float64,
    string
  };

  void append(const named_values &other);

  [[nodiscard]] std::size_t get_size() const;
  [[nodiscard]] std::vector<std::string> get_names() const;

  void set_value(std::string name, value_type value);

  void set_int64(std::string name, nvbench::int64_t value);
  void set_float64(std::string name, nvbench::float64_t value);
  void set_string(std::string name, std::string value);

  [[nodiscard]] nvbench::int64_t get_int64(const std::string &name) const;
  [[nodiscard]] nvbench::float64_t get_float64(const std::string &name) const;
  [[nodiscard]] const std::string &get_string(const std::string &name) const;

  [[nodiscard]] type get_type(const std::string &name) const;
  [[nodiscard]] bool has_value(const std::string &name) const;
  [[nodiscard]] const value_type &get_value(const std::string &name) const;

  void clear();

  void remove_value(const std::string &name);

private:
  struct named_value
  {
    std::string name;
    value_type value;
  };
  // Use a vector to preserve order:
  using storage_type = std::vector<named_value>;

  storage_type m_storage;
};

} // namespace nvbench
