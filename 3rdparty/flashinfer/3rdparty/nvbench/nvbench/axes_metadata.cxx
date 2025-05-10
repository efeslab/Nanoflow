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

#include <nvbench/axes_metadata.cuh>

#include <nvbench/detail/throw.cuh>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <stdexcept>

namespace nvbench
{

axes_metadata::axes_metadata(const axes_metadata &other)
{
  m_axes.reserve(other.get_axes().size());
  for (const auto &axis : other.get_axes())
  {
    m_axes.push_back(axis->clone());
  }
}

axes_metadata &axes_metadata::operator=(const axes_metadata &other)
{
  m_axes.clear();
  m_axes.reserve(other.get_axes().size());
  for (const auto &axis : other.get_axes())
  {
    m_axes.push_back(axis->clone());
  }
  return *this;
}

void axes_metadata::set_type_axes_names(std::vector<std::string> names)
try
{
  if (names.size() < m_axes.size())
  {
    NVBENCH_THROW(std::runtime_error,
                  "Number of names exceeds number of axes ({}).",
                  m_axes.size());
  }

  for (std::size_t i = 0; i < names.size(); ++i)
  {
    auto &axis = *m_axes[i];
    if (axis.get_type() != nvbench::axis_type::type)
    {
      NVBENCH_THROW(std::runtime_error, "Number of names exceeds number of type axes ({})", i);
    }

    axis.set_name(std::move(names[i]));
  }
}
catch (std::exception &e)
{
  NVBENCH_THROW(std::runtime_error,
                "Error in set_type_axes_names:\n{}\n"
                "TypeAxesNames: {}",
                e.what(),
                names);
}

void axes_metadata::add_float64_axis(std::string name, std::vector<nvbench::float64_t> data)
{
  auto axis = std::make_unique<nvbench::float64_axis>(std::move(name));
  axis->set_inputs(std::move(data));
  m_axes.push_back(std::move(axis));
}

void axes_metadata::add_int64_axis(std::string name,
                                   std::vector<nvbench::int64_t> data,
                                   nvbench::int64_axis_flags flags)
{
  auto axis = std::make_unique<nvbench::int64_axis>(std::move(name));
  axis->set_inputs(std::move(data), flags);
  m_axes.push_back(std::move(axis));
}

void axes_metadata::add_string_axis(std::string name, std::vector<std::string> data)
{
  auto axis = std::make_unique<nvbench::string_axis>(std::move(name));
  axis->set_inputs(std::move(data));
  m_axes.push_back(std::move(axis));
}

const int64_axis &axes_metadata::get_int64_axis(std::string_view name) const
{
  const auto &axis = this->get_axis(name, nvbench::axis_type::int64);
  return static_cast<const nvbench::int64_axis &>(axis);
}

int64_axis &axes_metadata::get_int64_axis(std::string_view name)
{
  auto &axis = this->get_axis(name, nvbench::axis_type::int64);
  return static_cast<nvbench::int64_axis &>(axis);
}

const float64_axis &axes_metadata::get_float64_axis(std::string_view name) const
{
  const auto &axis = this->get_axis(name, nvbench::axis_type::float64);
  return static_cast<const nvbench::float64_axis &>(axis);
}

float64_axis &axes_metadata::get_float64_axis(std::string_view name)
{
  auto &axis = this->get_axis(name, nvbench::axis_type::float64);
  return static_cast<nvbench::float64_axis &>(axis);
}

const string_axis &axes_metadata::get_string_axis(std::string_view name) const
{
  const auto &axis = this->get_axis(name, nvbench::axis_type::string);
  return static_cast<const nvbench::string_axis &>(axis);
}

string_axis &axes_metadata::get_string_axis(std::string_view name)
{
  auto &axis = this->get_axis(name, nvbench::axis_type::string);
  return static_cast<nvbench::string_axis &>(axis);
}

const type_axis &axes_metadata::get_type_axis(std::string_view name) const
{
  const auto &axis = this->get_axis(name, nvbench::axis_type::type);
  return static_cast<const nvbench::type_axis &>(axis);
}

type_axis &axes_metadata::get_type_axis(std::string_view name)
{
  auto &axis = this->get_axis(name, nvbench::axis_type::type);
  return static_cast<nvbench::type_axis &>(axis);
}

const nvbench::type_axis &axes_metadata::get_type_axis(std::size_t index) const
{
  for (const auto &axis : m_axes)
  {
    if (axis->get_type() == nvbench::axis_type::type)
    {
      const type_axis &t_axis = static_cast<const type_axis &>(*axis);
      if (t_axis.get_axis_index() == index)
      {
        return t_axis;
      }
    }
  }
  NVBENCH_THROW(std::runtime_error, "Invalid type axis index: {}.", index);
}

nvbench::type_axis &axes_metadata::get_type_axis(std::size_t index)
{
  for (auto &axis : m_axes)
  {
    if (axis->get_type() == nvbench::axis_type::type)
    {
      type_axis &t_axis = static_cast<type_axis &>(*axis);
      if (t_axis.get_axis_index() == index)
      {
        return t_axis;
      }
    }
  }
  NVBENCH_THROW(std::runtime_error, "Invalid type axis index: {}.", index);
}

const axis_base &axes_metadata::get_axis(std::string_view name) const
{
  auto iter = std::find_if(m_axes.cbegin(), m_axes.cend(), [&name](const auto &axis) {
    return axis->get_name() == name;
  });

  if (iter == m_axes.cend())
  {
    NVBENCH_THROW(std::runtime_error, "Axis '{}' not found.", name);
  }

  return **iter;
}

axis_base &axes_metadata::get_axis(std::string_view name)
{
  auto iter = std::find_if(m_axes.begin(), m_axes.end(), [&name](const auto &axis) {
    return axis->get_name() == name;
  });

  if (iter == m_axes.end())
  {
    NVBENCH_THROW(std::runtime_error, "Axis '{}' not found.", name);
  }

  return **iter;
}

const axis_base &axes_metadata::get_axis(std::string_view name, nvbench::axis_type type) const
{
  const auto &axis = this->get_axis(name);
  if (axis.get_type() != type)
  {
    NVBENCH_THROW(std::runtime_error,
                  "Axis '{}' type mismatch (expected {}, actual {}).",
                  name,
                  axis_type_to_string(type),
                  axis.get_type_as_string());
  }
  return axis;
}

axis_base &axes_metadata::get_axis(std::string_view name, nvbench::axis_type type)
{
  auto &axis = this->get_axis(name);
  if (axis.get_type() != type)
  {
    NVBENCH_THROW(std::runtime_error,
                  "Axis '{}' type mismatch (expected {}, actual {}).",
                  name,
                  axis_type_to_string(type),
                  axis.get_type_as_string());
  }
  return axis;
}

std::vector<std::string> axes_metadata::generate_default_type_axis_names(std::size_t num_type_axes)
{
  switch (num_type_axes)
  {
    case 0:
      return {};
    case 1:
      return {"T"};
    case 2:
      return {"T", "U"};
    case 3:
      return {"T", "U", "V"};
    case 4:
      return {"T", "U", "V", "W"};
    default:
      break;
  }

  std::vector<std::string> result;
  result.reserve(num_type_axes);
  for (std::size_t i = 0; i < num_type_axes; ++i)
  {
    result.emplace_back(fmt::format("T{}", i));
  }
  return result;
}

} // namespace nvbench
