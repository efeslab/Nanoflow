/*
 *  Copyright 2023 NVIDIA Corporation
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

#include <nvbench/stopping_criterion.cuh>

#include <nvbench/detail/throw.cuh>


namespace nvbench
{

// Default constructor for compatibility with old code
criterion_params::criterion_params()
    : criterion_params{{"max-noise", nvbench::detail::compat_max_noise()},
                       {"min-time", nvbench::detail::compat_min_time()}}
{}

criterion_params::criterion_params(
  std::initializer_list<std::pair<std::string, nvbench::named_values::value_type>> list)
{
  for (const auto &[name, value] : list)
  {
    m_named_values.set_value(name, value);
  }
}

void criterion_params::set_from(const criterion_params &other)
{
  for (const std::string &name : this->get_names())
  {
    if (other.has_value(name))
    {
      if (this->get_type(name) != other.get_type(name))
      {
        NVBENCH_THROW(std::runtime_error,
                      "Mismatched types for named value \"{}\". "
                      "Expected {}, got {}.",
                      name,
                      static_cast<int>(this->get_type(name)),
                      static_cast<int>(other.get_type(name)));
      }
      m_named_values.remove_value(name);
      m_named_values.set_value(name, other.m_named_values.get_value(name));
    }
  }
}

void criterion_params::set_int64(std::string name, nvbench::int64_t value)
{
  if (m_named_values.has_value(name)) 
  {
    m_named_values.remove_value(name);
  }

  m_named_values.set_int64(name, value);
}

void criterion_params::set_float64(std::string name, nvbench::float64_t value)
{
  if (m_named_values.has_value(name)) 
  {
    m_named_values.remove_value(name);
  }

  m_named_values.set_float64(name, value);
}

void criterion_params::set_string(std::string name, std::string value)
{
  if (m_named_values.has_value(name)) 
  {
    m_named_values.remove_value(name);
  }

  m_named_values.set_string(name, std::move(value));
}

bool criterion_params::has_value(const std::string &name) const
{
  return m_named_values.has_value(name);
}

nvbench::int64_t criterion_params::get_int64(const std::string &name) const
{
  return m_named_values.get_int64(name);
}

nvbench::float64_t criterion_params::get_float64(const std::string &name) const
{
  return m_named_values.get_float64(name);
}

std::string criterion_params::get_string(const std::string &name) const
{
  return m_named_values.get_string(name);
}

std::vector<std::string> criterion_params::get_names() const
{
  return m_named_values.get_names();
}

nvbench::named_values::type criterion_params::get_type(const std::string &name) const
{
  return m_named_values.get_type(name);
}


} // namespace nvbench::detail
