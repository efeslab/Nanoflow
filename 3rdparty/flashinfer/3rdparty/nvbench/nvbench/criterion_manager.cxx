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

#include <nvbench/criterion_manager.cuh>
#include <nvbench/detail/throw.cuh>

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace nvbench
{

criterion_manager::criterion_manager()
{
  m_map.emplace("stdrel", std::make_unique<nvbench::detail::stdrel_criterion>());
  m_map.emplace("entropy", std::make_unique<nvbench::detail::entropy_criterion>());
}

criterion_manager &criterion_manager::get()
{
  static criterion_manager registry;
  return registry;
}

stopping_criterion_base& criterion_manager::get_criterion(const std::string& name)
{
  auto iter = m_map.find(name);
  if (iter == m_map.end())
  {
    NVBENCH_THROW(std::runtime_error, "No stopping criterion named \"{}\".", name);
  }
  return *iter->second.get();
}

const nvbench::stopping_criterion_base& criterion_manager::get_criterion(const std::string& name) const
{
  auto iter = m_map.find(name);
  if (iter == m_map.end())
  {
    NVBENCH_THROW(std::runtime_error, "No stopping criterion named \"{}\".", name);
  }
  return *iter->second.get();
}

stopping_criterion_base &criterion_manager::add(std::unique_ptr<stopping_criterion_base> criterion)
{
  const std::string name = criterion->get_name();

  auto [it, success] = m_map.emplace(name, std::move(criterion));

  if (!success)
  {
    NVBENCH_THROW(std::runtime_error,
                  "Stopping criterion \"{}\" is already registered.", name);
  }

  return *it->second.get();
}

nvbench::criterion_manager::params_description criterion_manager::get_params_description() const
{
  nvbench::criterion_manager::params_description desc;

  for (auto &[criterion_name, criterion] : m_map)
  {
    nvbench::criterion_params params = criterion->get_params();

    for (auto param : params.get_names())
    {
      nvbench::named_values::type type = params.get_type(param);
      if (std::find_if(desc.begin(), desc.end(), [&](auto d) {
            return d.first == param && d.second != type;
          }) != desc.end())
      {
        NVBENCH_THROW(std::runtime_error,
                      "Stopping criterion \"{}\" parameter \"{}\" is already used by another "
                      "criterion with a different type.",
                      criterion_name,
                      param);
      }
      desc.emplace_back(param, type);
    }
  }

  return desc;
}

} // namespace nvbench
