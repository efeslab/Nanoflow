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

#pragma once

#include <nvbench/detail/entropy_criterion.cuh>
#include <nvbench/detail/stdrel_criterion.cuh>
#include <nvbench/stopping_criterion.cuh>
#include <nvbench/types.cuh>

#include <memory>

#include <unordered_map>

namespace nvbench
{

class criterion_manager
{
  std::unordered_map<std::string, std::unique_ptr<nvbench::stopping_criterion_base>> m_map;

  criterion_manager();

public:
  /**
   * @return The singleton criterion_manager instance.
   */
  static criterion_manager& get();

  /**
   * Register a new stopping criterion.
   */
  nvbench::stopping_criterion_base& add(std::unique_ptr<nvbench::stopping_criterion_base> criterion);
  nvbench::stopping_criterion_base& get_criterion(const std::string& name);
  const nvbench::stopping_criterion_base& get_criterion(const std::string& name) const;

  using params_description = std::vector<std::pair<std::string, nvbench::named_values::type>>;
  params_description get_params_description() const;
};

/**
 * Given a stopping criterion type `TYPE`, registers it in the criterion manager
 *
 * See the `custom_criterion.cu` example for usage.
 */
#define NVBENCH_REGISTER_CRITERION(TYPE)                                                           \
  static nvbench::stopping_criterion_base &NVBENCH_UNIQUE_IDENTIFIER(TYPE) =                       \
    nvbench::criterion_manager::get().add(std::make_unique<TYPE>())

} // namespace nvbench
