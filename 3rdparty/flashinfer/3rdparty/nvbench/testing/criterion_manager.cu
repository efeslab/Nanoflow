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
#include <nvbench/types.cuh>

#include "test_asserts.cuh"

void test_standard_criteria_exist()
{
  ASSERT(nvbench::criterion_manager::get().get_criterion("stdrel").get_name() == "stdrel");
  ASSERT(nvbench::criterion_manager::get().get_criterion("entropy").get_name() == "entropy");
}

class custom_criterion : public nvbench::stopping_criterion_base
{
public:
  custom_criterion()
      : nvbench::stopping_criterion_base("custom", nvbench::criterion_params{})
  {}

protected:
  virtual void do_initialize() override {}
  virtual void do_add_measurement(nvbench::float64_t /* measurement */) override {}
  virtual bool do_is_finished() override { return true; }
};

void test_no_duplicates_are_allowed()
{
  nvbench::criterion_manager& manager = nvbench::criterion_manager::get();
  bool exception_triggered = false;

  try {
    [[maybe_unused]] nvbench::stopping_criterion_base& _ = manager.get_criterion("custom");
  } catch(...) {
    exception_triggered = true;
  }
  ASSERT(exception_triggered);

  std::unique_ptr<custom_criterion> custom_ptr = std::make_unique<custom_criterion>();
  custom_criterion* custom_raw = custom_ptr.get();
  ASSERT(&manager.add(std::move(custom_ptr)) == custom_raw);

  nvbench::stopping_criterion_base& custom = nvbench::criterion_manager::get().get_criterion("custom");
  ASSERT(custom_raw == &custom);

  exception_triggered = false;
  try {
    manager.add(std::make_unique<custom_criterion>());
  } catch(...) {
    exception_triggered = true;
  }
  ASSERT(exception_triggered);
}

int main()
{
  test_standard_criteria_exist();
  test_no_duplicates_are_allowed();
}
