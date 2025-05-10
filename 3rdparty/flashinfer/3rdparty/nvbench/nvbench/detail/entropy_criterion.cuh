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

#include <nvbench/types.cuh>
#include <nvbench/stopping_criterion.cuh>
#include <nvbench/detail/ring_buffer.cuh>

#include <vector>

namespace nvbench::detail
{

class entropy_criterion final : public stopping_criterion_base
{
  // state
  nvbench::int64_t m_total_samples{};
  nvbench::float64_t m_total_cuda_time{};
  std::vector<std::pair<nvbench::float64_t, nvbench::int64_t>> m_freq_tracker;

  // TODO The window size should be user-configurable
  nvbench::detail::ring_buffer<nvbench::float64_t> m_entropy_tracker{299};

  // Used to avoid re-allocating temporary memory
  std::vector<nvbench::float64_t> m_probabilities; 

  nvbench::float64_t compute_entropy();

public:
  entropy_criterion();

protected:
  virtual void do_initialize() override;
  virtual void do_add_measurement(nvbench::float64_t measurement) override;
  virtual bool do_is_finished() override;
  
};

} // namespace nvbench::detail
