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

#include <nvbench/float64_axis.cuh>

#include <fmt/format.h>

namespace nvbench
{

float64_axis::~float64_axis() = default;

std::string float64_axis::do_get_input_string(std::size_t i) const
{
  return fmt::format("{:0.5g}", m_values[i]);
}

std::string float64_axis::do_get_description(std::size_t) const { return {}; }

} // namespace nvbench
