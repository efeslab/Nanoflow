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

#include <nvbench/printer_base.cuh>

#include <ostream>

namespace nvbench
{

printer_base::printer_base(std::ostream &ostream, std::string stream_name)
    : m_ostream{ostream}
    , m_stream_name{std::move(stream_name)}
{}

// Defined here to keep <ostream> out of the header
printer_base::~printer_base() = default;

void printer_base::do_set_completed_state_count(std::size_t states)
{
  m_completed_state_count = states;
}

void printer_base::do_add_completed_state() { ++m_completed_state_count; }

std::size_t printer_base::do_get_completed_state_count() const { return m_completed_state_count; }

void printer_base::do_set_total_state_count(std::size_t states) { m_total_state_count = states; }

std::size_t printer_base::do_get_total_state_count() const { return m_total_state_count; }

} // namespace nvbench
