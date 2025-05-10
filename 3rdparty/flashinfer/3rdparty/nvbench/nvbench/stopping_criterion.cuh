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

#include <nvbench/named_values.cuh>
#include <nvbench/types.cuh>

#include <string>

#include <initializer_list>
#include <unordered_map>

namespace nvbench
{

namespace detail 
{

constexpr nvbench::float64_t compat_min_time() { return 0.5; }    // 0.5 seconds
constexpr nvbench::float64_t compat_max_noise() { return 0.005; } // 0.5% relative standard deviation

} // namespace detail

/**
 * Stores all the parameters for stopping criterion in use
 */
class criterion_params
{
  nvbench::named_values m_named_values;
public:
  criterion_params();
  criterion_params(std::initializer_list<std::pair<std::string, nvbench::named_values::value_type>>);

  /**
   * Set parameter values from another criterion_params object if they exist
   *
   * Parameters in `other` that do not correspond to parameters in `this` are ignored.
   */
  void set_from(const criterion_params &other);

  void set_int64(std::string name, nvbench::int64_t value);
  void set_float64(std::string name, nvbench::float64_t value);
  void set_string(std::string name, std::string value);

  [[nodiscard]] std::vector<std::string> get_names() const;
  [[nodiscard]] nvbench::named_values::type get_type(const std::string &name) const;

  [[nodiscard]] bool has_value(const std::string &name) const;
  [[nodiscard]] nvbench::int64_t get_int64(const std::string &name) const;
  [[nodiscard]] nvbench::float64_t get_float64(const std::string &name) const;
  [[nodiscard]] std::string get_string(const std::string &name) const;
};

/**
 * Stopping criterion interface
 */
class stopping_criterion_base
{
protected:
  std::string m_name;
  criterion_params m_params;

public:
  /**
   * @param name Unique name of the criterion
   * @param params Default values for all parameters of the criterion
   */
  explicit stopping_criterion_base(std::string name, criterion_params params)
      : m_name{std::move(name)}
      , m_params{std::move(params)}
  {}

  virtual ~stopping_criterion_base() = default;

  [[nodiscard]] const std::string &get_name() const { return m_name; }
  [[nodiscard]] const criterion_params &get_params() const { return m_params; }

  /**
   * Initialize the criterion with the given parameters
   *
   * This method is called once per benchmark run, before any measurements are provided.
   */
  void initialize(const criterion_params &params) 
  {
    m_params.set_from(params);
    this->do_initialize();
  }

  /**
   * Add the latest measurement to the criterion
   */
  void add_measurement(nvbench::float64_t measurement)
  {
    this->do_add_measurement(measurement);
  }

  /**
   * Check if the criterion has been met for all measurements processed by `add_measurement`
   */
  bool is_finished()
  {
    return this->do_is_finished();
  }

protected:
  /**
   * Initialize the criterion after updaring the parameters
   */
  virtual void do_initialize() = 0;

  /**
   * Add the latest measurement to the criterion
   */
  virtual void do_add_measurement(nvbench::float64_t measurement) = 0;

  /**
   * Check if the criterion has been met for all measurements processed by `add_measurement`
   */
  virtual bool do_is_finished() = 0;
};

} // namespace nvbench
