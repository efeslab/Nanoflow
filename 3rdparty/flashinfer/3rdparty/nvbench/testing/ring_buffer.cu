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

#include <nvbench/detail/ring_buffer.cuh>

#include "test_asserts.cuh"

#include <algorithm>
#include <vector>

template <typename T>
bool equal(const nvbench::detail::ring_buffer<T> &buffer,
           const std::vector<T> &reference)
{
  return std::equal(buffer.begin(), buffer.end(), reference.begin());
}

int main()
try
{
  nvbench::detail::ring_buffer<int> avg(3);
  ASSERT(avg.capacity() == 3);
  ASSERT(avg.size() == 0);
  ASSERT(avg.empty());
  ASSERT(equal(avg, {0, 0, 0}));

  avg.push_back(32);
  ASSERT(!avg.empty());
  ASSERT(avg.size() == 1);
  ASSERT(avg.capacity() == 3);
  ASSERT_MSG(avg.back() == 32, " (got {})", avg.back());
  ASSERT(equal(avg, {32, 0, 0}));

  avg.push_back(2);
  ASSERT(avg.size() == 2);
  ASSERT(avg.capacity() == 3);
  ASSERT_MSG(avg.back() == 2, " (got {})", avg.back());
  ASSERT(equal(avg, {32, 2, 0}));

  avg.push_back(-15);
  ASSERT(avg.size() == 3);
  ASSERT(avg.capacity() == 3);
  ASSERT_MSG(avg.back() == -15, " (got {})", avg.back());
  ASSERT(equal(avg, {32, 2, -15}));

  avg.push_back(5);
  ASSERT(avg.size() == 3);
  ASSERT(avg.capacity() == 3);
  ASSERT_MSG(avg.back() == 5, " (got {})", avg.back());
  ASSERT(equal(avg, {2, -15, 5}));

  avg.push_back(0);
  ASSERT(avg.size() == 3);
  ASSERT(avg.capacity() == 3);
  ASSERT(equal(avg, {-15, 5, 0}));
  ASSERT_MSG(avg.back() == 0, " (got {})", avg.back());

  avg.push_back(128);
  ASSERT(avg.size() == 3);
  ASSERT(avg.capacity() == 3);
  ASSERT(equal(avg, {5, 0, 128}));
  ASSERT_MSG(avg.back() == 128, " (got {})", avg.back());

  avg.clear();
  ASSERT(avg.empty());
  ASSERT(avg.size() == 0);
  ASSERT(avg.capacity() == 3);

  return 0;
}
catch (std::exception &err)
{
  fmt::print(stderr, "{}", err.what());
  return 1;
}
