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

#include <nvbench/range.cuh>

#include "test_asserts.cuh"

void test_basic()
{
  ASSERT((nvbench::range(0, 6) ==
          std::vector<nvbench::int64_t>{0, 1, 2, 3, 4, 5, 6}));
  ASSERT((nvbench::range(0, 6, 1) ==
          std::vector<nvbench::int64_t>{0, 1, 2, 3, 4, 5, 6}));
  ASSERT(
    (nvbench::range(0, 6, 2) == std::vector<nvbench::int64_t>{0, 2, 4, 6}));
  ASSERT((nvbench::range(0, 6, 3) == std::vector<nvbench::int64_t>{0, 3, 6}));
  ASSERT((nvbench::range(0, 6, 4) == std::vector<nvbench::int64_t>{0, 4}));
  ASSERT((nvbench::range(0, 6, 5) == std::vector<nvbench::int64_t>{0, 5}));
  ASSERT((nvbench::range(0, 6, 7) == std::vector<nvbench::int64_t>{0}));
}

void test_result_type()
{
  // All ints should turn into int64 by default:
  ASSERT((std::is_same_v<decltype(nvbench::range(0ll, 1ll)),
                         std::vector<nvbench::int64_t>>));
  ASSERT((std::is_same_v<decltype(nvbench::range(0, 1)),
                         std::vector<nvbench::int64_t>>));
  ASSERT((std::is_same_v<decltype(nvbench::range(0u, 1u)),
                         std::vector<nvbench::int64_t>>));

  // All floats should turn into float64 by default:
  ASSERT((std::is_same_v<decltype(nvbench::range(0., 1.)),
                         std::vector<nvbench::float64_t>>));
  ASSERT((std::is_same_v<decltype(nvbench::range(0.f, 1.f)),
                         std::vector<nvbench::float64_t>>));

  // Other types may be explicitly specified:
  ASSERT((std::is_same_v<decltype(nvbench::range<nvbench::float32_t,
                                                 nvbench::float32_t>(0.f, 1.f)),
                         std::vector<nvbench::float32_t>>));
  ASSERT((std::is_same_v<
          decltype(nvbench::range<nvbench::int32_t, nvbench::int32_t>(0, 1)),
          std::vector<nvbench::int32_t>>));
}

void test_fp_tolerance()
{
  // Make sure that the range is padded a bit for floats to prevent rounding
  // errors from skipping `end`. This test will trigger failures without
  // the padding.
  const nvbench::float32_t start  = 0.1f;
  const nvbench::float32_t stride = 1e-4f;
  for (std::size_t size = 1; size < 1024; ++size)
  {
    const nvbench::float32_t end =
      start + stride * static_cast<nvbench::float32_t>(size - 1);
    ASSERT_MSG(nvbench::range(start, end, stride).size() == size,
               "size={}", size);
  }
}

int main()
{
  test_basic();
  test_result_type();
  test_fp_tolerance();
}
