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

#pragma once

#include <utility>

// Many compilers still don't ship transform_reduce with their STLs, so here's
// a naive implementation that will work everywhere. This is never used in a
// critical section, so perf isn't a concern.

namespace nvbench::detail
{

template <typename InIterT, typename InitValueT, typename ReduceOp, typename TransformOp>
InitValueT transform_reduce(InIterT first,
                            InIterT last,
                            InitValueT init,
                            ReduceOp &&reduceOp,
                            TransformOp &&transformOp)
{
  while (first != last)
  {
    init = reduceOp(std::move(init), transformOp(*first++));
  }
  return init;
}

} // namespace nvbench::detail
