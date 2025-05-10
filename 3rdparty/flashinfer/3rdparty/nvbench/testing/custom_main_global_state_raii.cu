/*
 *  Copyright 2024 NVIDIA Corporation
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

#include <nvbench/nvbench.cuh>

#include <algorithm>
#include <cstdlib>
#include <cstdio>

/******************************************************************************
 * Test having global state that is initialized and finalized via RAII.
 *****************************************************************************/

struct raii
{
  const char m_ref_data[6];
  char *m_data;
  bool m_cuda;

  const char *m_outer_data;
  bool m_outer_cuda;

  explicit raii(bool cuda, char *outer_data = nullptr, bool outer_cuda = false)
      : m_ref_data{'a', 'b', 'c', '1', '2', '3'}
      , m_data(nullptr)
      , m_cuda(cuda)
      , m_outer_data(outer_data)
      , m_outer_cuda(outer_cuda)
  {
    if (m_cuda)
    {
      printf("(%p) RAII test: allocating device memory\n", this);
      NVBENCH_CUDA_CALL(cudaMalloc(&m_data, 6));
      NVBENCH_CUDA_CALL(cudaMemcpy(m_data, m_ref_data, 6, cudaMemcpyHostToDevice));
    }
    else
    {
      printf("(%p) RAII test: allocating host memory\n", this);
      m_data = new char[6];
      std::copy(m_ref_data, m_ref_data + 6, m_data);
    }
  }

  ~raii()
  {
    this->verify();
    if (m_cuda)
    {
      printf("(%p) RAII test: invalidating device memory\n", this);
      NVBENCH_CUDA_CALL(cudaMemset(m_data, 0, 6));
      printf("(%p) RAII test: freeing device memory\n", this);
      NVBENCH_CUDA_CALL(cudaFree(m_data));
    }
    else
    {
      printf("(%p) RAII test: invalidating host memory\n", this);
      std::fill(m_data, m_data + 6, '\0');
      printf("(%p) RAII test: freeing host memory\n", this);
      delete[] m_data;
    }
  }

  void verify() noexcept
  {
    printf("(%p) RAII test: verifying instance state\n", this);
    this->verify(m_cuda, m_data);
    if (m_outer_data)
    {
      printf("(%p) RAII test: verifying outer state\n", this);
      this->verify(m_outer_cuda, m_outer_data);
    }
  }

  void verify(bool cuda, const char *data) noexcept
  {
    if (cuda)
    {
      char test_data[6];
      NVBENCH_CUDA_CALL(cudaMemcpy(test_data, data, 6, cudaMemcpyDeviceToHost));
      if (strncmp(test_data, m_ref_data, 6) != 0)
      {
        printf("(%p) RAII test failed: device data mismatch\n", this);
        std::exit(1);
      }
    }
    else
    {
      if (strncmp(data, m_ref_data, 6) != 0)
      {
        printf("(%p) RAII test failed: host data mismatch\n", this);
        std::exit(1);
      }
    }
  }
};

// These will be destroyed in the opposite order in which they are created:

#undef NVBENCH_MAIN_INITIALIZE_CUSTOM_PRE
#define NVBENCH_MAIN_INITIALIZE_CUSTOM_PRE(argc, argv) raii raii_outer(false);

#undef NVBENCH_MAIN_INITIALIZE_CUSTOM_POST
#define NVBENCH_MAIN_INITIALIZE_CUSTOM_POST(argc, argv)                                            \
  [[maybe_unused]] raii raii_inner(true, raii_outer.m_data, raii_outer.m_cuda);

NVBENCH_MAIN
