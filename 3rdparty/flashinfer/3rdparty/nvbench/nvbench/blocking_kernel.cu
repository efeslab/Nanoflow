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

#include <nvbench/blocking_kernel.cuh>

#include <nvbench/cuda_call.cuh>
#include <nvbench/cuda_stream.cuh>
#include <nvbench/types.cuh>

#include <nvbench/detail/throw.cuh>

#include <cuda/std/chrono>

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace
{

// Once launched, this kernel will block the stream until `flag` updates is
// non-zero. If `timeout` seconds pass, an error will be printed and the stream
// will unblock.
__global__ void block_stream(const volatile nvbench::int32_t *flag,
                             volatile nvbench::int32_t *timeout_flag,
                             nvbench::float64_t timeout)
{
  const auto start_point = cuda::std::chrono::high_resolution_clock::now();
  const auto timeout_ns =
    cuda::std::chrono::nanoseconds(static_cast<nvbench::int64_t>(timeout * 1e9));
  const auto timeout_point = start_point + timeout_ns;

  const bool use_timeout = timeout >= 0.;
  auto now               = cuda::std::chrono::high_resolution_clock::now();
  while (!(*flag) && (!use_timeout || now < timeout_point))
  {
    now = cuda::std::chrono::high_resolution_clock::now();
  }

  if (now >= timeout_point)
  {
    *timeout_flag = 1;
    __threadfence_system(); // Ensure timeout flag visibility on host.
    printf("\n"
           "######################################################################\n"
           "##################### Possible Deadlock Detected #####################\n"
           "######################################################################\n"
           "\n"
           "Forcing unblock: The current measurement appears to have deadlocked\n"
           "and the results cannot be trusted.\n"
           "\n"
           "This happens when the KernelLauncher synchronizes the CUDA device.\n"
           "If this is the case, pass the `sync` exec_tag to the `exec` call:\n"
           "\n"
           "    state.exec(<KernelLauncher>); // Deadlock\n"
           "    state.exec(nvbench::exec_tag::sync, <KernelLauncher>); // Safe\n"
           "\n"
           "This tells NVBench about the sync so it can run the benchmark safely.\n"
           "\n"
           "If the KernelLauncher does not synchronize but has a very long \n"
           "execution time, this may be a false positive. If so, disable this\n"
           "check with:\n"
           "\n"
           "    state.set_blocking_kernel_timeout(-1);\n"
           "\n"
           "The current timeout is set to %0.5g seconds.\n"
           "\n"
           "For more information, see the 'Benchmarks that sync' section of the\n"
           "NVBench documentation.\n"
           "\n"
           "If this happens while profiling with an external tool,\n"
           "pass the `--disable-blocking-kernel` flag or the `--profile` flag\n"
           "(to also only run the benchmark once) to the executable.\n"
           "\n"
           "For more information, see the 'Benchmark Properties' section of the\n"
           "NVBench documentation.\n\n",
           timeout);
  }
}

} // namespace

namespace nvbench
{

blocking_kernel::blocking_kernel()
{
  NVBENCH_CUDA_CALL(cudaHostRegister(&m_host_flag, sizeof(m_host_flag), cudaHostRegisterMapped));
  NVBENCH_CUDA_CALL(cudaHostGetDevicePointer(&m_device_flag, &m_host_flag, 0));
  NVBENCH_CUDA_CALL(
    cudaHostRegister(&m_host_timeout_flag, sizeof(m_host_timeout_flag), cudaHostRegisterMapped));
  NVBENCH_CUDA_CALL(cudaHostGetDevicePointer(&m_device_timeout_flag, &m_host_timeout_flag, 0));
}

blocking_kernel::~blocking_kernel()
{
  NVBENCH_CUDA_CALL_NOEXCEPT(cudaHostUnregister(&m_host_flag));
  NVBENCH_CUDA_CALL_NOEXCEPT(cudaHostUnregister(&m_host_timeout_flag));
}

void blocking_kernel::block(const nvbench::cuda_stream &stream, nvbench::float64_t timeout)
{
  m_host_flag         = 0;
  m_host_timeout_flag = 0;
  block_stream<<<1, 1, 0, stream>>>(m_device_flag, m_device_timeout_flag, timeout);
}

void blocking_kernel::timeout_detected()
{
  NVBENCH_THROW(std::runtime_error,
                "{}",
                "Deadlock detected -- missing nvbench::exec_tag::sync? "
                "See stdout for details.");
}

} // namespace nvbench
