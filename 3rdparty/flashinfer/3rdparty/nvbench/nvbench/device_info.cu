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

#include <nvbench/device_info.cuh>

#include <nvbench/config.cuh>
#include <nvbench/cuda_call.cuh>
#include <nvbench/detail/device_scope.cuh>
#include <nvbench/internal/nvml.cuh>

#include <cuda_runtime_api.h>

#define UNUSED(x) (void)(x)

namespace nvbench
{

device_info::memory_info device_info::get_global_memory_usage() const
{
  nvbench::detail::device_scope _{m_id};

  memory_info result{};
  NVBENCH_CUDA_CALL(cudaMemGetInfo(&result.bytes_free, &result.bytes_total));
  return result;
}

device_info::device_info(int id)
    : m_id{id}
    , m_prop{}
    , m_nvml_device(nullptr)
{
  NVBENCH_CUDA_CALL(cudaGetDeviceProperties(&m_prop, m_id));
  // NVML's lifetime should extend for the entirety of the process, so store in a
  // global.
  [[maybe_unused]] static auto nvml_lifetime = nvbench::nvml::NVMLLifetimeManager();

#ifdef NVBENCH_HAS_NVML
  // Retrieve the current device's pci_id as a null-terminated string.
  // Docs say 13 chars should always be sufficient.
  constexpr int pci_id_len = 13;
  char pci_id[pci_id_len];
  NVBENCH_CUDA_CALL(cudaDeviceGetPCIBusId(pci_id, pci_id_len, m_id));
  NVBENCH_NVML_CALL(nvmlDeviceGetHandleByPciBusId(pci_id, &m_nvml_device));
#endif // NVBENCH_HAS_NVML
}

void device_info::set_persistence_mode(bool state)
#ifndef NVBENCH_HAS_NVML
{
  UNUSED(state);
  throw nvbench::nvml::not_enabled{};
}
#else  // NVBENCH_HAS_NVML
try
{
  NVBENCH_NVML_CALL(
    nvmlDeviceSetPersistenceMode(m_nvml_device,
                                 state ? NVML_FEATURE_ENABLED : NVML_FEATURE_DISABLED));
}
catch (nvml::call_failed &e)
{
  if (e.get_error_code() == NVML_ERROR_NOT_SUPPORTED)
  {
    NVBENCH_THROW(std::runtime_error, "{}", "Persistence mode is only supported on Linux.");
  }
  else if (e.get_error_code() == NVML_ERROR_NO_PERMISSION)
  {
    NVBENCH_THROW(std::runtime_error,
                  "{}",
                  "Root/Admin permissions required to set persistence mode.");
  }

  throw;
}
#endif // NVBENCH_HAS_NVML

void device_info::lock_gpu_clocks(device_info::clock_rate rate)
#ifndef NVBENCH_HAS_NVML
{
  UNUSED(rate);
  throw nvbench::nvml::not_enabled{};
}
#else  // NVBENCH_HAS_NVML
try
{
  switch (rate)
  {
    case clock_rate::none:
      NVBENCH_NVML_CALL(nvmlDeviceResetGpuLockedClocks(m_nvml_device));
      break;

    case clock_rate::base:
      NVBENCH_NVML_CALL(
        nvmlDeviceSetGpuLockedClocks(m_nvml_device,
                                     static_cast<unsigned int>(NVML_CLOCK_LIMIT_ID_TDP),
                                     static_cast<unsigned int>(NVML_CLOCK_LIMIT_ID_TDP)));
      break;

    case clock_rate::maximum: {
      const auto max_mhz =
        static_cast<unsigned int>(this->get_sm_default_clock_rate() / (1000 * 1000));
      NVBENCH_NVML_CALL(nvmlDeviceSetGpuLockedClocks(m_nvml_device, max_mhz, max_mhz));
      break;
    }

    default:
      NVBENCH_THROW(std::runtime_error, "Unrecognized clock rate: {}", static_cast<int>(rate));
  }
}
catch (nvml::call_failed &e)
{
  if (e.get_error_code() == NVML_ERROR_NOT_SUPPORTED && this->get_sm_version() < 700)
  {
    NVBENCH_THROW(std::runtime_error,
                  "GPU clock rates can only be modified for Volta and later. "
                  "Device: {} ({}) SM: {} < {}. For older cards, look up the "
                  "desired frequency for your card and lock clocks manually "
                  "with `nvidia-smi -lgc <freq_MHz>,<freq_MHz>` (or -rgc to "
                  "unlock).",
                  this->get_name(),
                  this->get_id(),
                  this->get_sm_version(),
                  700);
  }
  else if (e.get_error_code() == NVML_ERROR_NO_PERMISSION)
  {
    NVBENCH_THROW(std::runtime_error,
                  "{}",
                  "Root/Admin permissions required to change GPU clock rates.");
  }

  throw;
}
#endif // NVBENCH_HAS_NVML

#ifdef NVBENCH_HAS_CUPTI
[[nodiscard]] CUcontext device_info::get_context() const
{
  if (!is_active())
  {
    NVBENCH_THROW(std::runtime_error, "{}", "get_context is called for inactive device");
  }

  CUcontext cu_context;
  NVBENCH_DRIVER_API_CALL(cuCtxGetCurrent(&cu_context));
  return cu_context;
}
#endif

} // namespace nvbench
