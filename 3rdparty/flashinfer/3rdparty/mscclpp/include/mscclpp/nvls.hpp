// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_NVLS_HPP_
#define MSCCLPP_NVLS_HPP_

#include <mscclpp/gpu_utils.hpp>
#include <mscclpp/nvls_device.hpp>

namespace mscclpp {

class NvlsConnection {
 public:
  NvlsConnection(size_t bufferSize, int numDevices);
  NvlsConnection(const std::vector<char>& data);
  NvlsConnection() = delete;
  std::vector<char> serialize();

  // the recommended buffer size for NVLS, returned by cuMulticastGetGranularity
  static const int DefaultNvlsBufferSize = (1 << 29);

  // Everyone needs to synchronize after creating a NVLS connection before adding devices
  void addDevice();
  void addDevice(int cudaDeviceId);

  struct DeviceMulticastPointer {
   private:
    std::shared_ptr<PhysicalCudaMemory<char>> deviceMem_;
    std::shared_ptr<char> mcPtr_;
    size_t bufferSize_;

   public:
    using DeviceHandle = DeviceMulticastPointerDeviceHandle;
    DeviceMulticastPointer(std::shared_ptr<PhysicalCudaMemory<char>> deviceMem, std::shared_ptr<char> mcPtr,
                           size_t bufferSize)
        : deviceMem_(deviceMem), mcPtr_(mcPtr), bufferSize_(bufferSize) {}
    DeviceHandle deviceHandle();
    char* getDevicePtr();

    friend class NvlsConnection;
  };

  std::shared_ptr<DeviceMulticastPointer> allocateAndBindCuda(size_t size);

  /// The \p handle to the allocation (its lifetime is managed by the caller)
  /// and the \p size of the allocation.
  std::shared_ptr<char> bindAllocatedCuda(CUmemGenericAllocationHandle memHandle, size_t size);

  size_t getMultiCastMinGranularity();

 private:
  class Impl;
  std::shared_ptr<Impl> pimpl_;
};

class Communicator;

/// Connect to NVLS on setup.
///
/// This function used to connect to NVLS on setup. NVLS collective using multicast operations to send/recv data.
/// Here we need to put all involved ranks into the collective group.
///
/// @param comm The communicator.
/// @param allRanks The ranks of all processes involved in the collective.
/// @param config The configuration for the local endpoint.
/// @return std::shared_ptr<NvlsConnection> A shared pointer to the NVLS connection.
std::shared_ptr<NvlsConnection> connectNvlsCollective(std::shared_ptr<Communicator> comm, std::vector<int> allRanks,
                                                      size_t bufferSize = NvlsConnection::DefaultNvlsBufferSize);

}  // namespace mscclpp

#endif  // MSCCLPP_NVLS_HPP_
