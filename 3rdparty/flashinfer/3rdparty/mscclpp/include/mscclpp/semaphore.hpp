// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_SEMAPHORE_HPP_
#define MSCCLPP_SEMAPHORE_HPP_

#include <memory>

#include "core.hpp"
#include "gpu_utils.hpp"
#include "semaphore_device.hpp"

namespace mscclpp {

/// A base class for semaphores.
///
/// An semaphore is a synchronization mechanism that allows the local peer to wait for the remote peer to complete a
/// data transfer. The local peer signals the remote peer that it has completed a data transfer by incrementing the
/// outbound semaphore ID. The incremented outbound semaphore ID is copied to the remote peer's inbound semaphore ID so
/// that the remote peer can wait for the local peer to complete a data transfer. Vice versa, the remote peer signals
/// the local peer that it has completed a data transfer by incrementing the remote peer's outbound semaphore ID and
/// copying the incremented value to the local peer's inbound semaphore ID.
///
/// @tparam InboundDeleter The deleter for inbound semaphore IDs. This is either `std::default_delete` for host memory
/// or @ref CudaDeleter for device memory.
/// @tparam OutboundDeleter The deleter for outbound semaphore IDs. This is either `std::default_delete` for host memory
/// or @ref CudaDeleter for device memory.
///
template <template <typename> typename InboundDeleter, template <typename> typename OutboundDeleter>
class BaseSemaphore {
 protected:
  /// The registered memory for the remote peer's inbound semaphore ID.
  NonblockingFuture<RegisteredMemory> remoteInboundSemaphoreIdsRegMem_;

  /// The inbound semaphore ID that is incremented by the remote peer and waited on by the local peer.
  ///
  /// The location of @ref localInboundSemaphore_ can be either on the host or on the device.
  std::unique_ptr<uint64_t, InboundDeleter<uint64_t>> localInboundSemaphore_;

  /// The expected inbound semaphore ID to be incremented by the local peer and compared to the
  /// @ref localInboundSemaphore_.
  ///
  /// The location of @ref expectedInboundSemaphore_ can be either on the host or on the device.
  std::unique_ptr<uint64_t, InboundDeleter<uint64_t>> expectedInboundSemaphore_;

  /// The outbound semaphore ID that is incremented by the local peer and copied to the remote peer's @ref
  /// localInboundSemaphore_.
  ///
  /// The location of @ref outboundSemaphore_ can be either on the host or on the device.
  std::unique_ptr<uint64_t, OutboundDeleter<uint64_t>> outboundSemaphore_;

 public:
  /// Constructs a BaseSemaphore.
  ///
  /// @param localInboundSemaphoreId The inbound semaphore ID
  /// @param expectedInboundSemaphoreId The expected inbound semaphore ID
  /// @param outboundSemaphoreId The outbound semaphore ID
  BaseSemaphore(std::unique_ptr<uint64_t, InboundDeleter<uint64_t>> localInboundSemaphoreId,
                std::unique_ptr<uint64_t, InboundDeleter<uint64_t>> expectedInboundSemaphoreId,
                std::unique_ptr<uint64_t, OutboundDeleter<uint64_t>> outboundSemaphoreId)
      : localInboundSemaphore_(std::move(localInboundSemaphoreId)),
        expectedInboundSemaphore_(std::move(expectedInboundSemaphoreId)),
        outboundSemaphore_(std::move(outboundSemaphoreId)) {}
};

/// A semaphore for sending signals from the host to the device.
class Host2DeviceSemaphore : public BaseSemaphore<CudaDeleter, std::default_delete> {
 private:
  std::shared_ptr<Connection> connection_;

 public:
  /// Constructor.
  /// @param communicator The communicator.
  /// @param connection The connection associated with this semaphore.
  Host2DeviceSemaphore(Communicator& communicator, std::shared_ptr<Connection> connection);

  /// Returns the connection.
  /// @return The connection associated with this semaphore.
  std::shared_ptr<Connection> connection();

  /// Signal the device.
  void signal();

  /// Device-side handle for @ref Host2DeviceSemaphore.
  using DeviceHandle = Host2DeviceSemaphoreDeviceHandle;

  /// Returns the device-side handle.
  DeviceHandle deviceHandle();
};

/// A semaphore for sending signals from the local host to a remote host.
class Host2HostSemaphore : public BaseSemaphore<std::default_delete, std::default_delete> {
 public:
  /// Constructor
  /// @param communicator The communicator.
  /// @param connection The connection associated with this semaphore. @ref Transport::CudaIpc is not allowed for @ref
  /// Host2HostSemaphore.
  Host2HostSemaphore(Communicator& communicator, std::shared_ptr<Connection> connection);

  /// Returns the connection.
  /// @return The connection associated with this semaphore.
  std::shared_ptr<Connection> connection();

  /// Signal the remote host.
  void signal();

  /// Check if the remote host has signaled.
  /// @return true if the remote host has signaled.
  bool poll();

  /// Wait for the remote host to signal.
  /// @param maxSpinCount The maximum number of spin counts before throwing an exception. Never throws if negative.
  void wait(int64_t maxSpinCount = 10000000);

 private:
  std::shared_ptr<Connection> connection_;
};

/// A semaphore for sending signals from the local device to a peer device via SM.
class SmDevice2DeviceSemaphore : public BaseSemaphore<CudaDeleter, CudaDeleter> {
 public:
  /// Constructor.
  /// @param communicator The communicator.
  /// @param connection The connection associated with this semaphore.
  SmDevice2DeviceSemaphore(Communicator& communicator, std::shared_ptr<Connection> connection);

  /// Constructor.
  SmDevice2DeviceSemaphore() = delete;

  /// Device-side handle for @ref SmDevice2DeviceSemaphore.
  using DeviceHandle = SmDevice2DeviceSemaphoreDeviceHandle;

  /// Returns the device-side handle.
  DeviceHandle deviceHandle() const;

  bool isRemoteInboundSemaphoreIdSet_;
};

}  // namespace mscclpp

#endif  // MSCCLPP_SEMAPHORE_HPP_
