// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_SM_CHANNEL_HPP_
#define MSCCLPP_SM_CHANNEL_HPP_

#include <type_traits>

#include "core.hpp"
#include "semaphore.hpp"
#include "sm_channel_device.hpp"

namespace mscclpp {

/// Channel for accessing peer memory directly from SM.
struct SmChannel {
 private:
  std::shared_ptr<SmDevice2DeviceSemaphore> semaphore_;
  RegisteredMemory dst_;
  void* src_;
  void* getPacketBuffer_;

 public:
  /// Constructor.
  SmChannel() = default;

  /// Constructor.
  /// @param semaphore The semaphore used to synchronize the communication.
  /// @param dst Registered memory of the destination.
  /// @param src The source memory address.
  /// @param getPacketBuffer The optional buffer used for @ref getPackets().
  SmChannel(std::shared_ptr<SmDevice2DeviceSemaphore> semaphore, RegisteredMemory dst, void* src,
            void* getPacketBuffer = nullptr);

  /// Device-side handle for @ref SmChannel.
  using DeviceHandle = SmChannelDeviceHandle;

  /// Returns the device-side handle.
  ///
  /// User should make sure the SmChannel is not released when using the returned handle.
  ///
  DeviceHandle deviceHandle() const;
};

}  // namespace mscclpp

#endif  // MSCCLPP_SM_CHANNEL_HPP_
