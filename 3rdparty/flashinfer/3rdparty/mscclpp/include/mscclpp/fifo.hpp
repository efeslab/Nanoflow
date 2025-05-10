// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_FIFO_HPP_
#define MSCCLPP_FIFO_HPP_

#include <cstdint>
#include <functional>
#include <memory>

#include "fifo_device.hpp"

namespace mscclpp {

constexpr size_t DEFAULT_FIFO_SIZE = 128;

/// A class representing a host proxy FIFO that can consume work elements pushed by device threads.
class Fifo {
 public:
  /// Constructs a new @ref Fifo object.
  /// @param size The number of entires in the FIFO.
  Fifo(int size = DEFAULT_FIFO_SIZE);

  /// Destroys the @ref Fifo object.
  ~Fifo();

  /// Polls the FIFO for a trigger.
  ///
  /// Returns @ref ProxyTrigger which is the trigger at the head of fifo.
  ProxyTrigger poll();

  /// Pops a trigger from the FIFO.
  void pop();

  /// Flushes the tail of the FIFO.
  ///
  /// @param sync If true, waits for the flush to complete before returning.
  void flushTail(bool sync = false);

  /// Return the FIFO size.
  /// @return The FIFO size.
  int size() const;

  /// Returns a @ref FifoDeviceHandle object representing the device FIFO.
  ///
  /// @return A @ref FifoDeviceHandle object representing the device FIFO.
  FifoDeviceHandle deviceHandle();

 private:
  struct Impl;
  std::unique_ptr<Impl> pimpl;
};

}  // namespace mscclpp

#endif  // MSCCLPP_FIFO_HPP_
