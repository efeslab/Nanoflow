// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_PROXY_HPP_
#define MSCCLPP_PROXY_HPP_

#include <functional>
#include <memory>

#include "fifo.hpp"

namespace mscclpp {

enum class ProxyHandlerResult {
  Continue,
  FlushFifoTailAndContinue,
  Stop,
};

class Proxy;
using ProxyHandler = std::function<ProxyHandlerResult(ProxyTrigger)>;

class Proxy {
 public:
  Proxy(ProxyHandler handler, std::function<void()> threadInit, size_t fifoSize = DEFAULT_FIFO_SIZE);
  Proxy(ProxyHandler handler, size_t fifoSize = DEFAULT_FIFO_SIZE);
  ~Proxy();

  void start();
  void stop();

  /// This is a concurrent fifo which is multiple threads from the device
  /// can produce for and the sole proxy thread consumes it.
  /// @return the fifo
  Fifo& fifo();

 private:
  struct Impl;
  std::unique_ptr<Impl> pimpl;
};

}  // namespace mscclpp

#endif  // MSCCLPP_PROXY_HPP_
