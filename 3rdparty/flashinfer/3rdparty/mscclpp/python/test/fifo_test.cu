// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <mscclpp/fifo_device.hpp>

extern "C" __global__ void __launch_bounds__(1024, 1) fifo(mscclpp::FifoDeviceHandle fifo) {
  mscclpp::ProxyTrigger trigger;
  trigger.fst = 123;
  fifo.push(trigger);
}
