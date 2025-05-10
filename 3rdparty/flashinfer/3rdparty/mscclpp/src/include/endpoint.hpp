// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_ENDPOINT_HPP_
#define MSCCLPP_ENDPOINT_HPP_

#include <mscclpp/core.hpp>
#include <vector>

#include "ib.hpp"
#include "socket.h"

#define MAX_IF_NAME_SIZE 16

namespace mscclpp {

struct Endpoint::Impl {
  Impl(EndpointConfig config, Context::Impl& contextImpl);
  Impl(const std::vector<char>& serialization);

  Transport transport_;
  uint64_t hostHash_;

  // The following are only used for IB and are undefined for other transports.
  bool ibLocal_;
  IbQp* ibQp_;
  IbQpInfo ibQpInfo_;

  // The following are only used for Ethernet and are undefined for other transports.
  std::unique_ptr<Socket> socket_;
  SocketAddress socketAddress_;
  volatile uint32_t* abortFlag_;
  char netIfName_[MAX_IF_NAME_SIZE + 1];
};

}  // namespace mscclpp

#endif  // MSCCLPP_ENDPOINT_HPP_
