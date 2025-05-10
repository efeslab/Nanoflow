// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_COMMUNICATOR_HPP_
#define MSCCLPP_COMMUNICATOR_HPP_

#include <memory>
#include <mscclpp/core.hpp>
#include <unordered_map>
#include <vector>

namespace mscclpp {

class ConnectionBase;

struct ConnectionInfo {
  int remoteRank;
  int tag;
};

struct Communicator::Impl {
  std::shared_ptr<Bootstrap> bootstrap_;
  std::shared_ptr<Context> context_;
  std::unordered_map<const Connection*, ConnectionInfo> connectionInfos_;
  std::vector<std::shared_ptr<Setuppable>> toSetup_;

  Impl(std::shared_ptr<Bootstrap> bootstrap, std::shared_ptr<Context> context);

  struct Connector;
};

}  // namespace mscclpp

#endif  // MSCCLPP_COMMUNICATOR_HPP_
