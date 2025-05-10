// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_CONTEXT_HPP_
#define MSCCLPP_CONTEXT_HPP_

#include <mscclpp/core.hpp>
#include <mscclpp/gpu_utils.hpp>
#include <unordered_map>
#include <vector>

#include "ib.hpp"

namespace mscclpp {

struct Context::Impl {
  std::vector<std::shared_ptr<Connection>> connections_;
  std::unordered_map<Transport, std::unique_ptr<IbCtx>> ibContexts_;
  CudaStreamWithFlags ipcStream_;
  CUmemGenericAllocationHandle mcHandle_;

  Impl();

  IbCtx* getIbContext(Transport ibTransport);
};

}  // namespace mscclpp

#endif  // MSCCLPP_CONTEXT_HPP_
