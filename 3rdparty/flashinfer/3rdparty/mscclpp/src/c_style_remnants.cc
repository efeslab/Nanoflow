// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "api.h"
#include "debug.h"

MSCCLPP_API void mscclppDefaultLogHandler(const char* msg) { mscclppDebugDefaultLogHandler(msg); }

MSCCLPP_API mscclppResult_t mscclppSetLogHandler(mscclppLogHandler_t handler) {
  return mscclppDebugSetLogHandler(handler);
}

MSCCLPP_API const char* mscclppGetErrorString(mscclppResult_t code) {
  switch (code) {
    case mscclppSuccess:
      return "no error";
    case mscclppUnhandledCudaError:
      return "unhandled cuda error";
    case mscclppSystemError:
      return "unhandled system error";
    case mscclppInternalError:
      return "internal error";
    case mscclppInvalidArgument:
      return "invalid argument";
    case mscclppInvalidUsage:
      return "invalid usage";
    case mscclppRemoteError:
      return "remote process exited or there was a network error";
    case mscclppInProgress:
      return "MSCCL++ operation in progress";
    default:
      return "unknown result code";
  }
}
