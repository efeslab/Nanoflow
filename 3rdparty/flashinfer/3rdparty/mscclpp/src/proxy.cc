// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <atomic>
#include <mscclpp/core.hpp>
#include <mscclpp/gpu_utils.hpp>
#include <mscclpp/proxy.hpp>
#include <mscclpp/utils.hpp>
#include <thread>

#include "api.h"

namespace mscclpp {

const int ProxyStopCheckPeriod = 1000;

// Unless explicitly requested, a flush of the tail to device memory is triggered for every ProxyFlushPeriod.
// As long as the FIFO size is large enough, having a stale tail is not a problem.
const int ProxyFlushPeriod = 4;

struct Proxy::Impl {
  ProxyHandler handler;
  std::function<void()> threadInit;
  Fifo fifo;
  std::thread service;
  std::atomic_bool running;

  Impl(ProxyHandler handler, std::function<void()> threadInit, size_t fifoSize)
      : handler(handler), threadInit(threadInit), fifo(fifoSize), running(false) {}
};

MSCCLPP_API_CPP Proxy::Proxy(ProxyHandler handler, std::function<void()> threadInit, size_t fifoSize) {
  pimpl = std::make_unique<Impl>(handler, threadInit, fifoSize);
}

MSCCLPP_API_CPP Proxy::Proxy(ProxyHandler handler, size_t fifoSize)
    : Proxy(
          handler, [] {}, fifoSize) {}

MSCCLPP_API_CPP Proxy::~Proxy() {
  if (pimpl) {
    stop();
  }
}

MSCCLPP_API_CPP void Proxy::start() {
  int cudaDevice;
  MSCCLPP_CUDATHROW(cudaGetDevice(&cudaDevice));

  pimpl->running = true;
  pimpl->service = std::thread([this, cudaDevice] {
    MSCCLPP_CUDATHROW(cudaSetDevice(cudaDevice));

    pimpl->threadInit();

    ProxyHandler handler = this->pimpl->handler;
    Fifo& fifo = this->pimpl->fifo;
    std::atomic_bool& running = this->pimpl->running;
    ProxyTrigger trigger;

    int flushPeriod = std::min(fifo.size(), ProxyFlushPeriod);

    int runCnt = ProxyStopCheckPeriod;
    uint64_t flushCnt = 0;
    for (;;) {
      if (runCnt-- == 0) {
        runCnt = ProxyStopCheckPeriod;
        if (!running) {
          break;
        }
      }
      // Poll to see if we are ready to send anything
      trigger = fifo.poll();
      if (trigger.fst == 0 || trigger.snd == 0) {  // TODO: this check is a potential pitfall for custom triggers
        continue;                                  // there is one in progress
      }
      trigger.snd ^= ((uint64_t)1 << (uint64_t)63);  // this is where the last bit of snd is reverted.

      ProxyHandlerResult result = handler(trigger);

      // Send completion: reset only the high 64 bits
      fifo.pop();
      // Flush the tail to device memory. This is either triggered every flushPeriod to make sure that the fifo can make
      // progress even if there is no request mscclppSync. However, mscclppSync type is for flush request.
      if ((++flushCnt % flushPeriod) == 0 || result == ProxyHandlerResult::FlushFifoTailAndContinue) {
        // TODO: relocate this check: || (trigger.fields.type & mscclppSync)
        fifo.flushTail();
      }

      if (result == ProxyHandlerResult::Stop) {
        break;
      }
    }

    // make sure the tail is flushed before we shut the proxy
    fifo.flushTail(/*sync=*/true);
    // TODO: do these need to run?
    // bool isP2pProxy = (proxyState->ibContext == nullptr);
    // if (isP2pProxy) {
    //   cudaStream_t p2pStream = proxyState->p2pStream;
    //   PROXYCUDACHECK(cudaStreamSynchronize(p2pStream));
    // }
  });
}

MSCCLPP_API_CPP void Proxy::stop() {
  pimpl->running = false;
  if (pimpl->service.joinable()) {
    pimpl->service.join();
  }
}

MSCCLPP_API_CPP Fifo& Proxy::fifo() { return pimpl->fifo; }

}  // namespace mscclpp
