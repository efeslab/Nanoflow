// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <mpi.h>

#include <mscclpp/gpu_utils.hpp>

#include "infiniband/verbs.h"
#include "mp_unit_tests.hpp"

void IbTestBase::SetUp() {
  MSCCLPP_CUDATHROW(cudaGetDeviceCount(&cudaDevNum));
  cudaDevId = (gEnv->rank % gEnv->nRanksPerNode) % cudaDevNum;
  MSCCLPP_CUDATHROW(cudaSetDevice(cudaDevId));

  int ibDevId = (gEnv->rank % gEnv->nRanksPerNode) / mscclpp::getIBDeviceCount();
  ibDevName = mscclpp::getIBDeviceName(ibIdToTransport(ibDevId));
}

void IbPeerToPeerTest::SetUp() {
  IbTestBase::SetUp();

  mscclpp::UniqueId id;

  if (gEnv->rank < 2) {
    // This test needs only two ranks
    bootstrap = std::make_shared<mscclpp::TcpBootstrap>(gEnv->rank, 2);
    if (bootstrap->getRank() == 0) id = bootstrap->createUniqueId();
  }
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
  if (gEnv->rank >= 2) {
    // This test needs only two ranks
    return;
  }

  bootstrap->initialize(id);

  ibCtx = std::make_shared<mscclpp::IbCtx>(ibDevName);
  qp = ibCtx->createQp(1024, 1, 8192, 0, 64);

  qpInfo[gEnv->rank] = qp->getInfo();
  bootstrap->allGather(qpInfo.data(), sizeof(mscclpp::IbQpInfo));
}

void IbPeerToPeerTest::registerBufferAndConnect(void* buf, size_t size) {
  bufSize = size;
  mr = ibCtx->registerMr(buf, size);
  mrInfo[gEnv->rank] = mr->getInfo();
  bootstrap->allGather(mrInfo.data(), sizeof(mscclpp::IbMrInfo));

  for (int i = 0; i < bootstrap->getNranks(); ++i) {
    if (i == gEnv->rank) continue;
    qp->rtr(qpInfo[i]);
    qp->rts();
    break;
  }
  bootstrap->barrier();
}

void IbPeerToPeerTest::stageSend(uint32_t size, uint64_t wrId, uint64_t srcOffset, uint64_t dstOffset, bool signaled) {
  const mscclpp::IbMrInfo& remoteMrInfo = mrInfo[(gEnv->rank == 1) ? 0 : 1];
  qp->stageSend(mr, remoteMrInfo, size, wrId, srcOffset, dstOffset, signaled);
}

void IbPeerToPeerTest::stageAtomicAdd(uint64_t wrId, uint64_t dstOffset, uint64_t addVal, bool signaled) {
  const mscclpp::IbMrInfo& remoteMrInfo = mrInfo[(gEnv->rank == 1) ? 0 : 1];
  qp->stageAtomicAdd(mr, remoteMrInfo, wrId, dstOffset, addVal, signaled);
}

void IbPeerToPeerTest::stageSendWithImm(uint32_t size, uint64_t wrId, uint64_t srcOffset, uint64_t dstOffset,
                                        bool signaled, unsigned int immData) {
  const mscclpp::IbMrInfo& remoteMrInfo = mrInfo[(gEnv->rank == 1) ? 0 : 1];
  qp->stageSendWithImm(mr, remoteMrInfo, size, wrId, srcOffset, dstOffset, signaled, immData);
}

TEST_F(IbPeerToPeerTest, SimpleSendRecv) {
  if (gEnv->rank >= 2) {
    // This test needs only two ranks
    return;
  }

  mscclpp::Timer timeout(3);

  const int maxIter = 100000;
  const int nelem = 1;
  auto data = mscclpp::allocUniqueCuda<int>(nelem);

  registerBufferAndConnect(data.get(), sizeof(int) * nelem);

  if (gEnv->rank == 1) {
    mscclpp::Timer timer;
    for (int iter = 0; iter < maxIter; ++iter) {
      stageSend(sizeof(int) * nelem, 0, 0, 0, true);
      qp->postSend();
      bool waiting = true;
      int spin = 0;
      while (waiting) {
        int wcNum = qp->pollCq();
        ASSERT_GE(wcNum, 0);
        for (int i = 0; i < wcNum; ++i) {
          const ibv_wc* wc = qp->getWc(i);
          EXPECT_EQ(wc->status, IBV_WC_SUCCESS);
          waiting = false;
          break;
        }
        if (spin++ > 1000000) {
          FAIL() << "Polling is stuck.";
        }
      }
    }
    float us = (float)timer.elapsed();
    std::cout << "IbPeerToPeerTest.SimpleSendRecv: " << us / maxIter << " us/iter" << std::endl;
  }
  bootstrap->barrier();
}

__global__ void kernelMemoryConsistency(uint64_t* data, volatile uint64_t* curIter, volatile int* result,
                                        uint64_t nelem, uint64_t maxIter) {
  if (blockIdx.x != 0) return;

  __shared__ int errs[1024];

  constexpr int FlagWrong = 1;
  constexpr int FlagAbort = 2;

  volatile uint64_t* ptr = data;
  for (uint64_t iter = 1; iter < maxIter + 1; ++iter) {
    int err = 0;

    if (threadIdx.x == 0) {
      *curIter = iter;

      // Wait for the first element arrival (expect equal to iter). Expect that the first element is delivered in
      // a special way that guarantees all other elements are completely delivered.
      uint64_t spin = 0;
      while (ptr[0] != iter) {
        if (spin++ == 1000000) {
          // Assume the program is stuck. Set the abort flag and escape the loop.
          *result |= FlagAbort;
          err = 1;
          break;
        }
      }
    }
    __syncthreads();

    // Check results (expect equal to iter) in backward that is more likely to see the wrong result.
    for (size_t i = nelem - 1 + threadIdx.x; i >= blockDim.x; i -= blockDim.x) {
      if (data[i - blockDim.x] != iter) {
#if 1
        *result |= FlagWrong;
        err = 1;
        break;
#else
        // For debugging purposes: try waiting for the correct result.
        uint64_t spin = 0;
        while (ptr[i - blockDim.x] != iter) {
          if (spin++ == 1000000) {
            *result |= FlagAbort;
            err = 1;
            break;
          }
        }
        if (spin >= 1000000) {
          break;
        }
#endif
      }
    }

    errs[threadIdx.x] = err;
    __threadfence();
    __syncthreads();

    // Check if any error is detected.
    int total_err = 0;
    for (size_t i = 0; i < blockDim.x; ++i) {
      total_err += errs[i];
    }

    if (total_err > 0) {
      // Exit if any error is detected.
      return;
    }
  }
  if (threadIdx.x == 0) {
    *curIter = maxIter + 1;
  }
}

TEST_F(IbPeerToPeerTest, MemoryConsistency) {
  if (gEnv->rank >= 2) {
    // This test needs only two ranks
    return;
  }

  const uint64_t signalPeriod = 1024;
  const uint64_t maxIter = 10000;
  const uint64_t nelem = 65536 + 1;
  auto data = mscclpp::allocUniqueCuda<uint64_t>(nelem);

  registerBufferAndConnect(data.get(), sizeof(uint64_t) * nelem);

  uint64_t res = 0;
  uint64_t iter = 0;

  if (gEnv->rank == 0) {
    // Receiver
    auto curIter = mscclpp::makeUniqueCudaHost<uint64_t>(0);
    auto result = mscclpp::makeUniqueCudaHost<int>(0);

    volatile uint64_t* ptrCurIter = (volatile uint64_t*)curIter.get();
    volatile int* ptrResult = (volatile int*)result.get();

    ASSERT_EQ(*ptrCurIter, 0);
    ASSERT_EQ(*ptrResult, 0);

    kernelMemoryConsistency<<<1, 1024>>>(data.get(), ptrCurIter, ptrResult, nelem, maxIter);
    MSCCLPP_CUDATHROW(cudaGetLastError());

    for (iter = 1; iter < maxIter + 1; ++iter) {
      mscclpp::Timer timeout(5);

      while (*ptrCurIter != iter + 1) {
        res = *ptrResult;
        if (res != 0) break;
      }

      // Send the result to the sender
      res = *ptrResult;
      uint64_t tmp[2];
      tmp[0] = res;
      bootstrap->allGather(tmp, sizeof(uint64_t));

      if (res != 0) break;
    }

    MSCCLPP_CUDATHROW(cudaDeviceSynchronize());
  } else if (gEnv->rank == 1) {
    // Sender
    std::vector<uint64_t> hostBuffer(nelem, 0);

    for (iter = 1; iter < maxIter + 1; ++iter) {
      mscclpp::Timer timeout(5);

      // Set data
      for (uint64_t i = 0; i < nelem; i++) {
        hostBuffer[i] = iter;
      }
      mscclpp::memcpyCuda<uint64_t>(data.get(), hostBuffer.data(), nelem, cudaMemcpyHostToDevice);

      // Need to signal from time to time to empty the IB send queue
      bool signaled = (iter % signalPeriod == 0);

      // Send from the second element to the last
      stageSend(sizeof(uint64_t) * (nelem - 1), 0, sizeof(uint64_t), sizeof(uint64_t), signaled);
      qp->postSend();

#if 0
      // Send the first element using a normal send. This should occasionally see the wrong result.
      stageSend(sizeof(uint64_t), 0, 0, 0, false);
      qp->postSend();
#else
      // For reference: send the first element using AtomicAdd. This should see the correct result.
      stageAtomicAdd(0, 0, 1, false);
      qp->postSend();
#endif

      if (signaled) {
        int wcNum = qp->pollCq();
        while (wcNum == 0) {
          wcNum = qp->pollCq();
        }
        ASSERT_EQ(wcNum, 1);
        const ibv_wc* wc = qp->getWc(0);
        ASSERT_EQ(wc->status, IBV_WC_SUCCESS);
      }

      // Get the result from the receiver
      uint64_t tmp[2];
      bootstrap->allGather(tmp, sizeof(uint64_t));
      res = tmp[0];

      if (res != 0) break;
    }
  }

  if (res & 2) {
    FAIL() << "The receiver is stuck at iteration " << iter << ".";
  } else if (res != 0 && res != 1) {
    FAIL() << "Unknown error is detected at iteration " << iter << ". res =" << res;
  }

  EXPECT_EQ(res, 0);
}

TEST_F(IbPeerToPeerTest, SimpleAtomicAdd) {
  if (gEnv->rank >= 2) {
    // This test needs only two ranks
    return;
  }

  mscclpp::Timer timeout(3);

  const int maxIter = 100000;
  const int nelem = 1;
  auto data = mscclpp::allocUniqueCuda<int>(nelem);

  registerBufferAndConnect(data.get(), sizeof(int) * nelem);

  if (gEnv->rank == 1) {
    mscclpp::Timer timer;
    for (int iter = 0; iter < maxIter; ++iter) {
      stageAtomicAdd(0, 0, 1, true);
      qp->postSend();
      bool waiting = true;
      int spin = 0;
      while (waiting) {
        int wcNum = qp->pollCq();
        ASSERT_GE(wcNum, 0);
        for (int i = 0; i < wcNum; ++i) {
          const ibv_wc* wc = qp->getWc(i);
          EXPECT_EQ(wc->status, IBV_WC_SUCCESS);
          waiting = false;
          break;
        }
        if (spin++ > 1000000) {
          FAIL() << "Polling is stuck.";
        }
      }
    }
    float us = (float)timer.elapsed();
    std::cout << "IbPeerToPeerTest.SimpleAtomicAdd: " << us / maxIter << " us/iter" << std::endl;
  }
  bootstrap->barrier();
}
