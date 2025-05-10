// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <cstdint>
#include <mscclpp/concurrency_device.hpp>

#include "mp_unit_tests.hpp"

void ProxyChannelOneToOneTest::SetUp() {
  // Use only two ranks
  setNumRanksToUse(2);
  CommunicatorTestBase::SetUp();
  proxyService = std::make_shared<mscclpp::ProxyService>();
}

void ProxyChannelOneToOneTest::TearDown() { CommunicatorTestBase::TearDown(); }

void ProxyChannelOneToOneTest::setupMeshConnections(std::vector<mscclpp::SimpleProxyChannel>& proxyChannels,
                                                    bool useIPC, bool useIb, bool useEthernet, void* sendBuff,
                                                    size_t sendBuffBytes, void* recvBuff, size_t recvBuffBytes) {
  const int rank = communicator->bootstrap()->getRank();
  const int worldSize = communicator->bootstrap()->getNranks();
  const bool isInPlace = (recvBuff == nullptr);
  mscclpp::TransportFlags transport;

  if (useIPC) transport |= mscclpp::Transport::CudaIpc;
  if (useIb) transport |= ibTransport;
  if (useEthernet) transport |= mscclpp::Transport::Ethernet;

  std::vector<mscclpp::NonblockingFuture<std::shared_ptr<mscclpp::Connection>>> connectionFutures(worldSize);
  std::vector<mscclpp::NonblockingFuture<mscclpp::RegisteredMemory>> remoteMemFutures(worldSize);

  mscclpp::RegisteredMemory sendBufRegMem = communicator->registerMemory(sendBuff, sendBuffBytes, transport);
  mscclpp::RegisteredMemory recvBufRegMem;
  if (!isInPlace) {
    recvBufRegMem = communicator->registerMemory(recvBuff, recvBuffBytes, transport);
  }

  for (int r = 0; r < worldSize; r++) {
    if (r == rank) {
      continue;
    }
    if ((rankToNode(r) == rankToNode(gEnv->rank)) && useIPC) {
      connectionFutures[r] = communicator->connectOnSetup(r, 0, mscclpp::Transport::CudaIpc);
    } else if (useIb) {
      connectionFutures[r] = communicator->connectOnSetup(r, 0, ibTransport);
    } else if (useEthernet) {
      connectionFutures[r] = communicator->connectOnSetup(r, 0, mscclpp::Transport::Ethernet);
    }

    if (isInPlace) {
      communicator->sendMemoryOnSetup(sendBufRegMem, r, 0);
    } else {
      communicator->sendMemoryOnSetup(recvBufRegMem, r, 0);
    }
    remoteMemFutures[r] = communicator->recvMemoryOnSetup(r, 0);
  }

  communicator->setup();

  for (int r = 0; r < worldSize; r++) {
    if (r == rank) {
      continue;
    }
    mscclpp::SemaphoreId cid = proxyService->buildAndAddSemaphore(*communicator, connectionFutures[r].get());

    proxyChannels.emplace_back(proxyService->proxyChannel(cid), proxyService->addMemory(remoteMemFutures[r].get()),
                               proxyService->addMemory(sendBufRegMem));
  }

  communicator->setup();
}

__constant__ DeviceHandle<mscclpp::SimpleProxyChannel> gChannelOneToOneTestConstProxyChans;

__global__ void kernelProxyPingPong(int* buff, int rank, int nElem, bool waitWithPoll, int nTries, int* ret) {
  DeviceHandle<mscclpp::SimpleProxyChannel>& proxyChan = gChannelOneToOneTestConstProxyChans;
  volatile int* sendBuff = (volatile int*)buff;
  int flusher = 0;
  int rank1Offset = 10000000;
  for (int i = 0; i < nTries; i++) {
    if (rank == 0) {
      if (i > 0) {
        if (threadIdx.x == 0) {
          if (waitWithPoll) {
            int spin = 1000000;
            while (!proxyChan.poll() && spin > 0) {
              spin--;
            }
            if (spin == 0) {
              // printf("rank 0 ERROR: poll timeout\n");
              *ret = 1;
            }
          } else {
            proxyChan.wait();
          }
        }
        __syncthreads();
        for (int j = threadIdx.x; j < nElem; j += blockDim.x) {
          if (sendBuff[j] != rank1Offset + i - 1 + j) {
            // printf("rank 0 ERROR: sendBuff[%d] = %d, expected %d\n", j, sendBuff[j], rank1Offset + i - 1 + j);
            *ret = 1;
            break;
          }
        }
      }
      for (int j = threadIdx.x; j < nElem; j += blockDim.x) {
        sendBuff[j] = i + j;
      }
      __syncthreads();
      // __threadfence_system(); // not necessary if we make sendBuff volatile
      if (threadIdx.x == 0) proxyChan.putWithSignal(0, nElem * sizeof(int));
    }
    if (rank == 1) {
      if (threadIdx.x == 0) {
        if (waitWithPoll) {
          int spin = 1000000;
          while (!proxyChan.poll() && spin > 0) {
            spin--;
          }
          if (spin == 0) {
            // printf("rank 0 ERROR: poll timeout\n");
            *ret = 1;
          }
        } else {
          proxyChan.wait();
        }
      }
      __syncthreads();
      for (int j = threadIdx.x; j < nElem; j += blockDim.x) {
        if (sendBuff[j] != i + j) {
          // printf("rank 1 ERROR: sendBuff[%d] = %d, expected %d\n", j, sendBuff[j], i + j);
          *ret = 1;
          break;
        }
      }
      if (i < nTries - 1) {
        for (int j = threadIdx.x; j < nElem; j += blockDim.x) {
          sendBuff[j] = rank1Offset + i + j;
        }
        __syncthreads();
        // __threadfence_system(); // not necessary if we make sendBuff volatile
        if (threadIdx.x == 0) proxyChan.putWithSignal(0, nElem * sizeof(int));
      }
    }
    flusher++;
    if (flusher == 1) {
      if (threadIdx.x == 0) proxyChan.flush();
      flusher = 0;
    }
  }
}

void ProxyChannelOneToOneTest::testPingPong(PingPongTestParams params) {
  if (gEnv->rank >= numRanksToUse) return;

  const int nElem = 4 * 1024 * 1024;

  std::vector<mscclpp::SimpleProxyChannel> proxyChannels;
  std::shared_ptr<int> buff = mscclpp::allocExtSharedCuda<int>(nElem);
  setupMeshConnections(proxyChannels, params.useIPC, params.useIB, params.useEthernet, buff.get(), nElem * sizeof(int));

  std::vector<DeviceHandle<mscclpp::SimpleProxyChannel>> proxyChannelHandles;
  for (auto& ch : proxyChannels) proxyChannelHandles.push_back(ch.deviceHandle());

  ASSERT_EQ(proxyChannels.size(), 1);
  MSCCLPP_CUDATHROW(cudaMemcpyToSymbol(gChannelOneToOneTestConstProxyChans, proxyChannelHandles.data(),
                                       sizeof(DeviceHandle<mscclpp::SimpleProxyChannel>)));

  proxyService->startProxy();

  std::shared_ptr<int> ret = mscclpp::makeSharedCudaHost<int>(0);

  const int nTries = 1000;

  kernelProxyPingPong<<<1, 1024>>>(buff.get(), gEnv->rank, 1, params.waitWithPoll, nTries, ret.get());
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  EXPECT_EQ(*ret, 0);

  kernelProxyPingPong<<<1, 1024>>>(buff.get(), gEnv->rank, 1024, params.waitWithPoll, nTries, ret.get());
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  EXPECT_EQ(*ret, 0);

  kernelProxyPingPong<<<1, 1024>>>(buff.get(), gEnv->rank, 1024 * 1024, params.waitWithPoll, nTries, ret.get());
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  EXPECT_EQ(*ret, 0);

  kernelProxyPingPong<<<1, 1024>>>(buff.get(), gEnv->rank, 4 * 1024 * 1024, params.waitWithPoll, nTries, ret.get());
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  EXPECT_EQ(*ret, 0);

  proxyService->stopProxy();
}

void ProxyChannelOneToOneTest::testPingPongPerf(PingPongTestParams params) {
  if (gEnv->rank >= numRanksToUse) return;

  const int nElem = 4 * 1024 * 1024;

  std::vector<mscclpp::SimpleProxyChannel> proxyChannels;
  std::shared_ptr<int> buff = mscclpp::allocExtSharedCuda<int>(nElem);
  setupMeshConnections(proxyChannels, params.useIPC, params.useIB, params.useEthernet, buff.get(), nElem * sizeof(int));

  std::vector<DeviceHandle<mscclpp::SimpleProxyChannel>> proxyChannelHandles;
  for (auto& ch : proxyChannels) proxyChannelHandles.push_back(ch.deviceHandle());

  ASSERT_EQ(proxyChannels.size(), 1);
  MSCCLPP_CUDATHROW(cudaMemcpyToSymbol(gChannelOneToOneTestConstProxyChans, proxyChannelHandles.data(),
                                       sizeof(DeviceHandle<mscclpp::SimpleProxyChannel>)));

  proxyService->startProxy();

  std::shared_ptr<int> ret = mscclpp::makeSharedCudaHost<int>(0);

  auto* testInfo = ::testing::UnitTest::GetInstance()->current_test_info();
  const std::string testName = std::string(testInfo->test_suite_name()) + "." + std::string(testInfo->name());
  const int nTries = 1000;

  // Warm-up
  kernelProxyPingPong<<<1, 1024>>>(buff.get(), gEnv->rank, 1, params.waitWithPoll, nTries, ret.get());
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  communicator->bootstrap()->barrier();

  // Measure latency
  mscclpp::Timer timer;
  kernelProxyPingPong<<<1, 1024>>>(buff.get(), gEnv->rank, 1, params.waitWithPoll, nTries, ret.get());
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  communicator->bootstrap()->barrier();

  if (gEnv->rank == 0) {
    std::cout << testName << ": " << std::setprecision(4) << (float)timer.elapsed() / (float)nTries << " us/iter\n";
  }

  proxyService->stopProxy();
}

TEST_F(ProxyChannelOneToOneTest, PingPong) {
  testPingPong(PingPongTestParams{.useIPC = true, .useIB = true, .useEthernet = false, .waitWithPoll = false});
}

TEST_F(ProxyChannelOneToOneTest, PingPongIb) {
  testPingPong(PingPongTestParams{.useIPC = false, .useIB = true, .useEthernet = false, .waitWithPoll = false});
}

TEST_F(ProxyChannelOneToOneTest, PingPongEthernet) {
  testPingPong(PingPongTestParams{.useIPC = false, .useIB = false, .useEthernet = true, .waitWithPoll = false});
}

TEST_F(ProxyChannelOneToOneTest, PingPongWithPoll) {
  testPingPong(PingPongTestParams{.useIPC = true, .useIB = true, .useEthernet = false, .waitWithPoll = true});
}

TEST_F(ProxyChannelOneToOneTest, PingPongIbWithPoll) {
  testPingPong(PingPongTestParams{.useIPC = false, .useIB = true, .useEthernet = false, .waitWithPoll = true});
}

TEST_F(ProxyChannelOneToOneTest, PingPongPerf) {
  testPingPongPerf(PingPongTestParams{.useIPC = true, .useIB = true, .useEthernet = false, .waitWithPoll = false});
}

TEST_F(ProxyChannelOneToOneTest, PingPongPerfIb) {
  testPingPongPerf(PingPongTestParams{.useIPC = false, .useIB = true, .useEthernet = false, .waitWithPoll = false});
}

TEST_F(ProxyChannelOneToOneTest, PingPongPerfEthernet) {
  testPingPongPerf(PingPongTestParams{.useIPC = false, .useIB = false, .useEthernet = true, .waitWithPoll = false});
}

__device__ mscclpp::DeviceSyncer gChannelOneToOneTestProxyChansSyncer;

template <bool CheckCorrectness>
__global__ void kernelProxyLLPingPong(int* buff, mscclpp::LLPacket* putPktBuf, mscclpp::LLPacket* getPktBuf, int rank,
                                      int nElem, int nTries, int* ret) {
  if (rank > 1) return;

  DeviceHandle<mscclpp::SimpleProxyChannel>& proxyChan = gChannelOneToOneTestConstProxyChans;
  volatile int* buffPtr = (volatile int*)buff;
  int putOffset = (rank == 0) ? 0 : 10000000;
  int getOffset = (rank == 0) ? 10000000 : 0;
  int threadId = threadIdx.x + blockIdx.x * blockDim.x;
  int numThreads = blockDim.x * gridDim.x;
  int flusher = 0;
  const int nPkt = nElem / 2;
  for (int i = 0; i < nTries; i++) {
    uint64_t flag = (uint64_t)i + 1;

    // rank=0: 0, 1, 0, 1, ...
    // rank=1: 1, 0, 1, 0, ...
    if ((rank ^ (i & 1)) == 0) {
      if (CheckCorrectness) {
        // If each thread writes 8 bytes at once, we don't need a barrier before putPackets().
        for (int j = threadId; j < nPkt; j += numThreads) {
          buffPtr[2 * j] = putOffset + i + 2 * j;
          buffPtr[2 * j + 1] = putOffset + i + 2 * j + 1;
        }
        // __syncthreads();
      }
      mscclpp::putPackets(putPktBuf, 0, buff, 0, nElem * sizeof(int), threadId, numThreads, flag);
      gChannelOneToOneTestProxyChansSyncer.sync(gridDim.x);
      if (threadId == 0) {
        // Send data from the local putPacketBuffer to the remote getPacketBuffer
        proxyChan.put(0, nPkt * sizeof(mscclpp::LLPacket));
      }
      flusher++;
      if (flusher == 64) {
        if (threadId == 0) proxyChan.flush();
        flusher = 0;
      }
    } else {
      mscclpp::getPackets(getPktBuf, 0, buff, 0, nElem * sizeof(int), threadId, numThreads, flag);
      if (CheckCorrectness) {
        // If each thread reads 8 bytes at once, we don't need a barrier after getPackets().
        // __syncthreads();
        for (int j = threadId; j < nPkt; j += numThreads) {
          if (buffPtr[2 * j] != getOffset + i + 2 * j) {
            // printf("ERROR: rank = %d, buffPtr[%d] = %d, expected %d. Skipping following errors\n", rank, 2 * j,
            //        buffPtr[2 * j], getOffset + i + 2 * j);
            *ret = 1;
            break;
          }
          if (buffPtr[2 * j + 1] != getOffset + i + 2 * j + 1) {
            // printf("ERROR: rank = %d, buffPtr[%d] = %d, expected %d. Skipping following errors\n", rank, 2 * j + 1,
            //        buffPtr[2 * j + 1], getOffset + i + 2 * j + 1);
            *ret = 1;
            break;
          }
        }
      }
      // Make sure all threads are done in this iteration
      gChannelOneToOneTestProxyChansSyncer.sync(gridDim.x);
    }
  }
}

void ProxyChannelOneToOneTest::testPacketPingPong(bool useIbOnly) {
  if (gEnv->rank >= numRanksToUse) return;

  const int nElem = 4 * 1024 * 1024;

  std::vector<mscclpp::SimpleProxyChannel> proxyChannels;
  std::shared_ptr<int> buff = mscclpp::allocExtSharedCuda<int>(nElem);

  const size_t nPacket = (nElem * sizeof(int) + sizeof(uint64_t) - 1) / sizeof(uint64_t);
  auto putPacketBuffer = mscclpp::allocExtSharedCuda<mscclpp::LLPacket>(nPacket);
  auto getPacketBuffer = mscclpp::allocExtSharedCuda<mscclpp::LLPacket>(nPacket);

  setupMeshConnections(proxyChannels, !useIbOnly, true, false, putPacketBuffer.get(),
                       nPacket * sizeof(mscclpp::LLPacket), getPacketBuffer.get(), nPacket * sizeof(mscclpp::LLPacket));

  ASSERT_EQ(proxyChannels.size(), 1);

  std::vector<DeviceHandle<mscclpp::SimpleProxyChannel>> proxyChannelHandles;
  for (auto& proxyChannel : proxyChannels) {
    proxyChannelHandles.push_back(proxyChannel.deviceHandle());
  }

  MSCCLPP_CUDATHROW(cudaMemcpyToSymbol(gChannelOneToOneTestConstProxyChans, proxyChannelHandles.data(),
                                       sizeof(DeviceHandle<mscclpp::SimpleProxyChannel>)));

  mscclpp::DeviceSyncer syncer = {};
  MSCCLPP_CUDATHROW(cudaMemcpyToSymbol(gChannelOneToOneTestProxyChansSyncer, &syncer, sizeof(mscclpp::DeviceSyncer)));

  proxyService->startProxy();

  std::shared_ptr<int> ret = mscclpp::makeSharedCudaHost<int>(0);

  const int nTries = 1000;

  // The least nelem is 2 for packet ping pong
  kernelProxyLLPingPong<true>
      <<<1, 1024>>>(buff.get(), putPacketBuffer.get(), getPacketBuffer.get(), gEnv->rank, 2, nTries, ret.get());
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  EXPECT_EQ(*ret, 0);
  *ret = 0;

  kernelProxyLLPingPong<true>
      <<<1, 1024>>>(buff.get(), putPacketBuffer.get(), getPacketBuffer.get(), gEnv->rank, 1024, nTries, ret.get());
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  EXPECT_EQ(*ret, 0);
  *ret = 0;

  kernelProxyLLPingPong<true><<<1, 1024>>>(buff.get(), putPacketBuffer.get(), getPacketBuffer.get(), gEnv->rank,
                                           1024 * 1024, nTries, ret.get());
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  EXPECT_EQ(*ret, 0);
  *ret = 0;

  kernelProxyLLPingPong<true><<<1, 1024>>>(buff.get(), putPacketBuffer.get(), getPacketBuffer.get(), gEnv->rank,
                                           4 * 1024 * 1024, nTries, ret.get());
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  EXPECT_EQ(*ret, 0);

  communicator->bootstrap()->barrier();

  proxyService->stopProxy();
}

void ProxyChannelOneToOneTest::testPacketPingPongPerf(bool useIbOnly) {
  if (gEnv->rank >= numRanksToUse) return;

  const int nElem = 4 * 1024 * 1024;

  std::vector<mscclpp::SimpleProxyChannel> proxyChannels;
  std::shared_ptr<int> buff = mscclpp::allocExtSharedCuda<int>(nElem);

  const size_t nPacket = (nElem * sizeof(int) + sizeof(uint64_t) - 1) / sizeof(uint64_t);
  auto putPacketBuffer = mscclpp::allocExtSharedCuda<mscclpp::LLPacket>(nPacket);
  auto getPacketBuffer = mscclpp::allocExtSharedCuda<mscclpp::LLPacket>(nPacket);

  setupMeshConnections(proxyChannels, !useIbOnly, true, false, putPacketBuffer.get(),
                       nPacket * sizeof(mscclpp::LLPacket), getPacketBuffer.get(), nPacket * sizeof(mscclpp::LLPacket));

  ASSERT_EQ(proxyChannels.size(), 1);

  std::vector<DeviceHandle<mscclpp::SimpleProxyChannel>> proxyChannelHandles;
  for (auto& proxyChannel : proxyChannels) {
    proxyChannelHandles.push_back(proxyChannel.deviceHandle());
  }

  MSCCLPP_CUDATHROW(cudaMemcpyToSymbol(gChannelOneToOneTestConstProxyChans, proxyChannelHandles.data(),
                                       sizeof(DeviceHandle<mscclpp::SimpleProxyChannel>)));

  mscclpp::DeviceSyncer syncer = {};
  MSCCLPP_CUDATHROW(cudaMemcpyToSymbol(gChannelOneToOneTestProxyChansSyncer, &syncer, sizeof(mscclpp::DeviceSyncer)));

  proxyService->startProxy();

  auto* testInfo = ::testing::UnitTest::GetInstance()->current_test_info();
  const std::string testName = std::string(testInfo->test_suite_name()) + "." + std::string(testInfo->name());
  const int nTries = 1000000;

  // Warm-up
  kernelProxyLLPingPong<false>
      <<<1, 1024>>>(buff.get(), putPacketBuffer.get(), getPacketBuffer.get(), gEnv->rank, 2, nTries, nullptr);
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  communicator->bootstrap()->barrier();

  // Measure latency
  mscclpp::Timer timer;
  kernelProxyLLPingPong<false>
      <<<1, 1024>>>(buff.get(), putPacketBuffer.get(), getPacketBuffer.get(), gEnv->rank, 2, nTries, nullptr);
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  communicator->bootstrap()->barrier();

  if (gEnv->rank == 0) {
    std::cout << testName << ": " << std::setprecision(4) << (float)timer.elapsed() / (float)nTries << " us/iter\n";
  }

  proxyService->stopProxy();
}

TEST_F(ProxyChannelOneToOneTest, PacketPingPong) { testPacketPingPong(false); }

TEST_F(ProxyChannelOneToOneTest, PacketPingPongIb) { testPacketPingPong(true); }

TEST_F(ProxyChannelOneToOneTest, PacketPingPongPerf) { testPacketPingPongPerf(false); }

TEST_F(ProxyChannelOneToOneTest, PacketPingPongPerfIb) { testPacketPingPongPerf(true); }
