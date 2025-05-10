// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <algorithm>

#include "mp_unit_tests.hpp"

void SmChannelOneToOneTest::SetUp() {
  // Need at least two ranks within a node
  if (gEnv->nRanksPerNode < 2) {
    GTEST_SKIP();
  }
  // Use only two ranks
  setNumRanksToUse(2);
  CommunicatorTestBase::SetUp();
}

void SmChannelOneToOneTest::TearDown() { CommunicatorTestBase::TearDown(); }

void SmChannelOneToOneTest::setupMeshConnections(std::vector<mscclpp::SmChannel>& smChannels, void* inputBuff,
                                                 size_t inputBuffBytes, void* outputBuff, size_t outputBuffBytes) {
  const int rank = communicator->bootstrap()->getRank();
  const int worldSize = communicator->bootstrap()->getNranks();
  const bool isInPlace = (outputBuff == nullptr);
  mscclpp::TransportFlags transport = mscclpp::Transport::CudaIpc | ibTransport;

  std::vector<mscclpp::NonblockingFuture<std::shared_ptr<mscclpp::Connection>>> connectionFutures(worldSize);
  std::vector<mscclpp::NonblockingFuture<mscclpp::RegisteredMemory>> remoteMemFutures(worldSize);

  mscclpp::RegisteredMemory inputBufRegMem = communicator->registerMemory(inputBuff, inputBuffBytes, transport);
  mscclpp::RegisteredMemory outputBufRegMem;
  if (!isInPlace) {
    outputBufRegMem = communicator->registerMemory(outputBuff, outputBuffBytes, transport);
  }

  for (int r = 0; r < worldSize; r++) {
    if (r == rank) {
      continue;
    }
    if (rankToNode(r) == rankToNode(gEnv->rank)) {
      connectionFutures[r] = communicator->connectOnSetup(r, 0, mscclpp::Transport::CudaIpc);
    } else {
      connectionFutures[r] = communicator->connectOnSetup(r, 0, ibTransport);
    }

    if (isInPlace) {
      communicator->sendMemoryOnSetup(inputBufRegMem, r, 0);
    } else {
      communicator->sendMemoryOnSetup(outputBufRegMem, r, 0);
    }
    remoteMemFutures[r] = communicator->recvMemoryOnSetup(r, 0);
  }

  communicator->setup();

  for (int r = 0; r < worldSize; r++) {
    if (r == rank) {
      continue;
    }
    connections[r] = connectionFutures[r].get();

    smSemaphores[r] = std::make_shared<mscclpp::SmDevice2DeviceSemaphore>(*communicator, connections[r]);

    smChannels.emplace_back(smSemaphores[r], remoteMemFutures[r].get(), inputBufRegMem.data(),
                            (isInPlace ? nullptr : outputBufRegMem.data()));
  }

  communicator->setup();
}

__constant__ DeviceHandle<mscclpp::SmChannel> gChannelOneToOneTestConstSmChans;

void SmChannelOneToOneTest::packetPingPongTest(const std::string testName, PacketPingPongKernelWrapper kernelWrapper) {
  if (gEnv->rank >= numRanksToUse) return;

  const int nElem = 4 * 1024 * 1024;
  const int defaultNTries = 1000;

  std::vector<mscclpp::SmChannel> smChannels;
  std::shared_ptr<int> buff = mscclpp::allocExtSharedCuda<int>(nElem);
  std::shared_ptr<int> intermBuff = mscclpp::allocExtSharedCuda<int>(nElem * 2);
  setupMeshConnections(smChannels, buff.get(), nElem * sizeof(int), intermBuff.get(), nElem * 2 * sizeof(int));
  std::vector<DeviceHandle<mscclpp::SmChannel>> deviceHandles(smChannels.size());
  std::transform(smChannels.begin(), smChannels.end(), deviceHandles.begin(),
                 [](const mscclpp::SmChannel& smChan) { return mscclpp::deviceHandle(smChan); });

  ASSERT_EQ(smChannels.size(), 1);
  MSCCLPP_CUDATHROW(cudaMemcpyToSymbol(gChannelOneToOneTestConstSmChans, deviceHandles.data(),
                                       sizeof(DeviceHandle<mscclpp::SmChannel>)));

  std::shared_ptr<int> ret = mscclpp::makeSharedCudaHost<int>(0);

  // The least nelem is 2 for packet ping pong
  kernelWrapper(buff.get(), gEnv->rank, 2, ret.get(), defaultNTries);
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());
  *ret = 0;

  kernelWrapper(buff.get(), gEnv->rank, 1024, ret.get(), defaultNTries);
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  EXPECT_EQ(*ret, 0);
  *ret = 0;

  kernelWrapper(buff.get(), gEnv->rank, 1024 * 1024, ret.get(), defaultNTries);
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  EXPECT_EQ(*ret, 0);
  *ret = 0;

  kernelWrapper(buff.get(), gEnv->rank, 4 * 1024 * 1024, ret.get(), defaultNTries);
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  EXPECT_EQ(*ret, 0);
  *ret = 0;

  int nTries = 1000000;
  communicator->bootstrap()->barrier();
  mscclpp::Timer timer;
  kernelWrapper(buff.get(), gEnv->rank, 1024, ret.get(), nTries);
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());
  communicator->bootstrap()->barrier();

  if (gEnv->rank == 0) {
    std::cout << testName << ": " << std::setprecision(4) << (float)timer.elapsed() / (float)(nTries) << " us/iter\n";
  }
}

__global__ void kernelSmPutPingPong(int* buff, int rank, int nElem, int* ret) {
  DeviceHandle<mscclpp::SmChannel>& smChan = gChannelOneToOneTestConstSmChans;
  volatile int* sendBuff = (volatile int*)buff;
  int nTries = 1000;
  int rank1Offset = 10000000;
  for (int i = 0; i < nTries; i++) {
    if (rank == 0) {
      if (i > 0) {
        if (threadIdx.x == 0) smChan.wait();
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
      smChan.put(0, 0, nElem * sizeof(int), threadIdx.x, blockDim.x);
      if (threadIdx.x == 0) smChan.signal();
    }
    if (rank == 1) {
      if (threadIdx.x == 0) smChan.wait();
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
        smChan.put(0, 0, nElem * sizeof(int), threadIdx.x, blockDim.x);
        if (threadIdx.x == 0) smChan.signal();
      }
    }
  }
}

TEST_F(SmChannelOneToOneTest, PutPingPong) {
  if (gEnv->rank >= numRanksToUse) return;

  const int nElem = 4 * 1024 * 1024;

  std::vector<mscclpp::SmChannel> smChannels;
  std::shared_ptr<int> buff = mscclpp::allocExtSharedCuda<int>(nElem);
  setupMeshConnections(smChannels, buff.get(), nElem * sizeof(int));
  std::vector<DeviceHandle<mscclpp::SmChannel>> deviceHandles(smChannels.size());
  std::transform(smChannels.begin(), smChannels.end(), deviceHandles.begin(),
                 [](const mscclpp::SmChannel& smChan) { return mscclpp::deviceHandle(smChan); });

  ASSERT_EQ(smChannels.size(), 1);
  MSCCLPP_CUDATHROW(cudaMemcpyToSymbol(gChannelOneToOneTestConstSmChans, deviceHandles.data(),
                                       sizeof(DeviceHandle<mscclpp::SmChannel>)));

  std::shared_ptr<int> ret = mscclpp::makeSharedCudaHost<int>(0);

  kernelSmPutPingPong<<<1, 1024>>>(buff.get(), gEnv->rank, 1, ret.get());
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  EXPECT_EQ(*ret, 0);
  *ret = 0;

  kernelSmPutPingPong<<<1, 1024>>>(buff.get(), gEnv->rank, 1024, ret.get());
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  EXPECT_EQ(*ret, 0);
  *ret = 0;

  kernelSmPutPingPong<<<1, 1024>>>(buff.get(), gEnv->rank, 1024 * 1024, ret.get());
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  EXPECT_EQ(*ret, 0);
  *ret = 0;

  kernelSmPutPingPong<<<1, 1024>>>(buff.get(), gEnv->rank, 4 * 1024 * 1024, ret.get());
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  EXPECT_EQ(*ret, 0);
}

__global__ void kernelSmGetPingPong(int* buff, int rank, int nElem, int* ret) {
  if (rank > 1) return;

  DeviceHandle<mscclpp::SmChannel>& smChan = gChannelOneToOneTestConstSmChans;
  volatile int* buffPtr = (volatile int*)buff;
  int offset0 = (rank == 0) ? 0 : 10000000;
  int offset1 = (rank == 0) ? 10000000 : 0;
  int nTries = 1000;

  for (int i = 0; i < nTries; i++) {
    // rank=0: 0, 1, 0, 1, ...
    // rank=1: 1, 0, 1, 0, ...
    if ((rank ^ (i & 1)) == 0) {
      for (int j = threadIdx.x; j < nElem; j += blockDim.x) {
        buffPtr[j] = offset0 + i + j;
      }
      if (threadIdx.x == 0) {
        smChan.signal();
      }
    } else {
      if (threadIdx.x == 0) {
        smChan.wait();
      }
      __syncthreads();
      smChan.get(0, 0, nElem * sizeof(int), threadIdx.x, blockDim.x);
      __syncthreads();
      for (int j = threadIdx.x; j < nElem; j += blockDim.x) {
        if (buffPtr[j] != offset1 + i + j) {
          // printf("rank %d ERROR: buff[%d] = %d, expected %d\n", rank, j, buffPtr[j], offset1 + i + j);
          *ret = 1;
          break;
        }
      }
    }
  }
}

TEST_F(SmChannelOneToOneTest, GetPingPong) {
  if (gEnv->rank >= numRanksToUse) return;

  const int nElem = 4 * 1024 * 1024;

  std::vector<mscclpp::SmChannel> smChannels;
  std::shared_ptr<int> buff = mscclpp::allocExtSharedCuda<int>(nElem);
  setupMeshConnections(smChannels, buff.get(), nElem * sizeof(int));
  std::vector<DeviceHandle<mscclpp::SmChannel>> deviceHandles(smChannels.size());
  std::transform(smChannels.begin(), smChannels.end(), deviceHandles.begin(),
                 [](const mscclpp::SmChannel& smChan) { return mscclpp::deviceHandle(smChan); });

  ASSERT_EQ(deviceHandles.size(), 1);
  MSCCLPP_CUDATHROW(cudaMemcpyToSymbol(gChannelOneToOneTestConstSmChans, deviceHandles.data(),
                                       sizeof(DeviceHandle<mscclpp::SmChannel>)));

  std::shared_ptr<int> ret = mscclpp::makeSharedCudaHost<int>(0);

  kernelSmGetPingPong<<<1, 1024>>>(buff.get(), gEnv->rank, 1, ret.get());
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  EXPECT_EQ(*ret, 0);
  *ret = 0;

  kernelSmGetPingPong<<<1, 1024>>>(buff.get(), gEnv->rank, 1024, ret.get());
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  EXPECT_EQ(*ret, 0);
  *ret = 0;

  kernelSmGetPingPong<<<1, 1024>>>(buff.get(), gEnv->rank, 1024 * 1024, ret.get());
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  EXPECT_EQ(*ret, 0);
  *ret = 0;

  kernelSmGetPingPong<<<1, 1024>>>(buff.get(), gEnv->rank, 4 * 1024 * 1024, ret.get());
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  EXPECT_EQ(*ret, 0);
}

__global__ void kernelSmLL8PacketPingPong(int* buff, int rank, int nElem, int* ret, int nTries) {
  if (rank > 1) return;

  DeviceHandle<mscclpp::SmChannel>& smChan = gChannelOneToOneTestConstSmChans;
  volatile int* sendBuff = (volatile int*)buff;
  int putOffset = (rank == 0) ? 0 : 10000000;
  int getOffset = (rank == 0) ? 10000000 : 0;
  for (int i = 0; i < nTries; i++) {
    uint64_t flag = (uint64_t)i + 1;

    // rank=0: 0, 1, 0, 1, ...
    // rank=1: 1, 0, 1, 0, ...
    if ((rank ^ (i & 1)) == 0) {
      // If each thread writes 8 bytes at once, we don't need a barrier before putPackets().
      for (int j = threadIdx.x; j < nElem; j += blockDim.x) {
        sendBuff[j] = putOffset + i + j;
        // sendBuff[2 * j + 1] = putOffset + i + 2 * j + 1;
      }
      // __syncthreads();
      smChan.putPackets<mscclpp::LL8Packet>(0, 0, nElem * sizeof(int), threadIdx.x, blockDim.x, flag);
    } else {
      smChan.getPackets<mscclpp::LL8Packet>(0, 0, nElem * sizeof(int), threadIdx.x, blockDim.x, flag);
      // If each thread reads 8 bytes at once, we don't need a barrier after getPackets().
      // __syncthreads();
      for (int j = threadIdx.x; j < nElem; j += blockDim.x) {
        if (sendBuff[j] != getOffset + i + j) {
          // printf("ERROR: rank = %d, sendBuff[%d] = %d, expected %d. Skipping following errors\n", rank, 2 * j,
          //        sendBuff[2 * j], getOffset + i + 2 * j);
          *ret = 1;
          break;
        }
      }
    }
    // Make sure all threads are done in this iteration
    __syncthreads();
  }
}

__global__ void kernelSmLL16PacketPingPong(int* buff, int rank, int nElem, int* ret, int nTries) {
  if (rank > 1) return;

  DeviceHandle<mscclpp::SmChannel>& smChan = gChannelOneToOneTestConstSmChans;
  volatile int* sendBuff = (volatile int*)buff;
  int putOffset = (rank == 0) ? 0 : 10000000;
  int getOffset = (rank == 0) ? 10000000 : 0;
  for (int i = 0; i < nTries; i++) {
    uint64_t flag = (uint64_t)i + 1;
    // rank=0: 0, 1, 0, 1, ...
    // rank=1: 1, 0, 1, 0, ...
    if ((rank ^ (i & 1)) == 0) {
      // If each thread writes 8 bytes at once, we don't need a barrier before putPackets().
      for (int j = threadIdx.x; j < nElem / 2; j += blockDim.x) {
        sendBuff[2 * j] = putOffset + i + 2 * j;
        sendBuff[2 * j + 1] = putOffset + i + 2 * j + 1;
      }
      // __syncthreads();
      smChan.putPackets<mscclpp::LL16Packet>(0, 0, nElem * sizeof(int), threadIdx.x, blockDim.x, flag);
    } else {
      smChan.getPackets<mscclpp::LL16Packet>(0, 0, nElem * sizeof(int), threadIdx.x, blockDim.x, flag);
      // If each thread reads 8 bytes at once, we don't need a barrier after getPackets().
      // __syncthreads();
      for (int j = threadIdx.x; j < nElem / 2; j += blockDim.x) {
        if (sendBuff[2 * j] != getOffset + i + 2 * j) {
          // printf("ERROR: rank = %d, sendBuff[%d] = %d, expected %d. Skipping following errors\n", rank, 2 * j,
          //        sendBuff[2 * j], getOffset + i + 2 * j);
          *ret = 1;
          break;
        }
        if (sendBuff[2 * j + 1] != getOffset + i + 2 * j + 1) {
          // printf("ERROR: rank = %d, sendBuff[%d] = %d, expected %d. Skipping following errors\n", rank, 2 * j + 1,
          //        sendBuff[2 * j + 1], getOffset + i + 2 * j + 1);
          *ret = 1;
          break;
        }
      }
    }
    // Make sure all threads are done in this iteration
    __syncthreads();
  }
}

TEST_F(SmChannelOneToOneTest, LL8PacketPingPong) {
  auto kernelSmLL8PacketPingPongWrapper = [](int* buff, int rank, int nElem, int* ret, int nTries) {
    kernelSmLL8PacketPingPong<<<1, 1024>>>(buff, rank, nElem, ret, nTries);
  };
  packetPingPongTest("smLL8PacketPingPong", kernelSmLL8PacketPingPongWrapper);
}

TEST_F(SmChannelOneToOneTest, LL16PacketPingPong) {
  auto kernelSmLL16PacketPingPongWrapper = [](int* buff, int rank, int nElem, int* ret, int nTries) {
    kernelSmLL16PacketPingPong<<<1, 1024>>>(buff, rank, nElem, ret, nTries);
  };
  packetPingPongTest("smLL16PacketPingPong", kernelSmLL16PacketPingPongWrapper);
}
