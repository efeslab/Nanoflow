// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <gtest/gtest.h>

#include <mscclpp/gpu_utils.hpp>

TEST(CudaUtilsTest, AllocShared) {
  auto p1 = mscclpp::allocSharedCuda<uint32_t>();
  auto p2 = mscclpp::allocSharedCuda<int64_t>(5);
}

TEST(CudaUtilsTest, AllocUnique) {
  auto p1 = mscclpp::allocUniqueCuda<uint32_t>();
  auto p2 = mscclpp::allocUniqueCuda<int64_t>(5);
}

TEST(CudaUtilsTest, MakeSharedHost) {
  auto p1 = mscclpp::makeSharedCudaHost<uint32_t>();
  auto p2 = mscclpp::makeSharedCudaHost<int64_t>(5);
}

TEST(CudaUtilsTest, MakeUniqueHost) {
  auto p1 = mscclpp::makeUniqueCudaHost<uint32_t>();
  auto p2 = mscclpp::makeUniqueCudaHost<int64_t>(5);
}

TEST(CudaUtilsTest, Memcpy) {
  const int nElem = 1024;
  std::vector<int> hostBuff(nElem);
  for (int i = 0; i < nElem; ++i) {
    hostBuff[i] = i + 1;
  }
  std::vector<int> hostBuffTmp(nElem, 0);
  auto devBuff = mscclpp::allocSharedCuda<int>(nElem);
  mscclpp::memcpyCuda<int>(devBuff.get(), hostBuff.data(), nElem, cudaMemcpyHostToDevice);
  mscclpp::memcpyCuda<int>(hostBuffTmp.data(), devBuff.get(), nElem, cudaMemcpyDeviceToHost);

  for (int i = 0; i < nElem; ++i) {
    EXPECT_EQ(hostBuff[i], hostBuffTmp[i]);
  }
}
