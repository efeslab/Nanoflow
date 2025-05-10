// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <mscclpp/core.hpp>

class LocalCommunicatorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    bootstrap = std::make_shared<mscclpp::TcpBootstrap>(0, 1);
    bootstrap->initialize(bootstrap->createUniqueId());
    comm = std::make_shared<mscclpp::Communicator>(bootstrap);
  }

  std::shared_ptr<mscclpp::TcpBootstrap> bootstrap;
  std::shared_ptr<mscclpp::Communicator> comm;
};

class MockSetuppable : public mscclpp::Setuppable {
 public:
  MOCK_METHOD(void, beginSetup, (std::shared_ptr<mscclpp::Bootstrap> bootstrap), (override));
  MOCK_METHOD(void, endSetup, (std::shared_ptr<mscclpp::Bootstrap> bootstrap), (override));
};

TEST_F(LocalCommunicatorTest, OnSetup) {
  auto mockSetuppable = std::make_shared<MockSetuppable>();
  comm->onSetup(mockSetuppable);
  EXPECT_CALL(*mockSetuppable, beginSetup(std::dynamic_pointer_cast<mscclpp::Bootstrap>(bootstrap)));
  EXPECT_CALL(*mockSetuppable, endSetup(std::dynamic_pointer_cast<mscclpp::Bootstrap>(bootstrap)));
  comm->setup();
}

TEST_F(LocalCommunicatorTest, RegisterMemory) {
  int dummy[42];
  auto memory = comm->registerMemory(&dummy, sizeof(dummy), mscclpp::NoTransports);
  EXPECT_EQ(memory.data(), &dummy);
  EXPECT_EQ(memory.size(), sizeof(dummy));
  EXPECT_EQ(memory.transports(), mscclpp::NoTransports);
}

TEST_F(LocalCommunicatorTest, SendMemoryToSelf) {
  int dummy[42];
  auto memory = comm->registerMemory(&dummy, sizeof(dummy), mscclpp::NoTransports);
  comm->sendMemoryOnSetup(memory, 0, 0);
  auto memoryFuture = comm->recvMemoryOnSetup(0, 0);
  comm->setup();
  auto sameMemory = memoryFuture.get();
  EXPECT_EQ(sameMemory.data(), memory.data());
  EXPECT_EQ(sameMemory.size(), memory.size());
  EXPECT_EQ(sameMemory.transports(), memory.transports());
}
