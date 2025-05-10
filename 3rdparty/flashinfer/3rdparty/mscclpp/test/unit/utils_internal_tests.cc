#include <gtest/gtest.h>

#include <thread>

#include "utils_internal.hpp"

TEST(UtilsInternalTest, getHostHash) {
  uint64_t hash1 = mscclpp::getHostHash();
  uint64_t hash2;

  std::thread th([&hash2]() { hash2 = mscclpp::getHostHash(); });

  ASSERT_TRUE(th.joinable());
  th.join();

  EXPECT_EQ(hash1, hash2);
}
