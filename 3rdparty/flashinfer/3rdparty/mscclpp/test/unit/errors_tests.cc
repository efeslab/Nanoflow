// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <gtest/gtest.h>

#include <mscclpp/errors.hpp>

TEST(ErrorsTest, SystemError) {
  mscclpp::Error error("test", mscclpp::ErrorCode::SystemError);
  EXPECT_EQ(error.getErrorCode(), mscclpp::ErrorCode::SystemError);
  EXPECT_EQ(error.what(), std::string("test (Mscclpp failure: SystemError)"));
}

TEST(ErrorsTest, InternalError) {
  mscclpp::Error error("test", mscclpp::ErrorCode::InternalError);
  EXPECT_EQ(error.getErrorCode(), mscclpp::ErrorCode::InternalError);
  EXPECT_EQ(error.what(), std::string("test (Mscclpp failure: InternalError)"));
}

TEST(ErrorsTest, InvalidUsage) {
  mscclpp::Error error("test", mscclpp::ErrorCode::InvalidUsage);
  EXPECT_EQ(error.getErrorCode(), mscclpp::ErrorCode::InvalidUsage);
  EXPECT_EQ(error.what(), std::string("test (Mscclpp failure: InvalidUsage)"));
}

TEST(ErrorsTest, Timeout) {
  mscclpp::Error error("test", mscclpp::ErrorCode::Timeout);
  EXPECT_EQ(error.getErrorCode(), mscclpp::ErrorCode::Timeout);
  EXPECT_EQ(error.what(), std::string("test (Mscclpp failure: Timeout)"));
}
