// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "gtest/gtest.h"
#include "ck/utility/data_type.hpp"
#include "ck/utility/type_convert.hpp"

using ck::bhalf_t;
using ck::type_convert;

TEST(BHALF_T, Nan)
{
    const uint16_t binary_bhalf_nan = 0x7FC0;
    const bhalf_t bhalf_nan         = ck::bit_cast<bhalf_t>(binary_bhalf_nan);
    EXPECT_EQ(bhalf_nan, type_convert<bhalf_t>(ck::NumericLimits<float>::QuietNaN()));
}

TEST(BHALF_T, Inf)
{
    const uint16_t binary_bhalf_inf = 0x7F80;
    const bhalf_t bhalf_inf         = ck::bit_cast<bhalf_t>(binary_bhalf_inf);
    EXPECT_EQ(bhalf_inf, type_convert<bhalf_t>(ck::NumericLimits<float>::Infinity()));
}

TEST(BHALF_T, MantisaOverflow)
{
    const float abs_tol   = std::pow(2, -7);
    const uint32_t val    = 0x81FFFFFF;
    const float float_val = ck::bit_cast<float>(val);

    ASSERT_NEAR(float_val, type_convert<float>(type_convert<bhalf_t>(float_val)), abs_tol);
}

TEST(BHALF_T, ExpOverflow)
{
    const uint32_t val    = 0xFF800000;
    const float float_val = ck::bit_cast<float>(val);
    ASSERT_EQ(type_convert<float>(type_convert<bhalf_t>(float_val)), float_val);
}

TEST(BHALF_T, MantisaExpOverflow)
{
    const uint32_t val    = 0xFFFFFFFF;
    const float float_val = ck::bit_cast<float>(val);

    ASSERT_TRUE(std::isnan(float_val));
    ASSERT_TRUE(std::isnan(type_convert<float>(type_convert<bhalf_t>(float_val))));
}
