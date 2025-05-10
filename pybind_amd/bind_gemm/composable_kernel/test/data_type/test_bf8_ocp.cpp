// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "gtest/gtest.h"
#include "ck/utility/data_type.hpp"
#include "ck/utility/type_convert.hpp"

using ck::bf8_ocp_t;
using ck::f8_convert_rne;
using ck::f8_convert_sr;
using ck::half_t;
using ck::type_convert;

TEST(BF8OCP, NumericLimits)
{ // constants given for OCP FP8
    EXPECT_EQ(ck::NumericLimits<bf8_ocp_t>::Min(),
              type_convert<bf8_ocp_t>(0x04)); // 0b00000100 = 2^-14
    EXPECT_EQ(ck::NumericLimits<bf8_ocp_t>::Max(),
              type_convert<bf8_ocp_t>(0x7B)); // 0b01111011 = 57344
    EXPECT_EQ(ck::NumericLimits<bf8_ocp_t>::Lowest(),
              type_convert<bf8_ocp_t>(0xFB)); // 0b11111011 = -57344
    EXPECT_EQ(ck::NumericLimits<bf8_ocp_t>::QuietNaN().data,
              type_convert<bf8_ocp_t>(0x7D).data); // 0b01111101
    EXPECT_FALSE(ck::NumericLimits<bf8_ocp_t>::QuietNaN() ==
                 ck::NumericLimits<bf8_ocp_t>::QuietNaN());
    EXPECT_TRUE(ck::fp8_is_inf(type_convert<bf8_ocp_t>(0xFC)) &&
                ck::fp8_is_inf(type_convert<bf8_ocp_t>(0x7C)));
}

TEST(BF8OCP, ConvertFP32Nearest)
{
    // fix the tolerance value
    float abs_tol = 1e-6;

    // convert 0 float to bfp8 and back, check if holds
    ASSERT_NEAR(0.0f, type_convert<float>(f8_convert_rne<bf8_ocp_t>(0.0f)), 0.0f);

    // convert minimal float to bf8 and back, check if holds
    ASSERT_NEAR(std::numeric_limits<float>::min(),
                type_convert<float>(f8_convert_rne<bf8_ocp_t>(std::numeric_limits<float>::min())),
                abs_tol);

    const auto max_bf8_t_float = type_convert<float>(ck::NumericLimits<bf8_ocp_t>::Max());

    // convert maximal bf8_ocp_t to float and check if equal to bf8 max
    ASSERT_NEAR(
        max_bf8_t_float, type_convert<float>(f8_convert_rne<bf8_ocp_t>(max_bf8_t_float)), 0.0f);

    // convert maximal float to bf8 and back, check if clipped to bf8 max (saturation to finite)
    ASSERT_NEAR(max_bf8_t_float,
                type_convert<float>(f8_convert_rne<bf8_ocp_t>(std::numeric_limits<float>::max())),
                0.0f);

    // convert float infinity to bf8_ocp_t and check if it is max value (saturation to finite)
    ASSERT_EQ(ck::NumericLimits<bf8_ocp_t>::Max(),
              f8_convert_rne<bf8_ocp_t>(std::numeric_limits<float>::infinity()));

    // positive normal float value to bf8 and back, check if holds
    float pos_float = 0.0000762939f; // 10*2^-17
    ASSERT_NEAR(pos_float, type_convert<float>(f8_convert_rne<bf8_ocp_t>(pos_float)), abs_tol);

    // negative smallest normal bf8 value to bf8 and back, check if holds
    constexpr auto neg_min_bf8 = -0.00006103515625f; //-2^-14
    ASSERT_NEAR(neg_min_bf8, type_convert<float>(f8_convert_rne<bf8_ocp_t>(neg_min_bf8)), 0.0f);

    // positive subnorm float value to bf8 and back, check if holds
    constexpr auto pos_subnorm_bf8 = 0.000030517578125f; // 2^-15
    ASSERT_NEAR(
        pos_subnorm_bf8, type_convert<float>(f8_convert_rne<bf8_ocp_t>(pos_subnorm_bf8)), 0.0f);

    // min subnorm bf8 value to bf8 and back, check if holds
    constexpr auto min_subnorm_bf8 = -0.0000152587890625f; //-2^-16
    ASSERT_NEAR(
        min_subnorm_bf8, type_convert<float>(f8_convert_rne<bf8_ocp_t>(min_subnorm_bf8)), 0.0f);

    // smaller than min subnorm bf8 value to bf8 must be zero
    constexpr auto less_than_min_subnorm = 0.00000762939453125f; // 2^-17
    ASSERT_EQ(0.0f, type_convert<float>(f8_convert_rne<bf8_ocp_t>(less_than_min_subnorm)));

    // convert quiet NaN to bf8_ocp_t and check if it is quiet NaN
    const auto bf8_nan = f8_convert_rne<bf8_ocp_t>(std::numeric_limits<float>::quiet_NaN());
    ASSERT_TRUE(ck::fp8_impl::ocp_bf8_is_nan(bf8_nan.data));
}

TEST(BF8OCP, ConvertFP32Stochastic)
{
    // fix the tolerance value
    float abs_tol = 1e-6;

    // convert 0 float to bfp8 and back, check if holds
    ASSERT_NEAR(0.0f, type_convert<float>(f8_convert_sr<bf8_ocp_t>(0.0f)), 0.0f);

    // convert minimal float to bf8 and back, check if holds
    ASSERT_NEAR(std::numeric_limits<float>::min(),
                type_convert<float>(f8_convert_sr<bf8_ocp_t>(std::numeric_limits<float>::min())),
                abs_tol);

    const auto max_bf8_t_float = type_convert<float>(ck::NumericLimits<bf8_ocp_t>::Max());

    // convert maximal bf8_ocp_t to float and check if equal to bf8 max
    ASSERT_NEAR(
        max_bf8_t_float, type_convert<float>(f8_convert_sr<bf8_ocp_t>(max_bf8_t_float)), 0.0f);

    // convert maximal float to bf8 and back, check if clipped to bf8 max (saturation to finite)
    ASSERT_NEAR(max_bf8_t_float,
                type_convert<float>(f8_convert_sr<bf8_ocp_t>(std::numeric_limits<float>::max())),
                0.0f);

    // convert float infinity to bf8_ocp_t and check if it is max value (saturation to finite)
    ASSERT_EQ(ck::NumericLimits<bf8_ocp_t>::Max(),
              f8_convert_sr<bf8_ocp_t>(std::numeric_limits<float>::infinity()));

    // positive normal float value to bf8 and back, check if holds
    float pos_float = 0.0000762939f; // 10*2^-17
    ASSERT_NEAR(pos_float, type_convert<float>(f8_convert_sr<bf8_ocp_t>(pos_float)), abs_tol);

    // negative smallest normal bf8 value to bf8 and back, check if holds
    constexpr auto neg_min_bf8 = -0.00006103515625f; //-2^-14
    ASSERT_NEAR(neg_min_bf8, type_convert<float>(f8_convert_sr<bf8_ocp_t>(neg_min_bf8)), 0.0f);

    // positive subnorm float value to bf8 and back, check if holds
    constexpr auto pos_subnorm_bf8 = 0.000030517578125f; // 2^-15
    ASSERT_NEAR(
        pos_subnorm_bf8, type_convert<float>(f8_convert_sr<bf8_ocp_t>(pos_subnorm_bf8)), 0.0f);

    // min subnorm bf8 value to bf8 and back, check if holds
    constexpr auto min_subnorm_bf8 = -0.0000152587890625f; //-2^-16
    ASSERT_NEAR(
        min_subnorm_bf8, type_convert<float>(f8_convert_sr<bf8_ocp_t>(min_subnorm_bf8)), 0.0f);

    // smaller than min subnorm bf8 value to bf8  alternates between 0 and 2^-16
    constexpr auto less_than_min_subnorm = 0.00000762939453125f; // 2^-17
    ASSERT_NEAR(0.0f,
                type_convert<float>(f8_convert_sr<bf8_ocp_t>(less_than_min_subnorm)),
                0.0000152587890625f);

    // convert quiet NaN to bf8_ocp_t and check if it is quiet NaN
    const auto bf8_nan = f8_convert_sr<bf8_ocp_t>(std::numeric_limits<float>::quiet_NaN());
    ASSERT_TRUE(ck::fp8_impl::ocp_bf8_is_nan(bf8_nan.data));
}

TEST(BF8OCP, ConvertFP16Nearest)
{
    // fix the tolerance value
    constexpr half_t half_t_tol  = 1e-3;
    constexpr half_t half_t_zero = 0.0;

    // convert 0 half_t to bfp8 and back, check if holds
    ASSERT_NEAR(
        half_t_zero, type_convert<half_t>(f8_convert_rne<bf8_ocp_t>(half_t_zero)), half_t_zero);

    // convert minimal half_t to bf8 and back, check if holds
    ASSERT_NEAR(ck::NumericLimits<half_t>::Min(),
                type_convert<half_t>(f8_convert_rne<bf8_ocp_t>(ck::NumericLimits<half_t>::Min())),
                half_t_tol);

    const auto max_bf8_t_half_t = type_convert<half_t>(ck::NumericLimits<bf8_ocp_t>::Max());

    // convert maximal bf8_ocp_t to half_t and check if equal to bf8 max
    ASSERT_NEAR(max_bf8_t_half_t,
                type_convert<half_t>(f8_convert_rne<bf8_ocp_t>(max_bf8_t_half_t)),
                half_t_zero);

    // convert maximal half_t to bf8 and back, check if clipped to bf8 max (saturation to finite)
    ASSERT_NEAR(max_bf8_t_half_t,
                type_convert<half_t>(f8_convert_rne<bf8_ocp_t>(ck::NumericLimits<half_t>::Max())),
                half_t_zero);

    // convert half_t infinity to bf8_ocp_t and check if it is max value (saturation to finite)
    ASSERT_EQ(
        ck::NumericLimits<bf8_ocp_t>::Max(),
        f8_convert_rne<bf8_ocp_t>(type_convert<half_t>(std::numeric_limits<float>::infinity())));

    // positive normal bf8 value to bf8 and back, check if holds
    constexpr half_t pos_norm_bf8{0.0000762939f}; // 10*2^-17
    ASSERT_NEAR(
        pos_norm_bf8, type_convert<half_t>(f8_convert_rne<bf8_ocp_t>(pos_norm_bf8)), half_t_tol);

    // negative smallest normal bf8 value to bf8 and back, check if holds
    constexpr half_t neg_min_bf8{-0.00006103515625f}; //-2^-14
    ASSERT_NEAR(
        neg_min_bf8, type_convert<half_t>(f8_convert_rne<bf8_ocp_t>(neg_min_bf8)), half_t_zero);

    // positive subnorm bf8 value to bf8 and back, check if holds
    constexpr half_t pos_subnorm_bf8{0.000030517578125f}; // 2^-15
    ASSERT_NEAR(pos_subnorm_bf8,
                type_convert<half_t>(f8_convert_rne<bf8_ocp_t>(pos_subnorm_bf8)),
                half_t_zero);

    // min subnorm bf8 value to bf8 and back, check if holds
    constexpr half_t min_subnorm_bf8{-0.0000152587890625f}; //-2^-16
    ASSERT_NEAR(min_subnorm_bf8,
                type_convert<half_t>(f8_convert_rne<bf8_ocp_t>(min_subnorm_bf8)),
                half_t_zero);

    // smaller than min subnorm bf8 value to bf8 must be zero
    constexpr half_t less_than_min_subnorm{0.00000762939453125f}; // 2^-17
    ASSERT_EQ(half_t_zero, type_convert<half_t>(f8_convert_rne<bf8_ocp_t>(less_than_min_subnorm)));

    // convert quiet NaN to bf8_ocp_t and check if it is quiet NaN
    const auto bf8_nan = f8_convert_rne<bf8_ocp_t>(ck::NumericLimits<half_t>::QuietNaN());
    ASSERT_TRUE(ck::fp8_impl::ocp_bf8_is_nan(bf8_nan.data));
}

TEST(BF8OCP, ConvertFP16Stochastic)
{
    // fix the tolerance value
    constexpr half_t half_t_tol    = 1e-3;
    constexpr half_t half_t_zero   = 0.0;
    constexpr auto min_subnorm_bf8 = 0.0000152587890625f; // 2^-16

    // convert 0 half_t to bfp8 and back, check if holds
    ASSERT_NEAR(
        half_t_zero, type_convert<half_t>(f8_convert_sr<bf8_ocp_t>(half_t_zero)), half_t_zero);

    // convert minimal half_t (6.103515625e-05) to fp8 and back
    ASSERT_NEAR(ck::NumericLimits<half_t>::Min(),
                type_convert<half_t>(f8_convert_sr<bf8_ocp_t>(ck::NumericLimits<half_t>::Min())),
                half_t_zero);

    const auto max_bf8_t_half_t = type_convert<half_t>(ck::NumericLimits<bf8_ocp_t>::Max());

    // convert maximal bf8_ocp_t to half_t and check if equal to bf8 max
    ASSERT_NEAR(max_bf8_t_half_t,
                type_convert<half_t>(f8_convert_sr<bf8_ocp_t>(max_bf8_t_half_t)),
                half_t_zero);

    // convert maximal half_t to bf8 and back, check if clipped to bf8 max (saturation to finite)
    ASSERT_NEAR(max_bf8_t_half_t,
                type_convert<half_t>(f8_convert_sr<bf8_ocp_t>(ck::NumericLimits<half_t>::Max())),
                half_t_zero);

    // convert half_t infinity to bf8_ocp_t and check if it is max value (saturation to finite)
    ASSERT_EQ(
        ck::NumericLimits<bf8_ocp_t>::Max(),
        f8_convert_sr<bf8_ocp_t>(type_convert<half_t>(std::numeric_limits<float>::infinity())));

    // positive normal bf8 value to bf8 and back, check if holds
    constexpr half_t pos_norm_bf8{0.0000762939f}; // 10*2^-17
    ASSERT_NEAR(
        pos_norm_bf8, type_convert<half_t>(f8_convert_sr<bf8_ocp_t>(pos_norm_bf8)), half_t_tol);

    // negative smallest normal bf8 value to bf8 and back, check if holds
    constexpr half_t neg_min_bf8{-0.00006103515625f}; //-2^-14
    ASSERT_NEAR(
        neg_min_bf8, type_convert<half_t>(f8_convert_sr<bf8_ocp_t>(neg_min_bf8)), half_t_zero);

    // positive subnorm bf8 value to bf8 and back, check if holds
    constexpr half_t pos_subnorm_bf8{0.000030517578125f}; // 2^-15
    ASSERT_NEAR(pos_subnorm_bf8,
                type_convert<half_t>(f8_convert_sr<bf8_ocp_t>(pos_subnorm_bf8)),
                half_t_zero);

    // min subnorm bf8 value to bf8 and back, check if holds
    ASSERT_NEAR(half_t{-min_subnorm_bf8},
                type_convert<half_t>(f8_convert_sr<bf8_ocp_t>(half_t{-min_subnorm_bf8})),
                half_t_zero);

    // smaller than min subnorm bf8 value to bf8  alternates between 0 and 2^-16
    constexpr half_t less_than_min_subnorm{0.00000762939453125f}; // 2^-17
    ASSERT_NEAR(half_t_zero,
                type_convert<half_t>(f8_convert_sr<bf8_ocp_t>(less_than_min_subnorm)),
                half_t{min_subnorm_bf8});

    // convert quiet NaN to bf8_ocp_t and check if it is quiet NaN
    const auto bf8_nan = f8_convert_sr<bf8_ocp_t>(ck::NumericLimits<half_t>::QuietNaN());
    ASSERT_TRUE(ck::fp8_impl::ocp_bf8_is_nan(bf8_nan.data));
}
