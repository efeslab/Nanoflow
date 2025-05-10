// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include <cstdlib>
#include <iostream>
#include <initializer_list>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "profiler/profile_grouped_conv_bwd_data_impl.hpp"

template <typename Tuple>
class TestGroupedConvndBwdData : public ::testing::Test
{
    protected:
    using DataType  = std::tuple_element_t<0, Tuple>;
    using OutLayout = std::tuple_element_t<1, Tuple>;
    using WeiLayout = std::tuple_element_t<2, Tuple>;
    using InLayout  = std::tuple_element_t<3, Tuple>;

    std::vector<ck::utils::conv::ConvParam> conv_params;

    template <ck::index_t NDimSpatial>
    void Run()
    {
        EXPECT_FALSE(conv_params.empty());
        bool pass = true;
        for(auto& param : conv_params)
        {
            pass = pass && ck::profiler::profile_grouped_conv_bwd_data_impl<NDimSpatial,
                                                                            OutLayout,
                                                                            WeiLayout,
                                                                            InLayout,
                                                                            DataType,
                                                                            DataType,
                                                                            DataType>(
                               true,  // do_verification
                               1,     // init_method: integer value
                               false, // do_log
                               false, // time_kernel
                               param);
        }
        EXPECT_TRUE(pass);
    }
};

using namespace ck::tensor_layout::convolution;

using KernelTypes2d = ::testing::Types<std::tuple<float, GNHWK, GKYXC, GNHWC>,
                                       std::tuple<ck::half_t, GNHWK, GKYXC, GNHWC>,
                                       std::tuple<ck::bhalf_t, GNHWK, GKYXC, GNHWC>,
                                       std::tuple<int8_t, GNHWK, GKYXC, GNHWC>,
                                       std::tuple<float, NHWGK, GKYXC, NHWGC>,
                                       std::tuple<ck::half_t, NHWGK, GKYXC, NHWGC>,
                                       std::tuple<ck::bhalf_t, NHWGK, GKYXC, NHWGC>,
                                       std::tuple<int8_t, NHWGK, GKYXC, NHWGC>>;

using KernelTypes3d = ::testing::Types<std::tuple<float, GNDHWK, GKZYXC, GNDHWC>,
                                       std::tuple<ck::half_t, GNDHWK, GKZYXC, GNDHWC>,
                                       std::tuple<ck::bhalf_t, GNDHWK, GKZYXC, GNDHWC>,
                                       std::tuple<int8_t, GNDHWK, GKZYXC, GNDHWC>,
                                       std::tuple<float, NDHWGK, GKZYXC, NDHWGC>,
                                       std::tuple<ck::half_t, NDHWGK, GKZYXC, NDHWGC>,
                                       std::tuple<ck::bhalf_t, NDHWGK, GKZYXC, NDHWGC>,
                                       std::tuple<int8_t, NDHWGK, GKZYXC, NDHWGC>>;

template <typename Tuple>
class TestGroupedConvndBwdData2d : public TestGroupedConvndBwdData<Tuple>
{
};

template <typename Tuple>
class TestGroupedConvndBwdData3d : public TestGroupedConvndBwdData<Tuple>
{
};

TYPED_TEST_SUITE(TestGroupedConvndBwdData2d, KernelTypes2d);
TYPED_TEST_SUITE(TestGroupedConvndBwdData3d, KernelTypes3d);

TYPED_TEST(TestGroupedConvndBwdData2d, Test2D)
{
    this->conv_params.clear();

    this->conv_params.push_back(
        {2, 2, 4, 192, 192, {3, 3}, {28, 28}, {1, 1}, {1, 1}, {1, 1}, {1, 1}});
    this->conv_params.push_back(
        {2, 2, 128, 128, 256, {3, 3}, {14, 14}, {1, 1}, {1, 1}, {1, 1}, {1, 1}});
    this->conv_params.push_back(
        {2, 2, 128, 128, 256, {1, 1}, {7, 7}, {2, 2}, {1, 1}, {0, 0}, {0, 0}});
    this->conv_params.push_back(
        {2, 2, 128, 128, 256, {1, 1}, {3, 3}, {1, 1}, {1, 1}, {0, 0}, {0, 0}});
    this->conv_params.push_back({2, 1, 1, 1, 32, {8, 8}, {32, 32}, {1, 1}, {1, 1}, {1, 1}, {1, 1}});
    this->conv_params.push_back({2, 1, 1, 64, 3, {8, 8}, {32, 32}, {1, 1}, {1, 1}, {1, 1}, {1, 1}});
    this->conv_params.push_back({2, 1, 1, 1, 1, {8, 8}, {32, 32}, {1, 1}, {1, 1}, {1, 1}, {1, 1}});
    this->template Run<2>();
}

TYPED_TEST(TestGroupedConvndBwdData3d, Test3D)
{
    this->conv_params.clear();
    this->conv_params.push_back(
        {3, 2, 16, 128, 256, {1, 1, 1}, {7, 7, 7}, {2, 2, 2}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}});
    this->conv_params.push_back(
        {3, 2, 2, 128, 256, {3, 3, 3}, {14, 14, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}});
    this->conv_params.push_back(
        {3, 2, 32, 128, 256, {1, 1, 1}, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}});
    this->conv_params.push_back(
        {3, 1, 1, 1, 32, {3, 3, 3}, {32, 32, 32}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}});
    this->conv_params.push_back(
        {3, 1, 1, 64, 3, {3, 3, 3}, {32, 32, 32}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}});
    this->conv_params.push_back(
        {3, 1, 1, 1, 1, {3, 3, 3}, {32, 32, 32}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}});
    this->template Run<3>();
}
