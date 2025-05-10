// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#include <cstdlib>
#include <iostream>
#include <initializer_list>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "ck/utility/common_header.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/host_utility/device_prop.hpp"

#include "profiler/profile_grouped_conv_bwd_weight_impl.hpp"

using namespace ck::tensor_layout::convolution;

template <typename Tuple>
class TestGroupedConvndBwdWeight : public ::testing::Test
{
    protected:
    using InDataType  = std::tuple_element_t<0, Tuple>;
    using WeiDataType = std::tuple_element_t<1, Tuple>;
    using OutDataType = std::tuple_element_t<2, Tuple>;
    using InLayout    = std::tuple_element_t<3, Tuple>;
    using WeiLayout   = std::tuple_element_t<4, Tuple>;
    using OutLayout   = std::tuple_element_t<5, Tuple>;
    using NDimSpatial = std::tuple_element_t<6, Tuple>;

    std::vector<ck::utils::conv::ConvParam> conv_params;
    std::vector<ck::index_t> split_ks{1, 2};

    bool skip_case(const ck::index_t split_k)
    {
        // 1d NWGC is only supported by DL kernel
        // DL kernel is only supported for split_k=1
        if constexpr(std::is_same_v<InLayout, NWGC> && std::is_same_v<OutLayout, NWGK>)
        {
            if(split_k != 1)
            {
                return true;
            }
        }

        if(ck::is_gfx11_supported())
        {
            // on gfx11 only support for 3d is implemented
            if constexpr(NDimSpatial{} != 3)
            {
                return true;
            }
            // on gfx11 only support for i8 and fp16 is implemented
            if constexpr(!((std::is_same_v<InDataType, int8_t> &&
                            std::is_same_v<WeiDataType, int8_t> &&
                            std::is_same_v<OutDataType, int8_t>) ||
                           (std::is_same_v<InDataType, ck::half_t> &&
                            std::is_same_v<WeiDataType, ck::half_t> &&
                            std::is_same_v<OutDataType, ck::half_t>)))
            {
                return true;
            }
            // WMMA kernel is only supported for split_k=1
            if(split_k != 1)
            {
                return true;
            }
        }
        else
        {
            // support for i8 is only implemented on gfx11
            if constexpr(std::is_same_v<InDataType, int8_t> &&
                         std::is_same_v<WeiDataType, int8_t> && std::is_same_v<OutDataType, int8_t>)
            {
                return true;
            }
        }

        return false;
    }

    void Run()
    {
        EXPECT_FALSE(conv_params.empty());
        bool pass = true;

        for(auto split_k : split_ks)
        {
            for(auto& param : conv_params)
            {
                if(!skip_case(split_k))
                {
                    pass = pass && ck::profiler::profile_grouped_conv_bwd_weight_impl<NDimSpatial{},
                                                                                      InLayout,
                                                                                      WeiLayout,
                                                                                      OutLayout,
                                                                                      InDataType,
                                                                                      WeiDataType,
                                                                                      OutDataType>(
                                       true,  // do_verification
                                       1,     // init_method: integer value
                                       false, // do_log
                                       false, // time_kernel
                                       param,
                                       split_k);
                }
            }
        }
        EXPECT_TRUE(pass);
    }
};

template <typename Tuple>
class TestGroupedConvndBwdWeight1d : public TestGroupedConvndBwdWeight<Tuple>
{
};

template <typename Tuple>
class TestGroupedConvndBwdWeight2d : public TestGroupedConvndBwdWeight<Tuple>
{
};

template <typename Tuple>
class TestGroupedConvndBwdWeight3d : public TestGroupedConvndBwdWeight<Tuple>
{
};

using KernelTypes1d = ::testing::Types<
    std::tuple<float, float, float, GNWC, GKXC, GNWK, ck::Number<1>>,
    std::tuple<ck::half_t, ck::half_t, ck::half_t, GNWC, GKXC, GNWK, ck::Number<1>>,
    std::tuple<ck::bhalf_t, float, ck::bhalf_t, GNWC, GKXC, GNWK, ck::Number<1>>,
    std::tuple<float, float, float, NWGC, GKXC, NWGK, ck::Number<1>>,
    std::tuple<ck::half_t, ck::half_t, ck::half_t, NWGC, GKXC, NWGK, ck::Number<1>>,
    std::tuple<ck::bhalf_t, float, ck::bhalf_t, NWGC, GKXC, NWGK, ck::Number<1>>>;
using KernelTypes2d = ::testing::Types<
    std::tuple<float, float, float, GNHWC, GKYXC, GNHWK, ck::Number<2>>,
    std::tuple<ck::half_t, ck::half_t, ck::half_t, GNHWC, GKYXC, GNHWK, ck::Number<2>>,
    std::tuple<ck::bhalf_t, float, ck::bhalf_t, GNHWC, GKYXC, GNHWK, ck::Number<2>>,
    std::tuple<float, float, float, NHWGC, GKYXC, NHWGK, ck::Number<2>>,
    std::tuple<ck::half_t, ck::half_t, ck::half_t, NHWGC, GKYXC, NHWGK, ck::Number<2>>,
    std::tuple<ck::bhalf_t, float, ck::bhalf_t, NHWGC, GKYXC, NHWGK, ck::Number<2>>>;
using KernelTypes3d = ::testing::Types<
    std::tuple<float, float, float, GNDHWC, GKZYXC, GNDHWK, ck::Number<3>>,
    std::tuple<ck::half_t, ck::half_t, ck::half_t, GNDHWC, GKZYXC, GNDHWK, ck::Number<3>>,
    std::tuple<ck::bhalf_t, float, ck::bhalf_t, GNDHWC, GKZYXC, GNDHWK, ck::Number<3>>,
    std::tuple<int8_t, int8_t, int8_t, GNDHWC, GKZYXC, GNDHWK, ck::Number<3>>,
    std::tuple<float, float, float, NDHWGC, GKZYXC, NDHWGK, ck::Number<3>>,
    std::tuple<ck::half_t, ck::half_t, ck::half_t, NDHWGC, GKZYXC, NDHWGK, ck::Number<3>>,
    std::tuple<ck::bhalf_t, float, ck::bhalf_t, NDHWGC, GKZYXC, NDHWGK, ck::Number<3>>,
    std::tuple<int8_t, int8_t, int8_t, NDHWGC, GKZYXC, NDHWGK, ck::Number<3>>>;

TYPED_TEST_SUITE(TestGroupedConvndBwdWeight1d, KernelTypes1d);
TYPED_TEST_SUITE(TestGroupedConvndBwdWeight2d, KernelTypes2d);
TYPED_TEST_SUITE(TestGroupedConvndBwdWeight3d, KernelTypes3d);

TYPED_TEST(TestGroupedConvndBwdWeight1d, Test1D)
{
    this->conv_params.clear();
    this->conv_params.push_back({1, 2, 128, 128, 256, {1}, {14}, {2}, {1}, {0}, {0}});
    this->conv_params.push_back({1, 2, 32, 128, 256, {3}, {28}, {1}, {1}, {1}, {1}});
    this->conv_params.push_back({1, 2, 128, 128, 256, {1}, {3}, {1}, {1}, {0}, {0}});
    this->conv_params.push_back({1, 1, 1, 1, 32, {3}, {32}, {1}, {1}, {1}, {1}});
    this->conv_params.push_back({1, 1, 1, 64, 3, {3}, {32}, {1}, {1}, {1}, {1}});
    this->conv_params.push_back({1, 1, 1, 1, 1, {3}, {32}, {1}, {1}, {1}, {1}});
    this->Run();
}

TYPED_TEST(TestGroupedConvndBwdWeight2d, Test2D)
{
    this->conv_params.clear();
    this->conv_params.push_back(
        {2, 2, 64, 128, 256, {1, 1}, {7, 7}, {2, 2}, {1, 1}, {0, 0}, {0, 0}});
    this->conv_params.push_back(
        {2, 2, 4, 128, 256, {3, 3}, {14, 14}, {1, 1}, {1, 1}, {1, 1}, {1, 1}});
    this->conv_params.push_back(
        {2, 2, 128, 128, 256, {1, 1}, {3, 3}, {1, 1}, {1, 1}, {0, 0}, {0, 0}});
    this->conv_params.push_back({2, 1, 1, 1, 32, {3, 3}, {32, 32}, {1, 1}, {1, 1}, {1, 1}, {1, 1}});
    this->conv_params.push_back({2, 1, 1, 64, 3, {3, 3}, {32, 32}, {1, 1}, {1, 1}, {1, 1}, {1, 1}});
    this->conv_params.push_back({2, 1, 1, 1, 1, {3, 3}, {32, 32}, {1, 1}, {1, 1}, {1, 1}, {1, 1}});
    this->conv_params.push_back(
        {2, 16, 16, 1, 1, {3, 3}, {28, 28}, {2, 2}, {1, 1}, {1, 1}, {1, 1}});
    this->Run();
}

TYPED_TEST(TestGroupedConvndBwdWeight3d, Test3D)
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
    this->conv_params.push_back(
        {3, 16, 16, 1, 1, {3, 3, 3}, {28, 28, 28}, {2, 2, 2}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}});
    this->Run();
}
