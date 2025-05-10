// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include "gtest/gtest.h"
#include "profiler/profile_layernorm_fwd_impl.hpp"

using F16 = ck::half_t;
using F32 = float;
using ck::index_t;

template <typename Tuple>
class TestLayernorm4d : public ::testing::Test
{
    protected:
    using XDataType              = std::tuple_element_t<0, Tuple>;
    using GammaDataType          = std::tuple_element_t<1, Tuple>;
    using BetaDataType           = std::tuple_element_t<2, Tuple>;
    using ComputeDataType        = std::tuple_element_t<3, Tuple>;
    using YDataType              = std::tuple_element_t<4, Tuple>;
    using SaveMeanInvStdDataType = std::tuple_element_t<5, Tuple>;

    void Run()
    {
        // [N, D], reduce D
        std::vector<std::vector<ck::index_t>> lengths = {
            {1, 1, 1, 1}, {7, 7, 7, 7}, {256, 16, 16, 8}};

        for(auto length : lengths)
        {
            bool success = ck::profiler::profile_layernorm_impl<XDataType,
                                                                GammaDataType,
                                                                BetaDataType,
                                                                ComputeDataType,
                                                                YDataType,
                                                                SaveMeanInvStdDataType,
                                                                true,
                                                                4>(true, 2, false, false, length);
            EXPECT_TRUE(success);
        }
    }
};

using KernelTypes = ::testing::Types<
    // XDataType, GammaDataType, BetaDataType, ComputeDataType, YDataType, SaveMeanInvStdDataType>
    std::tuple<F16, F16, F16, F32, F16, F16>>;

TYPED_TEST_SUITE(TestLayernorm4d, KernelTypes);
TYPED_TEST(TestLayernorm4d, Test_FP16) { this->Run(); }
