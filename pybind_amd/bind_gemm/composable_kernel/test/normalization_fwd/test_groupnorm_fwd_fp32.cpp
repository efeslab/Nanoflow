// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include "gtest/gtest.h"
#include "profiler/profile_groupnorm_fwd_impl.hpp"

using F16 = ck::half_t;
using F32 = float;
using ck::index_t;

template <typename Tuple>
class TestGroupnorm : public ::testing::Test
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
        // [N, H, W, G, C], reduce H, W, C
        std::vector<std::vector<ck::index_t>> lengths = {{1, 1, 1, 1, 1},
                                                         {1, 2, 3, 4, 5},
                                                         {256, 9, 9, 9, 9},
                                                         {1, 64, 64, 32, 10},
                                                         {1, 32, 32, 32, 20},
                                                         {1, 16, 16, 32, 40}};

        for(auto length : lengths)
        {
            bool success =
                ck::profiler::profile_groupnorm_impl<XDataType,
                                                     GammaDataType,
                                                     BetaDataType,
                                                     ComputeDataType,
                                                     YDataType,
                                                     SaveMeanInvStdDataType,
                                                     true>(true, 2, false, false, length);
            EXPECT_TRUE(success);
        }
    }
};

using KernelTypes = ::testing::Types<
    // XDataType, GammaDataType, BetaDataType, ComputeDataType, YDataType, SaveMeanInvStdDataType>
    std::tuple<F32, F32, F32, F32, F32, F32>>;

TYPED_TEST_SUITE(TestGroupnorm, KernelTypes);
TYPED_TEST(TestGroupnorm, Test_FP32) { this->Run(); }
