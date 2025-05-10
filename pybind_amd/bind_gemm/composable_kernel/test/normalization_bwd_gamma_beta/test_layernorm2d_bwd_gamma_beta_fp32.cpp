// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include "gtest/gtest.h"
#include "profiler/profile_layernorm_bwd_gamma_beta_impl.hpp"

using F16 = ck::half_t;
using F32 = float;
using ck::index_t;

template <typename Tuple>
class TestLayernorm2dBwdGammaBeta : public ::testing::Test
{
    protected:
    using DYDataType         = std::tuple_element_t<0, Tuple>;
    using XDataType          = std::tuple_element_t<1, Tuple>;
    using MeanInvStdDataType = std::tuple_element_t<2, Tuple>;
    using ComputeDataType    = std::tuple_element_t<3, Tuple>;
    using DGammaDataType     = std::tuple_element_t<4, Tuple>;
    using DBetaDataType      = std::tuple_element_t<5, Tuple>;

    void Run()
    {
        // Bwd data: [N, D], reduce D
        std::vector<std::vector<ck::index_t>> lengths = {
            {4, 256}, {8, 511}, {9, 1032}, {4, 2048}, {1, 8192}, {4000, 2000}};

        for(auto length : lengths)
        {
            bool success = ck::profiler::profile_layernorm_bwd_gamma_beta_impl<DYDataType,
                                                                               XDataType,
                                                                               MeanInvStdDataType,
                                                                               ComputeDataType,
                                                                               DGammaDataType,
                                                                               DBetaDataType,
                                                                               2>(
                true, 2, false, false, length);
            EXPECT_TRUE(success);
        }
    }
};

using KernelTypes = ::testing::Types<
    // DYDataType XDataType, MeanInvStdDataType, ComputeDataType, DGammaDataType, DBetaDataType>
    std::tuple<F32, F32, F32, F32, F32, F32>>;

TYPED_TEST_SUITE(TestLayernorm2dBwdGammaBeta, KernelTypes);
TYPED_TEST(TestLayernorm2dBwdGammaBeta, Test_FP32) { this->Run(); }
