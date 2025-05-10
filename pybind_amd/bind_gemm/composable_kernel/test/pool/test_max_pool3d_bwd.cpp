// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include "gtest/gtest.h"
#include "profiler/profile_max_pool3d_bwd_impl.hpp"
#include "test_pool_fwd_common.hpp"

template <typename Tuple>
class TestMaxPool3dBwd : public ::testing::Test
{
    protected:
    using DOutDataType  = std::tuple_element_t<0, Tuple>;
    using DInDataType   = std::tuple_element_t<1, Tuple>;
    using IndexDataType = std::tuple_element_t<2, Tuple>;

    using InDataType  = DInDataType;
    using OutDataType = DOutDataType;

    std::vector<PoolingParam> params;

    void Run()
    {
        for(auto param : params)
        {
            bool success =
                ck::profiler::profile_max_pool3d_bwd_impl<InDataType,
                                                          OutDataType,
                                                          IndexDataType,
                                                          DOutDataType,
                                                          DInDataType,
                                                          false>(true,
                                                                 2,
                                                                 false,
                                                                 false,
                                                                 param.length_,
                                                                 param.window_spatial_lengths_,
                                                                 param.window_strides_,
                                                                 param.window_dilations_,
                                                                 param.input_left_pads_,
                                                                 param.input_right_pads_);
            EXPECT_TRUE(success);
        }
    }
};

#if defined(CK_ENABLE_FP16) && defined(CK_ENABLE_BF16) && defined(CK_ENABLE_FP32)
using KernelTypes = ::testing::Types<std::tuple<F16, F16, I32, NDHWC, NDHWC>,
                                     std::tuple<BF16, BF16, I32, NDHWC, NDHWC>,
                                     std::tuple<F32, F32, I32, NDHWC, NDHWC>>;
#elif defined(CK_ENABLE_FP16) && defined(CK_ENABLE_FP32)
using KernelTypes = ::testing::Types<std::tuple<F16, F16, I32, NDHWC, NDHWC>,
                                     std::tuple<F32, F32, I32, NDHWC, NDHWC>>;
#elif defined(CK_ENABLE_BF16) && defined(CK_ENABLE_FP32)
using KernelTypes = ::testing::Types<std::tuple<BF16, BF16, I32, NDHWC, NDHWC>,
                                     std::tuple<F32, F32, I32, NDHWC, NDHWC>>;
#elif defined(CK_ENABLE_FP16) && defined(CK_ENABLE_BF16)
using KernelTypes = ::testing::Types<std::tuple<F16, F16, I32, NDHWC, NDHWC>,
                                     std::tuple<BF16, BF16, I32, NDHWC, NDHWC>>;
#elif defined(CK_ENABLE_FP16)
using KernelTypes = ::testing::Types<std::tuple<F16, F16, I32, NDHWC, NDHWC>>;
#elif defined(CK_ENABLE_BF16)
using KernelTypes = ::testing::Types<std::tuple<BF16, BF16, I32, NDHWC, NDHWC>>;
#elif defined(CK_ENABLE_FP32)
using KernelTypes = ::testing::Types<std::tuple<F32, F32, I32, NDHWC, NDHWC>>;
#endif

TYPED_TEST_SUITE(TestMaxPool3dBwd, KernelTypes);
TYPED_TEST(TestMaxPool3dBwd, Test_Pool)
{
    // length, window_length, window_stride, window_dilation, left_pad, right_pad
    this->params = {{{1, 1, 1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}},
                    {{2, 16, 64, 64, 64}, {4, 4, 4}, {4, 4, 4}, {2, 2, 2}, {0, 0, 0}, {0, 0, 0}},
                    {{2, 32, 30, 30, 30}, {2, 2, 2}, {2, 2, 2}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}}};

    // this->params = {{{2, 32, 30, 30, 30}, {2, 2, 2}, {2, 2, 2}, {1, 1, 1}, {1, 1, 1}, {1, 1,
    // 1}}};

    this->Run();
}
