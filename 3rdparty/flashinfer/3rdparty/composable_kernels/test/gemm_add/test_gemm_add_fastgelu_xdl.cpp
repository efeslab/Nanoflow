// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "gtest/gtest.h"
#include "ck/ck.hpp"
#include "profiler/profile_gemm_add_fastgelu_impl.hpp"
#include "test_gemm_add_xdl.hpp"

template <typename Tuple>
class TestGemmAddFastgelu : public TestGemmAdd<Tuple>
{
    private:
    using ADataType   = std::tuple_element_t<0, Tuple>;
    using BDataType   = std::tuple_element_t<1, Tuple>;
    using AccDataType = std::tuple_element_t<2, Tuple>;
    using D0DataType  = std::tuple_element_t<3, Tuple>;
    using EDataType   = std::tuple_element_t<4, Tuple>;
    using ALayout     = std::tuple_element_t<5, Tuple>;
    using BLayout     = std::tuple_element_t<6, Tuple>;
    using D0Layout    = std::tuple_element_t<7, Tuple>;
    using ELayout     = std::tuple_element_t<8, Tuple>;

    constexpr static auto ProfileGemmAddFastgeluImpl =
        ck::profiler::profile_gemm_add_fastgelu_impl<ADataType,
                                                     BDataType,
                                                     AccDataType,
                                                     D0DataType,
                                                     EDataType,
                                                     ALayout,
                                                     BLayout,
                                                     D0Layout,
                                                     ELayout>;

    decltype(ProfileGemmAddFastgeluImpl) GetImpl() override { return ProfileGemmAddFastgeluImpl; }
};

using KernelTypes = ::testing::Types<std::tuple<F16, I8, F32, F16, F16, Row, Row, Row, Row>,
                                     std::tuple<BF16, I8, F32, BF16, BF16, Row, Row, Row, Row>>;

TYPED_TEST_SUITE(TestGemmAddFastgelu, KernelTypes);
TYPED_TEST(TestGemmAddFastgelu, Test_BF16FP16) { this->Run(); }
