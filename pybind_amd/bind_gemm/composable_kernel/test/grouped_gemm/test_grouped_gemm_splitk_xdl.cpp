// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#include <tuple>
#include <vector>

#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/utility/data_type.hpp"

#include "gtest/gtest.h"
#include "test_grouped_gemm_util.hpp"

using F16  = ck::half_t;
using BF16 = ck::bhalf_t;
using F8   = ck::f8_t;
using I8   = int8_t;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

template <typename Tuple>
class TestGroupedGemm : public ck::test::TestGroupedGemm<Tuple>
{
};

// clang-format off
using KernelTypes = ::testing::Types<
    std::tuple<     Row, Row, Row, F16, F16, F16>,
    std::tuple<     Row, Col, Row, F16, F16, F16>,
    std::tuple<     Col, Row, Row, F16, F16, F16>,
    std::tuple<     Col, Col, Row, F16, F16, F16>,
    std::tuple<     Row, Row, Row, BF16, BF16, BF16>,
    std::tuple<     Row, Col, Row, BF16, BF16, BF16>,
    std::tuple<     Col, Row, Row, BF16, BF16, BF16>,
    std::tuple<     Row, Row, Row, BF16, I8, BF16>,
    std::tuple<     Row, Col, Row, BF16, I8, BF16>,
    std::tuple<     Row, Row, Row, F16, F8, F16>,
    std::tuple<     Row, Row, Row, F8, F16, F16>
    >;
// clang-format on

TYPED_TEST_SUITE(TestGroupedGemm, KernelTypes);

#include "test_grouped_gemm_ut_cases.inc"
