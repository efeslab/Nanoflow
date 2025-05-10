// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include <tuple>
#include <vector>

#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/utility/data_type.hpp"

#include "gtest/gtest.h"
#include "test_grouped_gemm_util.hpp"

using F16 = ck::half_t;
using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using RRR_F16_F16_F16 = ck::test::TestGroupedGemm<std::tuple<Row, Row, Row, F16, F16, F16>>;
using RCR_F16_F16_F16 = ck::test::TestGroupedGemm<std::tuple<Row, Col, Row, F16, F16, F16>>;

using RRR_F16_F16_F16_LargeK = ck::test::TestGroupedGemm<std::tuple<Row, Row, Row, F16, F16, F16>>;
using RCR_F16_F16_F16_LargeK = ck::test::TestGroupedGemm<std::tuple<Row, Col, Row, F16, F16, F16>>;

const std::vector<int> KBATCH{1, 2, 3, 5, 8};

INSTANTIATE_TEST_SUITE_P(TestGroupedGemm_splitk_MK_KN, RRR_F16_F16_F16, testing::ValuesIn(KBATCH));
INSTANTIATE_TEST_SUITE_P(TestGroupedGemm_splitk_MK_NK, RCR_F16_F16_F16, testing::ValuesIn(KBATCH));
INSTANTIATE_TEST_SUITE_P(TestGroupedGemm_splitk_LargeK_MK_KN,
                         RRR_F16_F16_F16_LargeK,
                         testing::Values(32, 64));
INSTANTIATE_TEST_SUITE_P(TestGroupedGemm_splitk_LargeK_MK_NK,
                         RCR_F16_F16_F16_LargeK,
                         testing::Values(32, 64));

#include "test_grouped_gemm_ut_cases.inc"
