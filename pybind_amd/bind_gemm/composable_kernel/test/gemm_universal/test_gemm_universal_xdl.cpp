// SPDX-License-Identifier: MIT
// Copyright (c) 2023-2024, Advanced Micro Devices, Inc. All rights reserved.

#include <tuple>

#include "gtest/gtest.h"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "test_gemm_universal_util.hpp"

using F8   = ck::f8_t;
using F16  = ck::half_t;
using BF16 = ck::bhalf_t;
using F32  = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

namespace {

template <typename X, typename Y>
struct tuple_concat;

template <typename... Xs, typename... Ys>
struct tuple_concat<std::tuple<Xs...>, std::tuple<Ys...>>
{
    using type = std::tuple<Xs..., Ys...>;
};

} // namespace

template <typename Tuple>
class TestGemmUniversal_MK_KN
    : public ck::test::TestGemmUniversal<typename tuple_concat<std::tuple<Row, Row>, Tuple>::type>
{
};

template <typename Tuple>
class TestGemmUniversal_MK_NK
    : public ck::test::TestGemmUniversal<typename tuple_concat<std::tuple<Row, Col>, Tuple>::type>
{
};

template <typename Tuple>
class TestGemmUniversal_KM_KN
    : public ck::test::TestGemmUniversal<typename tuple_concat<std::tuple<Col, Row>, Tuple>::type>
{
};

template <typename Tuple>
class TestGemmUniversal_KM_NK
    : public ck::test::TestGemmUniversal<typename tuple_concat<std::tuple<Col, Col>, Tuple>::type>
{
};

// clang-format off
using KernelTypes_MK_KN = ::testing::Types<
    //         ADataType, BDataType, ComputeDataType, CDataType
    std::tuple<      F16,       F16,             F16,     F16>,
#if defined(CK_ENABLE_FP8) && (defined(CK_USE_FP8_ON_UNSUPPORTED_ARCH) || defined(CK_USE_GFX94))
    std::tuple<      F16,        F8,             F16,     F16>,
    std::tuple<       F8,       F16,             F16,     F16>,
    std::tuple<       F8,        F8,              F8,    BF16>,
#endif
    std::tuple<     BF16,      BF16,            BF16,    BF16>
    >;
using KernelTypes_MK_NK = ::testing::Types<
    //         ADataType, BDataType, ComputeDataType, CDataType
    std::tuple<      F16,       F16,             F16,     F16>,
#if defined(CK_ENABLE_FP8) && (defined(CK_USE_FP8_ON_UNSUPPORTED_ARCH) || defined(CK_USE_GFX94))
    std::tuple<      F16,        F8,             F16,     F16>,
    std::tuple<       F8,       F16,             F16,     F16>,
    std::tuple<       F8,        F8,              F8,    BF16>,
#endif
    std::tuple<     BF16,      BF16,            BF16,    BF16>
    >;

using KernelTypes_KM_NK = ::testing::Types<
    //         ADataType, BDataType, ComputeDataType, CDataType
    std::tuple<     BF16,      BF16,            BF16,    BF16>
    >;

using KernelTypes_KM_KN = ::testing::Types<
    //         ADataType, BDataType, ComputeDataType, CDataType
    std::tuple<     BF16,      BF16,            BF16,    BF16>
    >;

// clang-format on

TYPED_TEST_SUITE(TestGemmUniversal_MK_KN, KernelTypes_MK_KN);
TYPED_TEST_SUITE(TestGemmUniversal_MK_NK, KernelTypes_MK_NK);
TYPED_TEST_SUITE(TestGemmUniversal_KM_KN, KernelTypes_KM_KN);
TYPED_TEST_SUITE(TestGemmUniversal_KM_NK, KernelTypes_KM_NK);

#include "test_gemm_universal_ut_cases.inc"
