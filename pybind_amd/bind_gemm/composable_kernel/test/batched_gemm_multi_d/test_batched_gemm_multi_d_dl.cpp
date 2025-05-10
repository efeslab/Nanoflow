// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <gtest/gtest.h>

#include "profiler/profile_batched_gemm_impl.hpp"
#include "ck/library/tensor_operation_instance/gpu/batched_gemm_multi_d.hpp"

namespace {
using F16 = ck::half_t;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using Empty_Tuple = ck::Tuple<>;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

template <typename Tuple>
class TestBatchedGemmMultiD : public ::testing::Test
{
    protected:
    using ALayout = std::tuple_element_t<0, Tuple>;
    using BLayout = std::tuple_element_t<1, Tuple>;
    using CLayout = std::tuple_element_t<2, Tuple>;

    static constexpr int M          = 512;
    static constexpr int N          = 256;
    static constexpr int K          = 128;
    static constexpr int BatchCount = 3;

    template <typename DataType>
    void Run()
    {
        using namespace ck::tensor_operation::device;

        const bool pass =
            ck::profiler::profile_batched_gemm_impl<DataType,
                                                    DataType,
                                                    DataType,
                                                    ALayout,
                                                    BLayout,
                                                    CLayout,
                                                    PassThrough,
                                                    PassThrough,
                                                    PassThrough,
                                                    DeviceBatchedGemmMultiD<ALayout,
                                                                            BLayout,
                                                                            Empty_Tuple,
                                                                            CLayout,
                                                                            DataType,
                                                                            DataType,
                                                                            Empty_Tuple,
                                                                            DataType,
                                                                            PassThrough,
                                                                            PassThrough,
                                                                            PassThrough>>(
                true, 1, false, 1, M, N, K, K, N, N, M * K, K * N, M * N, BatchCount);
        EXPECT_TRUE(pass);
    }
};

using KernelTypes = ::testing::Types<std::tuple<Row, Row, Row>,
                                     std::tuple<Row, Col, Row>,
                                     std::tuple<Col, Row, Row>,
                                     std::tuple<Col, Col, Row>>;
} // namespace

TYPED_TEST_SUITE(TestBatchedGemmMultiD, KernelTypes);
#ifdef __fp16
TYPED_TEST(TestBatchedGemmMultiD, f16) { this->template Run<F16>(); }
#endif
#ifdef CK_ENABLE_INT8
TYPED_TEST(TestBatchedGemmMultiD, int8) { this->template Run<int8_t>(); }
#endif
