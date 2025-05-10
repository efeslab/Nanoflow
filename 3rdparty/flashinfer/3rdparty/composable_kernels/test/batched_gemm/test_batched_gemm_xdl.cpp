// SPDX-License-Identifier: MIT
// Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.

#include <cstdlib>
#include <iostream>
#include <initializer_list>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "profiler/profile_batched_gemm_impl.hpp"

#include "ck/library/tensor_operation_instance/gpu/batched_gemm.hpp"

struct GemmParams
{
    ck::index_t M;
    ck::index_t N;
    ck::index_t K;
    ck::index_t BatchCount;
};

class TestBatchedGemm : public ::testing::Test
{
    protected:
    using Row = ck::tensor_layout::gemm::RowMajor;
    using Col = ck::tensor_layout::gemm::ColumnMajor;

    using PassThrough = ck::tensor_operation::element_wise::PassThrough;

    std::vector<GemmParams> params;

    template <typename DataType>
    void Run()
    {
        using namespace ck::tensor_operation::device;

        bool pass = true;
        for(auto& param : params)
        {
            const auto M          = param.M;
            const auto N          = param.N;
            const auto K          = param.K;
            const auto BatchCount = param.BatchCount;

            pass =
                pass && ck::profiler::profile_batched_gemm_impl<DataType,
                                                                DataType,
                                                                DataType,
                                                                Row,
                                                                Row,
                                                                Row,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough,
                                                                DeviceBatchedGemm<Row,
                                                                                  Row,
                                                                                  Row,
                                                                                  DataType,
                                                                                  DataType,
                                                                                  DataType,
                                                                                  PassThrough,
                                                                                  PassThrough,
                                                                                  PassThrough>>(
                            true, 1, false, 1, M, N, K, K, N, N, M * K, K * N, M * N, BatchCount);

            pass =
                pass && ck::profiler::profile_batched_gemm_impl<DataType,
                                                                DataType,
                                                                DataType,
                                                                Row,
                                                                Col,
                                                                Row,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough,
                                                                DeviceBatchedGemm<Row,
                                                                                  Col,
                                                                                  Row,
                                                                                  DataType,
                                                                                  DataType,
                                                                                  DataType,
                                                                                  PassThrough,
                                                                                  PassThrough,
                                                                                  PassThrough>>(
                            true, 1, false, 1, M, N, K, K, K, N, M * K, K * N, M * N, BatchCount);

            pass =
                pass && ck::profiler::profile_batched_gemm_impl<DataType,
                                                                DataType,
                                                                DataType,
                                                                Col,
                                                                Row,
                                                                Row,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough,
                                                                DeviceBatchedGemm<Col,
                                                                                  Row,
                                                                                  Row,
                                                                                  DataType,
                                                                                  DataType,
                                                                                  DataType,
                                                                                  PassThrough,
                                                                                  PassThrough,
                                                                                  PassThrough>>(
                            true, 1, false, 1, M, N, K, M, N, N, M * K, K * N, M * N, BatchCount);

            pass =
                pass && ck::profiler::profile_batched_gemm_impl<DataType,
                                                                DataType,
                                                                DataType,
                                                                Col,
                                                                Col,
                                                                Row,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough,
                                                                DeviceBatchedGemm<Col,
                                                                                  Col,
                                                                                  Row,
                                                                                  DataType,
                                                                                  DataType,
                                                                                  DataType,
                                                                                  PassThrough,
                                                                                  PassThrough,
                                                                                  PassThrough>>(
                            true, 1, false, 1, M, N, K, M, K, N, M * K, K * N, M * N, BatchCount);
        }
        EXPECT_TRUE(pass);
    }
};

#ifdef CK_ENABLE_INT8
TEST_F(TestBatchedGemm, i8)
{
    this->params.push_back({64, 64, 64, 2});
    this->params.push_back({64, 64, 64, 1});
    this->params.push_back({60, 60, 60, 2});
    this->params.push_back({68, 68, 68, 2});
    this->params.push_back({40, 40, 40, 2});
    this->params.push_back({256, 256, 128, 3});
    this->template Run<int8_t>();
}
#endif

#ifdef CK_ENABLE_BF16
TEST_F(TestBatchedGemm, bf16)
{
    this->params.push_back({64, 64, 64, 2});
    this->params.push_back({64, 64, 64, 1});
    this->params.push_back({60, 60, 60, 2});
    this->params.push_back({68, 68, 68, 2});
    this->params.push_back({40, 40, 40, 2});
    this->params.push_back({256, 256, 128, 3});
    this->template Run<ck::bhalf_t>();
}
#endif

#ifdef CK_ENABLE_FP16
TEST_F(TestBatchedGemm, fp16)
{
    this->params.push_back({64, 64, 64, 2});
    this->params.push_back({64, 64, 64, 1});
    this->params.push_back({60, 60, 60, 2});
    this->params.push_back({68, 68, 68, 2});
    this->params.push_back({40, 40, 40, 2});
    this->params.push_back({256, 256, 128, 3});
    this->template Run<ck::half_t>();
}
#endif

#ifdef CK_ENABLE_FP32
TEST_F(TestBatchedGemm, fp32)
{
    this->params.push_back({64, 64, 64, 2});
    this->params.push_back({64, 64, 64, 1});
    this->params.push_back({60, 60, 60, 2});
    this->params.push_back({68, 68, 68, 2});
    this->params.push_back({40, 40, 40, 2});
    this->params.push_back({256, 256, 128, 3});
    this->template Run<float>();
}
#endif
