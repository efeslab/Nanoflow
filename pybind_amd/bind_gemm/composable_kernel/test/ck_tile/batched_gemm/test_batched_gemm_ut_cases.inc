#pragma once

TYPED_TEST(TestCkTileBatchedGemm, Basic)
{
    constexpr int M = 256;
    constexpr int N = 128;
    constexpr int K = 128;
    this->Run(M, N, K);
}
