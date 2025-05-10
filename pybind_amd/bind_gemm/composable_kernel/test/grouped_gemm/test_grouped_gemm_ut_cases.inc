#pragma once

TYPED_TEST(TestGroupedGemm, TinyCases)
{
    const std::vector<int> Ms{0, 1};
    constexpr int N = 768;
    constexpr int K = 544;

    const std::vector<int> Ns(Ms.size(), N);
    const std::vector<int> Ks(Ms.size(), K);

    this->Run(Ms, Ns, Ks);
}

TYPED_TEST(TestGroupedGemm, SmallCases)
{
    const std::vector<int> Ms{2, 1, 3, 4, 5, 0};
    constexpr int N = 768;
    constexpr int K = 544;

    const std::vector<int> Ns(Ms.size(), N);
    const std::vector<int> Ks(Ms.size(), K);

    this->Run(Ms, Ns, Ks);
}

TYPED_TEST(TestGroupedGemm, MidCases)
{
    const std::vector<int> Ms{167, 183, 177, 153, 139, 204};
    constexpr int N = 768;
    constexpr int K = 544;

    const std::vector<int> Ns(Ms.size(), N);
    const std::vector<int> Ks(Ms.size(), K);

    this->Run(Ms, Ns, Ks);
}

TYPED_TEST(TestGroupedGemm, Regular)
{
    const std::vector<int> Ms{64, 128, 256};
    constexpr int N = 768;
    constexpr int K = 320;

    const std::vector<int> Ns(Ms.size(), N);
    const std::vector<int> Ks(Ms.size(), K);

    this->Run(Ms, Ns, Ks);
}

TYPED_TEST(TestGroupedGemm, MNKPadded)
{
    const std::vector<int> Ms{127, 150, 188, 210};
    constexpr int N = 136;
    constexpr int K = 280;

    const std::vector<int> Ns(Ms.size(), N);
    const std::vector<int> Ks(Ms.size(), K);

    this->Run(Ms, Ns, Ks);
}

TYPED_TEST(TestGroupedGemm, TestLargeKBatch)
{
    const std::vector<int> Ms{188, 210};
    constexpr int N = 768;
    constexpr int K = 4096;

    const std::vector<int> Ns(Ms.size(), N);
    const std::vector<int> Ks(Ms.size(), K);

    this->k_batches_ = {32, 64};

    this->Run(Ms, Ns, Ks);
}
