#pragma once

TEST_P(RRR_F16_F16_F16, TinyCases)
{
    const std::vector<int> Ms{0, 1};
    constexpr int N = 768;
    constexpr int K = 544;

    const std::vector<int> Ns(Ms.size(), N);
    const std::vector<int> Ks(Ms.size(), K);
    const std::vector<int> StrideAs(Ms.size(), K);
    const std::vector<int> StrideBs(Ms.size(), N);
    const std::vector<int> StrideCs(Ms.size(), N);

    this->Run(Ms, Ns, Ks, StrideAs, StrideBs, StrideCs, this->GetParam());
}

TEST_P(RRR_F16_F16_F16, SmallCases)
{
    const std::vector<int> Ms{2, 1, 3, 4, 5, 0};
    constexpr int N = 768;
    constexpr int K = 544;

    const std::vector<int> Ns(Ms.size(), N);
    const std::vector<int> Ks(Ms.size(), K);
    const std::vector<int> StrideAs(Ms.size(), K);
    const std::vector<int> StrideBs(Ms.size(), N);
    const std::vector<int> StrideCs(Ms.size(), N);

    this->Run(Ms, Ns, Ks, StrideAs, StrideBs, StrideCs, this->GetParam());
}

TEST_P(RRR_F16_F16_F16, MidCases)
{
    const std::vector<int> Ms{167, 183, 177, 153, 139, 204};
    constexpr int N = 768;
    constexpr int K = 544;

    const std::vector<int> Ns(Ms.size(), N);
    const std::vector<int> Ks(Ms.size(), K);
    const std::vector<int> StrideAs(Ms.size(), K);
    const std::vector<int> StrideBs(Ms.size(), N);
    const std::vector<int> StrideCs(Ms.size(), N);

    this->Run(Ms, Ns, Ks, StrideAs, StrideBs, StrideCs, this->GetParam());
}

TEST_P(RRR_F16_F16_F16, Regular)
{
    const std::vector<int> Ms{64, 128, 256};
    constexpr int N = 768;
    constexpr int K = 320;

    const std::vector<int> Ns(Ms.size(), N);
    const std::vector<int> Ks(Ms.size(), K);
    const std::vector<int> StrideAs(Ms.size(), K);
    const std::vector<int> StrideBs(Ms.size(), N);
    const std::vector<int> StrideCs(Ms.size(), N);

    this->Run(Ms, Ns, Ks, StrideAs, StrideBs, StrideCs, this->GetParam());
}

TEST_P(RRR_F16_F16_F16, MNKPadded)
{
    const std::vector<int> Ms{127, 150, 188, 210};
    constexpr int N = 136;
    constexpr int K = 280;

    const std::vector<int> Ns(Ms.size(), N);
    const std::vector<int> Ks(Ms.size(), K);
    const std::vector<int> StrideAs(Ms.size(), K);
    const std::vector<int> StrideBs(Ms.size(), N);
    const std::vector<int> StrideCs(Ms.size(), N);

    this->Run(Ms, Ns, Ks, StrideAs, StrideBs, StrideCs, this->GetParam());
}

TEST_P(RCR_F16_F16_F16, TinyCases)
{
    const std::vector<int> Ms{0, 1};
    constexpr int N = 768;
    constexpr int K = 544;

    const std::vector<int> Ns(Ms.size(), N);
    const std::vector<int> Ks(Ms.size(), K);
    const std::vector<int> StrideAs(Ms.size(), K);
    const std::vector<int> StrideBs(Ms.size(), K);
    const std::vector<int> StrideCs(Ms.size(), N);
    this->Run(Ms, Ns, Ks, StrideAs, StrideBs, StrideCs, this->GetParam());
}

TEST_P(RCR_F16_F16_F16, SmallCases)
{
    const std::vector<int> Ms{2, 1, 3, 4, 5, 0};
    constexpr int N = 768;
    constexpr int K = 544;

    const std::vector<int> Ns(Ms.size(), N);
    const std::vector<int> Ks(Ms.size(), K);
    const std::vector<int> StrideAs(Ms.size(), K);
    const std::vector<int> StrideBs(Ms.size(), K);
    const std::vector<int> StrideCs(Ms.size(), N);

    this->Run(Ms, Ns, Ks, StrideAs, StrideBs, StrideCs, this->GetParam());
}

TEST_P(RCR_F16_F16_F16, MidCases)
{
    const std::vector<int> Ms{167, 183, 177, 153, 139, 204};
    constexpr int N = 768;
    constexpr int K = 544;

    const std::vector<int> Ns(Ms.size(), N);
    const std::vector<int> Ks(Ms.size(), K);
    const std::vector<int> StrideAs(Ms.size(), K);
    const std::vector<int> StrideBs(Ms.size(), K);
    const std::vector<int> StrideCs(Ms.size(), N);

    this->Run(Ms, Ns, Ks, StrideAs, StrideBs, StrideCs, this->GetParam());
}

TEST_P(RCR_F16_F16_F16, Regular)
{
    const std::vector<int> Ms{32, 64, 128, 256};
    constexpr int N = 768;
    constexpr int K = 320;

    const std::vector<int> Ns(Ms.size(), N);
    const std::vector<int> Ks(Ms.size(), K);
    const std::vector<int> StrideAs(Ms.size(), K);
    const std::vector<int> StrideBs(Ms.size(), K);
    const std::vector<int> StrideCs(Ms.size(), N);

    this->Run(Ms, Ns, Ks, StrideAs, StrideBs, StrideCs, this->GetParam());
}

TEST_P(RCR_F16_F16_F16, MNKPadded)
{
    const std::vector<int> Ms{127, 150, 188, 210};
    constexpr int N = 136;
    constexpr int K = 280;

    const std::vector<int> Ns(Ms.size(), N);
    const std::vector<int> Ks(Ms.size(), K);
    const std::vector<int> StrideAs(Ms.size(), K);
    const std::vector<int> StrideBs(Ms.size(), K);
    const std::vector<int> StrideCs(Ms.size(), N);

    this->Run(Ms, Ns, Ks, StrideAs, StrideBs, StrideCs, this->GetParam());
}

TEST_P(RRR_F16_F16_F16_LargeK, TestLargeKBatch)
{
    const std::vector<int> Ms{188, 210};
    constexpr int N = 768;
    constexpr int K = 4096;

    const std::vector<int> Ns(Ms.size(), N);
    const std::vector<int> Ks(Ms.size(), K);
    const std::vector<int> StrideAs(Ms.size(), K);
    const std::vector<int> StrideBs(Ms.size(), N);
    const std::vector<int> StrideCs(Ms.size(), N);

    this->Run(Ms, Ns, Ks, StrideAs, StrideBs, StrideCs, this->GetParam());
}

TEST_P(RCR_F16_F16_F16_LargeK, TestLargeKBatch)
{
    const std::vector<int> Ms{188, 210};
    constexpr int N = 768;
    constexpr int K = 4096;

    const std::vector<int> Ns(Ms.size(), N);
    const std::vector<int> Ks(Ms.size(), K);
    const std::vector<int> StrideAs(Ms.size(), K);
    const std::vector<int> StrideBs(Ms.size(), K);
    const std::vector<int> StrideCs(Ms.size(), N);

    this->Run(Ms, Ns, Ks, StrideAs, StrideBs, StrideCs, this->GetParam());
}
