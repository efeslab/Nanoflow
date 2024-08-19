#pragma once
#include "cutlassGemmWrapper.cuh"
#include <string>

// The canonical name for cutlassGemmWrapper template parameters:
//  cta_m, cta_n, cta_k, warp_m, warp_n, warp_k, split_k, stages, A_major, B_major, O_major
static constexpr std::array gemmConfig = {
    "128_128_32_64_64_32_3_5_RowMajor_RowMajor_RowMajor", // O1
    "128_128_32_64_64_32_1_4_RowMajor_RowMajor_RowMajor", // O2
    "128_128_32_64_64_32_3_4_RowMajor_RowMajor_RowMajor",    // UG1
    "128_128_32_64_64_32_1_5_RowMajor_RowMajor_RowMajor",    // D1
    "128_256_32_64_64_32_1_3_RowMajor_RowMajor_RowMajor",    // UG2
    "128_128_32_64_64_32_1_4_RowMajor_RowMajor_RowMajor",    // D2
    "128_64_64_64_32_64_2_3_RowMajor_RowMajor_RowMajor",        // KQV1
    "128_128_32_64_64_32_2_5_RowMajor_RowMajor_RowMajor",       // KQV2
    "128_128_32_64_64_32_1_5_RowMajor_RowMajor_RowMajor",       // KQV3
    "128_64_64_64_32_64_2_3_RowMajor_RowMajor_RowMajor",       // KQV4
    "128_256_32_64_64_32_2_3_RowMajor_RowMajor_RowMajor",       // LOGITS1
    "128_256_32_64_64_32_2_3_RowMajor_RowMajor_RowMajor"        // LOGITS2
};

enum class GEMM_NAME {
    O1=0,
    O2,
    UG1,
    D1,
    UG2,
    D2,
    KQV1,
    KQV2,
    KQV3,
    KQV4,
    LOGITS1,
    LOGITS2,
    NUM
};

constexpr int gemmNum = static_cast<int>(GEMM_NAME::NUM);
constexpr int gemvNum = 4;