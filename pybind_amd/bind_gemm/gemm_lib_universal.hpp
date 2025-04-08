#pragma once

#include <vector>
#include <string>

/**
 * \brief Query all CK GEMM type strings for half-precision (fp16).
 * \return A list of strings describing each available CK gemm kernel instance.
 */
std::vector<std::string> get_all_typestrings_universal();

/**
 * \brief Perform a single half-precision GEMM using CK's matching instance for the given type_string.
 *
 *  - A: [M, K], row-major, leading dim = K
 *  - B: [K, N], row-major, leading dim = N
 *  - C: [M, N], row-major, leading dim = N
 *
 * \param M, N, K   GEMM dimensions
 * \param d_A, d_B, d_C  GPU pointers to half data (ck::half_t)
 * \param type_string The CK "type string" identifying the specific GEMM kernel
 * \return true on success, false if unsupported or not found
 */
bool gemm_fp16_universal(int M, int N, int K,
               const void* d_A,
               const void* d_B,
               void*       d_C,
               const std::string& type_string);
