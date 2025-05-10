// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

namespace ck {

// Pseudo random number generator
// version for fp32
template <typename T, uint32_t seed_t, std::enable_if_t<std::is_same<float, T>{}, bool> = false>
__host__ __device__ uint32_t prand_generator(index_t id, T val, uint32_t seed = seed_t)
{
    uint32_t x         = *(reinterpret_cast<uint32_t*>(&val));
    uint32_t drop_bits = uint32_t(x) & 0xFFFFu;
    drop_bits ^= x >> 16;
    drop_bits = ((drop_bits & 31) << 11) | (drop_bits >> 5);
    drop_bits *= 0x7000149;
    // NOTE: If id is in 64 bit, we are only using lower 32 bit.
    //       So, it can have an effect of using same id for multiple elements when the id is very
    //       large!
    uint32_t rng = (drop_bits ^ 0x13371337 ^ (id * 229791) ^ seed);
    return rng;
}

// version for fp16
template <typename T, uint32_t seed_t, std::enable_if_t<std::is_same<half_t, T>{}, bool> = false>
__host__ __device__ uint32_t prand_generator(index_t id, T val, uint32_t seed = seed_t)
{
    uint16_t x         = *(reinterpret_cast<uint16_t*>(&val));
    uint32_t drop_bits = uint32_t(x) & 0xFFFFu;
    drop_bits          = ((drop_bits & 31) << 11) | (drop_bits >> 5);
    drop_bits *= 0x7000149;
    // NOTE: If id is in 64 bit, we are only using lower 32 bit.
    //       So, it can have an effect of using same id for multiple elements when the id is very
    //       large!
    uint32_t rng = (drop_bits ^ 0x13371337 ^ (id * 229791) ^ seed);
    return rng;
}

// return 0 if data is not fp16 or fp32
template <typename T,
          uint32_t seed_t,
          std::enable_if_t<!(std::is_same<float, T>{} || std::is_same<half_t, T>{}), bool> = false>
__host__ __device__ uint32_t prand_generator(int id, T val, uint32_t seed = seed_t)
{
    std::ignore = id;
    std::ignore = val;
    std::ignore = seed;

    return 0;
}

} // namespace ck
