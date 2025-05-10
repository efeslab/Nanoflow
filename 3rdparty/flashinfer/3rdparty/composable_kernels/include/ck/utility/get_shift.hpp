// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

namespace ck {

template <index_t N>
static constexpr __device__ index_t get_shift()
{
    return (get_shift<N / 2>() + 1);
};

template <>
constexpr __device__ index_t get_shift<1>()
{
    return (0);
}

} // namespace ck
