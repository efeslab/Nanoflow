// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/host_tensor.hpp"
#include <thread>

namespace ck_tile {

template <typename ADataType, typename AccDataType, typename BDataType>
CK_TILE_HOST void reference_softmax(const HostTensor<ADataType>& a_m_n,
                                    HostTensor<BDataType>& b_m_n)
{
    auto f = [&](auto m) {
        const int N = a_m_n.mDesc.get_lengths()[1];

        AccDataType v_max = ck_tile::numeric<ADataType>::Lowest();

        // max
        for(int n = 0; n < N; ++n)
        {
            const ADataType v_a = a_m_n(m, n);

            v_max = v_max < v_a ? v_a : v_max;
        }

        AccDataType v_exp_sum = 0;

        // sum
        for(int n = 0; n < N; ++n)
        {
            const ADataType v_a = a_m_n(m, n);

            v_exp_sum += ck_tile::exp(v_a - v_max);
        }

        // elementwise
        for(int n = 0; n < N; ++n)
        {
            const ADataType v_a = a_m_n(m, n);

            b_m_n(m, n) = ck_tile::exp(v_a - v_max) / v_exp_sum;
        }
    };

    make_ParallelTensorFunctor(f,
                               b_m_n.mDesc.get_lengths()[0])(std::thread::hardware_concurrency());
}
} // namespace ck_tile
