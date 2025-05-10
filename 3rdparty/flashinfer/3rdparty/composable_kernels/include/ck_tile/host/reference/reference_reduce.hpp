// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/host_tensor.hpp"
#include <thread>

namespace ck_tile {

template <typename ADataType, typename AccDataType, typename BDataType>
CK_TILE_HOST void reference_reduce(const HostTensor<ADataType>& a_m_n, HostTensor<BDataType>& b_m)
{
    auto f = [&](auto m) {
        const int N = a_m_n.mDesc.get_lengths()[1];

        AccDataType v_acc = 0;

        for(int n = 0; n < N; ++n)
        {
            const ADataType v_a = a_m_n(m, n);

            v_acc += v_a;
        }

        b_m(m) = ck_tile::type_convert<BDataType>(v_acc);
    };

    make_ParallelTensorFunctor(f, b_m.mDesc.get_lengths()[0])(std::thread::hardware_concurrency());
}
} // namespace ck_tile
