// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_max_pool_bwd_impl.hpp"
#include "ck/utility/data_type.hpp"

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using I32  = int32_t;
using F16  = ck::half_t;
using BF16 = ck::bhalf_t;
using F32  = float;

template <typename DOutDataType, typename IndexDataType, typename DInDataType>
using device_maxpool_bwd_instances =
    // clang-format off
    std::tuple <
        DeviceMaxPoolBwdImpl<DOutDataType, IndexDataType, DInDataType, 1>,
        DeviceMaxPoolBwdImpl<DOutDataType, IndexDataType, DInDataType, 2>,
        DeviceMaxPoolBwdImpl<DOutDataType, IndexDataType, DInDataType, 4>
               // clang-format on
               >;

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
