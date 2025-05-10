// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

// Default policy for BlockGemmPipelineAGmemBGmemCRegV2
// Default policy class should not be templated, put template on member functions instead
// NOTE: policy should be binded to its corresponding operation. It's just a coincidence that
//   BlockGemmPipelineAGmemBGmemCRegV2DefaultPolicy is the same as
//   BlockGemmPipelineAGmemBGmemCRegV1DefaultPolicy
using BlockGemmPipelineAGmemBGmemCRegV2DefaultPolicy =
    BlockGemmPipelineAGmemBGmemCRegV1DefaultPolicy;

} // namespace ck_tile
