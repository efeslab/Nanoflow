// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

namespace ck_tile {

// This class is used for codegen pattern matching
enum class BlockFmhaBwdPipelineEnum
{
    KRKTRVR_IGLP = 0,
    KRKTRVR,
};

} // namespace ck_tile
