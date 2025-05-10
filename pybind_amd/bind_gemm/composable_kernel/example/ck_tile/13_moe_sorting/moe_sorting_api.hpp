// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once
#include <string>
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/fused_moe.hpp"

struct moe_sorting_trait
{
    std::string index_type;
    std::string weight_type; // currently always float
};

struct moe_sorting_args : public ck_tile::MoeSortingHostArgs
{
};

float moe_sorting(moe_sorting_trait t, moe_sorting_args a, ck_tile::stream_config s);
