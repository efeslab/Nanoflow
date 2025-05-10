// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cstdint>
#include <unordered_set>

namespace ck {
namespace host {

std::size_t integer_divide_ceil(std::size_t x, std::size_t y);

const std::unordered_set<std::string>& get_xdlop_archs();

} // namespace host
} // namespace ck
