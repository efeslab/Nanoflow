// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <string>
#include <string_view>
#include <utility>
#include <unordered_map>
#include <vector>

namespace ck {
namespace host {

std::unordered_map<std::string_view, std::string_view> GetHeaders();

} // namespace host
} // namespace ck
