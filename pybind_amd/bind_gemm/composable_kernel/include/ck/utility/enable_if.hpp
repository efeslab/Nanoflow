// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

namespace ck {

#ifndef CK_CODE_GEN_RTC
template <bool B, typename T = void>
using enable_if = std::enable_if<B, T>;

template <bool B, typename T = void>
using enable_if_t = typename std::enable_if<B, T>::type;

#else
template <bool B, class T = void>
struct enable_if
{
};

template <class T>
struct enable_if<true, T>
{
    using type = T;
};

template <bool B, class T = void>
using enable_if_t = typename enable_if<B, T>::type;
#endif

} // namespace ck
