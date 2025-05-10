// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#ifndef CK_CODE_GEN_RTC
#include <ostream>
#endif

#pragma once

#include "ck/utility/common_header.hpp"

namespace ck {

enum struct LoopScheduler
{
    Default,
    Interwave,
};

constexpr LoopScheduler make_default_loop_scheduler()
{
#if CK_EXPERIMENTAL_DEFAULT_TO_INTER_WAVE_SCHEDULING
    return LoopScheduler::Interwave;
#else
    return LoopScheduler::Default;
#endif // if CK_EXPERIMENTAL_DEFAULT_TO_INTER_WAVE_SCHEDULING
}

} // namespace ck

#ifndef CK_CODE_GEN_RTC
inline std::ostream& operator<<(std::ostream& os, const ck::LoopScheduler& s)
{
    switch(s)
    {
    case ck::LoopScheduler::Default: os << "Default"; break;
    case ck::LoopScheduler::Interwave: os << "Interwave"; break;
    default: os << "";
    }
    return os;
}
#endif
