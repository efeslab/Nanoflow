// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core/utility/type_traits.hpp"

namespace ck_tile {

enum class Rmsnorm2dFusedAddEnum
{
    NO_ADD = 0,
    // fused add before RMSNorm and store result to global
    PRE_ADD_STORE = 1,
    // fused add before RMSNorm, but not store result
    PRE_ADD = 2,
};

// clang-format off
template<Rmsnorm2dFusedAddEnum> struct Rmsnorm2dFusedAddEnumName;
template<> struct Rmsnorm2dFusedAddEnumName<Rmsnorm2dFusedAddEnum::NO_ADD> { static constexpr const char * name = "no"; };
template<> struct Rmsnorm2dFusedAddEnumName<Rmsnorm2dFusedAddEnum::PRE_ADD_STORE> { static constexpr const char * name = "pras"; };
template<> struct Rmsnorm2dFusedAddEnumName<Rmsnorm2dFusedAddEnum::PRE_ADD> { static constexpr const char * name = "pra"; };
// clang-format on

enum class Rmsnorm2dFusedQuantEnum
{
    NO_SWEEP             = 0,
    SMOOTH_DYNAMIC_QUANT = 1, // smooth oulier + rowwise quant, need input x-scale and store y_scale
    DYNAMIC_QUANT        = 2, // rowwise quant, store out a y-scale
};

// clang-format off
template<Rmsnorm2dFusedQuantEnum> struct Rmsnorm2dFusedQuantEnumName;
template<> struct Rmsnorm2dFusedQuantEnumName<Rmsnorm2dFusedQuantEnum::NO_SWEEP> { static constexpr const char * name = "no"; };
template<> struct Rmsnorm2dFusedQuantEnumName<Rmsnorm2dFusedQuantEnum::DYNAMIC_QUANT> { static constexpr const char * name = "dqt"; };
template<> struct Rmsnorm2dFusedQuantEnumName<Rmsnorm2dFusedQuantEnum::SMOOTH_DYNAMIC_QUANT> { static constexpr const char * name = "smdqt"; };
// clang-format on

template <bool kPadN_,
          bool kSaveInvRms_,
          bool kTwoPass_,
          Rmsnorm2dFusedAddEnum kFusedAdd_,
          Rmsnorm2dFusedQuantEnum kFusedQuant_>
struct Rmsnorm2dFwdTraits
{
    static constexpr bool kPadN                          = kPadN_;
    static constexpr bool kSaveInvRms                    = kSaveInvRms_;
    static constexpr bool kTwoPass                       = kTwoPass_;
    static constexpr Rmsnorm2dFusedAddEnum kFusedAdd     = kFusedAdd_;
    static constexpr Rmsnorm2dFusedQuantEnum kFusedQuant = kFusedQuant_;
};

} // namespace ck_tile
