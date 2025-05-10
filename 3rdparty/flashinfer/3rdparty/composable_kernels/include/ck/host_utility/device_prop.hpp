// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <string>
#include <map>
#include <hip/hip_runtime.h>

namespace ck {

inline std::string get_device_name()
{
    hipDeviceProp_t props{};
    int device;
    auto status = hipGetDevice(&device);
    if(status != hipSuccess)
    {
        return std::string();
    }

    status = hipGetDeviceProperties(&props, device);
    if(status != hipSuccess)
    {
        return std::string();
    }
    const std::string raw_name(props.gcnArchName);

    // https://github.com/ROCm/MIOpen/blob/8498875aef84878e04c1eabefdf6571514891086/src/target_properties.cpp#L40
    static std::map<std::string, std::string> device_name_map = {
        {"Ellesmere", "gfx803"},
        {"Baffin", "gfx803"},
        {"RacerX", "gfx803"},
        {"Polaris10", "gfx803"},
        {"Polaris11", "gfx803"},
        {"Tonga", "gfx803"},
        {"Fiji", "gfx803"},
        {"gfx800", "gfx803"},
        {"gfx802", "gfx803"},
        {"gfx804", "gfx803"},
        {"Vega10", "gfx900"},
        {"gfx901", "gfx900"},
        {"10.3.0 Sienna_Cichlid 18", "gfx1030"},
    };

    const auto name = raw_name.substr(0, raw_name.find(':')); // str.substr(0, npos) returns str.

    auto match = device_name_map.find(name);
    if(match != device_name_map.end())
        return match->second;
    return name;
}

inline bool is_xdl_supported()
{
    return ck::get_device_name() == "gfx908" || ck::get_device_name() == "gfx90a" ||
           ck::get_device_name() == "gfx940" || ck::get_device_name() == "gfx941" ||
           ck::get_device_name() == "gfx942";
}

inline bool is_lds_direct_load_supported()
{
    // Check if direct loads from global memory to LDS are supported.
    return ck::get_device_name() == "gfx90a" || ck::get_device_name() == "gfx940" ||
           ck::get_device_name() == "gfx941" || ck::get_device_name() == "gfx942";
}

inline bool is_gfx101_supported()
{
    return ck::get_device_name() == "gfx1010" || ck::get_device_name() == "gfx1011" ||
           ck::get_device_name() == "gfx1012";
}

inline bool is_gfx103_supported()
{
    return ck::get_device_name() == "gfx1030" || ck::get_device_name() == "gfx1031" ||
           ck::get_device_name() == "gfx1032" || ck::get_device_name() == "gfx1034" ||
           ck::get_device_name() == "gfx1035" || ck::get_device_name() == "gfx1036";
}

inline bool is_gfx11_supported()
{
    return ck::get_device_name() == "gfx1100" || ck::get_device_name() == "gfx1101" ||
           ck::get_device_name() == "gfx1102" || ck::get_device_name() == "gfx1103";
}

} // namespace ck
