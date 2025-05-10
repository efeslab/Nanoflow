// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#ifndef GUARD_HOST_TEST_RTC_INCLUDE_RTC_COMPILE_KERNEL
#define GUARD_HOST_TEST_RTC_INCLUDE_RTC_COMPILE_KERNEL

#include <rtc/kernel.hpp>
#include <rtc/filesystem.hpp>
#include <string>

namespace rtc {

struct src_file
{
    fs::path path;
    std::string_view content;
};

struct compile_options
{
    std::string flags       = "";
    std::string kernel_name = "main";
};

kernel compile_kernel(const std::vector<src_file>& src,
                      compile_options options = compile_options{});

} // namespace rtc

#endif
