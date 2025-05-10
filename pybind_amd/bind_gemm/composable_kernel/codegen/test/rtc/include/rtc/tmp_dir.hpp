// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#ifndef GUARD_HOST_TEST_RTC_INCLUDE_RTC_TMP_DIR
#define GUARD_HOST_TEST_RTC_INCLUDE_RTC_TMP_DIR

#include <string>
#include <rtc/filesystem.hpp>

namespace rtc {

struct tmp_dir
{
    fs::path path;
    tmp_dir(const std::string& prefix = "");

    void execute(const std::string& cmd) const;

    tmp_dir(tmp_dir const&) = delete;
    tmp_dir& operator=(tmp_dir const&) = delete;

    ~tmp_dir();
};

} // namespace rtc

#endif
