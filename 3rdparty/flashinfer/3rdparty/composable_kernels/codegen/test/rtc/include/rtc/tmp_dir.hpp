#ifndef GUARD_HOST_TEST_RTC_INCLUDE_RTC_TMP_DIR
#define GUARD_HOST_TEST_RTC_INCLUDE_RTC_TMP_DIR

#include <string>
#include <filesystem>

namespace rtc {

struct tmp_dir
{
    std::filesystem::path path;
    tmp_dir(const std::string& prefix = "");

    void execute(const std::string& cmd) const;

    tmp_dir(tmp_dir const&) = delete;
    tmp_dir& operator=(tmp_dir const&) = delete;

    ~tmp_dir();
};

} // namespace rtc

#endif
