
find_package(hip)
file(GLOB RTC_SOURCES CONFIGURE_DEPENDS src/*.cpp)
add_library(ck_rtc ${RTC_SOURCES})
target_include_directories(ck_rtc PUBLIC include)
target_link_libraries(ck_rtc PUBLIC hip::host)
