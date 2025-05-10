
list(APPEND CMAKE_PREFIX_PATH /opt/rocm)
add_subdirectory(rtc)

file(GLOB TEST_SRCS CONFIGURE_DEPENDS *.cpp)
foreach(TEST_SRC ${TEST_SRCS})
get_filename_component(BASE_NAME ${TEST_SRC} NAME_WE)
rocm_add_test_executable(test_host_${BASE_NAME} ${TEST_SRC})
target_link_libraries(test_host_${BASE_NAME} ck_rtc ck_host)
target_include_directories(test_host_${BASE_NAME} PUBLIC include())
endforeach()
