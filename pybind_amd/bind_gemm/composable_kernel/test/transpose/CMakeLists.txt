add_gtest_executable(test_transpose test_transpose_xdl.cpp)
if(result EQUAL 0)
    target_link_libraries(test_transpose PRIVATE utility device_transpose_instance)
endif()
