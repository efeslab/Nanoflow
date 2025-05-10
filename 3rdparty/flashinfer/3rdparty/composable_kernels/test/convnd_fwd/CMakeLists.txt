add_gtest_executable(test_convnd_fwd convnd_fwd_xdl.cpp)
if(result EQUAL 0)
   target_link_libraries(test_convnd_fwd PRIVATE utility device_conv2d_fwd_instance)
endif()
