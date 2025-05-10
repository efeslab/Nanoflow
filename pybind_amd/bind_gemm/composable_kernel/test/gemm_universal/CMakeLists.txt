add_gtest_executable(test_gemm_universal test_gemm_universal_xdl.cpp)
if(result EQUAL 0)
   target_link_libraries(test_gemm_universal PRIVATE utility device_gemm_universal_instance)
 endif()
