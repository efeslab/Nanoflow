add_gtest_executable(test_batched_gemm test_batched_gemm_xdl.cpp)
if(result EQUAL 0)
   target_link_libraries(test_batched_gemm PRIVATE utility device_batched_gemm_instance)
endif()
