add_test_executable(test_gemm_reduce_fp16 gemm_reduce_fp16_xdl.cpp)
if(result EQUAL 0)
  target_link_libraries(test_gemm_reduce_fp16 PRIVATE utility device_gemm_reduce_instance)
endif()