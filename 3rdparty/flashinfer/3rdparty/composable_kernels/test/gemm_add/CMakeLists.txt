add_gtest_executable(test_gemm_add test_gemm_add_xdl.hpp)
if(result EQUAL 0)
    target_link_libraries(test_gemm_add PRIVATE utility device_gemm_add_instance)
endif()

add_gtest_executable(test_gemm_add_relu test_gemm_add_relu_xdl.cpp)
if(result EQUAL 0)
    target_link_libraries(test_gemm_add_relu PRIVATE utility device_gemm_add_instance device_gemm_add_relu_instance)
endif()

add_gtest_executable(test_gemm_add_silu test_gemm_add_silu_xdl.cpp)
if(result EQUAL 0)
    target_link_libraries(test_gemm_add_silu PRIVATE utility device_gemm_add_instance device_gemm_add_silu_instance)
endif()

add_gtest_executable(test_gemm_add_fastgelu test_gemm_add_fastgelu_xdl.cpp)
if(result EQUAL 0)
    target_link_libraries(test_gemm_add_fastgelu PRIVATE utility device_gemm_add_instance device_gemm_add_fastgelu_instance)
endif()
