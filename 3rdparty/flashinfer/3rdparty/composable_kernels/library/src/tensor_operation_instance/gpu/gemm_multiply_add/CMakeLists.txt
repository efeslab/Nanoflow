# ONLY XDL_KERNELS
set(GEMM_MULTIPLY_ADD_INSTANCES)
list(APPEND GEMM_MULTIPLY_ADD_INSTANCES device_gemm_multiply_add_xdl_c_shuffle_f16_f16_f16_f16_f16_mk_kn_mn_mn_mn_instance.cpp
                                        device_gemm_multiply_add_xdl_c_shuffle_f16_f16_f16_f16_f16_mk_nk_mn_mn_mn_instance.cpp
                                        device_gemm_multiply_add_xdl_c_shuffle_f16_f8_f32_f32_f16_mk_kn_mn_mn_mn_instance.cpp
                                        device_gemm_multiply_add_xdl_c_shuffle_f16_f8_f32_f32_f16_mk_nk_mn_mn_mn_instance.cpp)
add_instance_library(device_gemm_multiply_add_instance ${GEMM_MULTIPLY_ADD_INSTANCES})
