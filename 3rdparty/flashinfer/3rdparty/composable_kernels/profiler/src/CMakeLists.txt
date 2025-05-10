# ckProfiler
set(PROFILER_SOURCES
    profiler.cpp
    profile_gemm.cpp
    profile_reduce.cpp
    profile_groupnorm_bwd_data.cpp
    profile_groupnorm_fwd.cpp
    profile_layernorm_bwd_data.cpp
    profile_layernorm_bwd_gamma_beta.cpp
    profile_groupnorm_bwd_gamma_beta.cpp
    profile_layernorm_fwd.cpp
    profile_max_pool3d_fwd.cpp
    profile_avg_pool3d_bwd.cpp
    profile_max_pool3d_bwd.cpp
    profile_softmax.cpp
    profile_batchnorm_fwd.cpp
    profile_batchnorm_bwd.cpp
    profile_batchnorm_infer.cpp
    profile_conv_tensor_rearrange.cpp
    profile_transpose.cpp
    profile_permute_scale.cpp
)

if(GPU_TARGETS MATCHES "gfx9")
  if(DTYPES MATCHES "fp32" OR DTYPES MATCHES "fp64" OR NOT DEFINED DTYPES)
    list(APPEND PROFILER_SOURCES profile_contraction_bilinear.cpp)
    list(APPEND PROFILER_SOURCES profile_contraction_scale.cpp)
  endif()
  if(DTYPES MATCHES "fp16" OR NOT DEFINED DTYPES)
    list(APPEND PROFILER_SOURCES profile_gemm_reduce.cpp)
    list(APPEND PROFILER_SOURCES profile_batched_gemm_gemm.cpp)
    list(APPEND PROFILER_SOURCES profile_batched_gemm_add_relu_gemm_add.cpp)
    list(APPEND PROFILER_SOURCES profile_gemm_add.cpp)
    list(APPEND PROFILER_SOURCES profile_gemm_add_add_fastgelu.cpp)
    list(APPEND PROFILER_SOURCES profile_gemm_add_fastgelu.cpp)
    list(APPEND PROFILER_SOURCES profile_grouped_gemm.cpp)
    list(APPEND PROFILER_SOURCES profile_gemm_streamk.cpp)
    list(APPEND PROFILER_SOURCES profile_gemm_fastgelu.cpp)
    list(APPEND PROFILER_SOURCES profile_gemm_add_relu.cpp)
    list(APPEND PROFILER_SOURCES profile_gemm_add_silu.cpp)
    list(APPEND PROFILER_SOURCES profile_gemm_add_relu_add_layernorm.cpp)
    list(APPEND PROFILER_SOURCES profile_grouped_gemm_fixed_nk.cpp)
    list(APPEND PROFILER_SOURCES profile_grouped_gemm_two_stage.cpp)
    list(APPEND PROFILER_SOURCES profile_grouped_gemm_fastgelu.cpp)
    list(APPEND PROFILER_SOURCES profile_grouped_gemm_tile_loop.cpp)
  endif()
  list(APPEND PROFILER_SOURCES profile_gemm_multiply_add.cpp)
  list(APPEND PROFILER_SOURCES profile_batched_gemm.cpp)
  list(APPEND PROFILER_SOURCES profile_batched_gemm_reduce.cpp)
  list(APPEND PROFILER_SOURCES profile_gemm_add_multiply.cpp)
  list(APPEND PROFILER_SOURCES profile_gemm_bias_add_reduce.cpp)
  list(APPEND PROFILER_SOURCES profile_gemm_splitk.cpp)
  list(APPEND PROFILER_SOURCES profile_gemm_universal.cpp)
  list(APPEND PROFILER_SOURCES profile_conv_fwd_bias_relu.cpp)
  list(APPEND PROFILER_SOURCES profile_conv_fwd_bias_relu_add.cpp)
  list(APPEND PROFILER_SOURCES profile_conv_bwd_data.cpp)
  list(APPEND PROFILER_SOURCES profile_conv_fwd.cpp)

endif()

if(GPU_TARGETS MATCHES "gfx11" OR GPU_TARGETS MATCHES "gfx9")
  if(DTYPES MATCHES "fp16" OR NOT DEFINED DTYPES)
    list(APPEND PROFILER_SOURCES profile_gemm_bilinear.cpp)
  endif()
  list(APPEND PROFILER_SOURCES profile_grouped_conv_fwd.cpp)
  list(APPEND PROFILER_SOURCES profile_grouped_conv_bwd_data.cpp)
  list(APPEND PROFILER_SOURCES profile_grouped_conv_bwd_weight.cpp)
endif()

if(DL_KERNELS)
  list(APPEND PROFILER_SOURCES profile_batched_gemm_multi_d.cpp)
  list(APPEND PROFILER_SOURCES profile_grouped_conv_bwd_weight.cpp)
endif()

set(PROFILER_EXECUTABLE ckProfiler)

add_executable(${PROFILER_EXECUTABLE} ${PROFILER_SOURCES})
target_compile_options(${PROFILER_EXECUTABLE} PRIVATE -Wno-global-constructors)

target_link_libraries(${PROFILER_EXECUTABLE} PRIVATE utility getopt::getopt)
target_link_libraries(${PROFILER_EXECUTABLE} PRIVATE device_gemm_instance)
target_link_libraries(${PROFILER_EXECUTABLE} PRIVATE device_normalization_fwd_instance)
target_link_libraries(${PROFILER_EXECUTABLE} PRIVATE device_normalization_bwd_data_instance)
target_link_libraries(${PROFILER_EXECUTABLE} PRIVATE device_normalization_bwd_gamma_beta_instance)
target_link_libraries(${PROFILER_EXECUTABLE} PRIVATE device_softmax_instance)
target_link_libraries(${PROFILER_EXECUTABLE} PRIVATE device_reduce_instance)
target_link_libraries(${PROFILER_EXECUTABLE} PRIVATE device_batchnorm_instance)
target_link_libraries(${PROFILER_EXECUTABLE} PRIVATE device_pool3d_fwd_instance)
target_link_libraries(${PROFILER_EXECUTABLE} PRIVATE device_avg_pool3d_bwd_instance)
target_link_libraries(${PROFILER_EXECUTABLE} PRIVATE device_max_pool_bwd_instance)
target_link_libraries(${PROFILER_EXECUTABLE} PRIVATE device_image_to_column_instance)
target_link_libraries(${PROFILER_EXECUTABLE} PRIVATE device_column_to_image_instance)
target_link_libraries(${PROFILER_EXECUTABLE} PRIVATE device_transpose_instance)
target_link_libraries(${PROFILER_EXECUTABLE} PRIVATE device_permute_scale_instance)

if(GPU_TARGETS MATCHES "gfx9")
  if(DTYPES MATCHES "fp32" OR DTYPES MATCHES "fp64" OR NOT DEFINED DTYPES)
    target_link_libraries(${PROFILER_EXECUTABLE} PRIVATE device_contraction_bilinear_instance)
    target_link_libraries(${PROFILER_EXECUTABLE} PRIVATE device_contraction_scale_instance)
  endif()
  if(DTYPES MATCHES "fp16" OR NOT DEFINED DTYPES)
    target_link_libraries(${PROFILER_EXECUTABLE} PRIVATE device_gemm_add_instance)
    target_link_libraries(${PROFILER_EXECUTABLE} PRIVATE device_gemm_add_add_fastgelu_instance)
    target_link_libraries(${PROFILER_EXECUTABLE} PRIVATE device_gemm_fastgelu_instance)
    target_link_libraries(${PROFILER_EXECUTABLE} PRIVATE device_batched_gemm_gemm_instance)
    target_link_libraries(${PROFILER_EXECUTABLE} PRIVATE device_batched_gemm_add_relu_gemm_add_instance)
    target_link_libraries(${PROFILER_EXECUTABLE} PRIVATE device_grouped_gemm_instance)
    target_link_libraries(${PROFILER_EXECUTABLE} PRIVATE device_gemm_streamk_instance)
    target_link_libraries(${PROFILER_EXECUTABLE} PRIVATE device_gemm_add_fastgelu_instance)
    target_link_libraries(${PROFILER_EXECUTABLE} PRIVATE device_gemm_add_relu_instance)
    target_link_libraries(${PROFILER_EXECUTABLE} PRIVATE device_gemm_add_silu_instance)
    target_link_libraries(${PROFILER_EXECUTABLE} PRIVATE device_gemm_add_relu_add_layernorm_instance)
    target_link_libraries(${PROFILER_EXECUTABLE} PRIVATE device_grouped_gemm_fixed_nk_instance)
    target_link_libraries(${PROFILER_EXECUTABLE} PRIVATE device_grouped_gemm_fastgelu_instance)
    target_link_libraries(${PROFILER_EXECUTABLE} PRIVATE device_grouped_gemm_tile_loop_instance)
  endif()
  target_link_libraries(${PROFILER_EXECUTABLE} PRIVATE device_batched_gemm_instance)
  target_link_libraries(${PROFILER_EXECUTABLE} PRIVATE device_batched_gemm_reduce_instance)
  target_link_libraries(${PROFILER_EXECUTABLE} PRIVATE device_gemm_multiply_add_instance)
  target_link_libraries(${PROFILER_EXECUTABLE} PRIVATE device_gemm_splitk_instance)
  target_link_libraries(${PROFILER_EXECUTABLE} PRIVATE device_gemm_universal_instance)
  target_link_libraries(${PROFILER_EXECUTABLE} PRIVATE device_gemm_add_multiply_instance)
  target_link_libraries(${PROFILER_EXECUTABLE} PRIVATE device_gemm_reduce_instance)
  target_link_libraries(${PROFILER_EXECUTABLE} PRIVATE device_gemm_bias_add_reduce_instance)
  target_link_libraries(${PROFILER_EXECUTABLE} PRIVATE device_conv2d_fwd_instance)
  target_link_libraries(${PROFILER_EXECUTABLE} PRIVATE device_conv2d_fwd_bias_relu_instance)
  target_link_libraries(${PROFILER_EXECUTABLE} PRIVATE device_conv2d_fwd_bias_relu_add_instance)
  target_link_libraries(${PROFILER_EXECUTABLE} PRIVATE device_grouped_conv1d_fwd_instance)
  target_link_libraries(${PROFILER_EXECUTABLE} PRIVATE device_conv1d_bwd_data_instance)
  target_link_libraries(${PROFILER_EXECUTABLE} PRIVATE device_conv3d_bwd_data_instance)
  target_link_libraries(${PROFILER_EXECUTABLE} PRIVATE device_conv2d_bwd_data_instance)
  target_link_libraries(${PROFILER_EXECUTABLE} PRIVATE device_grouped_conv1d_bwd_weight_instance)
  target_link_libraries(${PROFILER_EXECUTABLE} PRIVATE device_grouped_conv2d_bwd_weight_instance)
endif()

if(GPU_TARGETS MATCHES "gfx9" OR GPU_TARGETS MATCHES "gfx11")
  if(DTYPES MATCHES "fp16" OR NOT DEFINED DTYPES)
    target_link_libraries(${PROFILER_EXECUTABLE} PRIVATE device_gemm_bilinear_instance)
  endif()
  target_link_libraries(${PROFILER_EXECUTABLE} PRIVATE device_grouped_conv3d_fwd_instance)
  target_link_libraries(${PROFILER_EXECUTABLE} PRIVATE device_grouped_conv2d_bwd_data_instance)
  target_link_libraries(${PROFILER_EXECUTABLE} PRIVATE device_grouped_conv3d_bwd_data_instance)
  target_link_libraries(${PROFILER_EXECUTABLE} PRIVATE device_grouped_conv2d_fwd_instance)
  target_link_libraries(${PROFILER_EXECUTABLE} PRIVATE device_grouped_conv3d_bwd_weight_instance)
endif()

if(DL_KERNELS)
  target_link_libraries(${PROFILER_EXECUTABLE} PRIVATE device_batched_gemm_multi_d_instance)
  target_link_libraries(${PROFILER_EXECUTABLE} PRIVATE device_grouped_conv1d_bwd_weight_instance)
  target_link_libraries(${PROFILER_EXECUTABLE} PRIVATE device_grouped_conv2d_bwd_weight_instance)
  target_link_libraries(${PROFILER_EXECUTABLE} PRIVATE device_grouped_conv3d_bwd_weight_instance)
endif()

rocm_install(TARGETS ${PROFILER_EXECUTABLE} COMPONENT profiler)
