if(GPU_TARGETS MATCHES "gfx9")
    add_executable(client_conv3d_bwd_data_fp16 conv3d_bwd_data_fp16.cpp)
    add_executable(client_conv3d_bwd_data_fp32 conv3d_bwd_data_fp32.cpp)

    target_link_libraries(client_conv3d_bwd_data_fp16 PRIVATE composable_kernel::device_conv_operations)
    target_link_libraries(client_conv3d_bwd_data_fp32 PRIVATE composable_kernel::device_conv_operations)
endif()
