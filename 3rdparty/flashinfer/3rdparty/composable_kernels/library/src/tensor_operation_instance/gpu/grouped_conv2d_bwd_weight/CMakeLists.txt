# ONLY XDL_AND_DL_KERNELS
set(GROUPED_CONV2D_BWD_WEIGHT
    xdl/device_grouped_conv2d_bwd_weight_xdl_gnhwc_gkyxc_gnhwk_f16_instance.cpp
    xdl/device_grouped_conv2d_bwd_weight_xdl_gnhwc_gkyxc_gnhwk_f32_instance.cpp
    xdl/device_grouped_conv2d_bwd_weight_xdl_gnhwc_gkyxc_gnhwk_bf16_instance.cpp
    xdl/device_grouped_conv2d_bwd_weight_xdl_nhwgc_gkyxc_nhwgk_f16_instance.cpp
    xdl/device_grouped_conv2d_bwd_weight_xdl_nhwgc_gkyxc_nhwgk_f32_instance.cpp
    xdl/device_grouped_conv2d_bwd_weight_xdl_nhwgc_gkyxc_nhwgk_bf16_instance.cpp
    xdl/device_grouped_conv2d_bwd_weight_two_stage_xdl_nhwgc_gkyxc_nhwgk_f16_pipev2_instance.cpp
    xdl/device_grouped_conv2d_bwd_weight_two_stage_xdl_nhwgc_gkyxc_nhwgk_f16_pipev5_instance.cpp
    )

if(DL_KERNELS)
    list(APPEND GROUPED_CONV2D_BWD_WEIGHT
        dl/device_grouped_conv2d_bwd_weight_dl_gnhwc_gkyxc_gnhwk_f16_instance.cpp
        dl/device_grouped_conv2d_bwd_weight_dl_gnhwc_gkyxc_gnhwk_f32_instance.cpp
        dl/device_grouped_conv2d_bwd_weight_dl_gnhwc_gkyxc_gnhwk_bf16_instance.cpp
        dl/device_grouped_conv2d_bwd_weight_dl_nhwgc_gkyxc_nhwgk_f16_instance.cpp
        dl/device_grouped_conv2d_bwd_weight_dl_nhwgc_gkyxc_nhwgk_f32_instance.cpp
        dl/device_grouped_conv2d_bwd_weight_dl_nhwgc_gkyxc_nhwgk_bf16_instance.cpp)
endif()

add_instance_library(device_grouped_conv2d_bwd_weight_instance ${GROUPED_CONV2D_BWD_WEIGHT})
