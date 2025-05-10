 # XDL_DL_WMMA_KERNELS
set(GROUPED_CONV3D_BWD_WEIGHT
     xdl/device_grouped_conv3d_bwd_weight_xdl_gndhwc_gkzyxc_gndhwk_f16_instance.cpp
     xdl/device_grouped_conv3d_bwd_weight_xdl_gndhwc_gkzyxc_gndhwk_f32_instance.cpp
     xdl/device_grouped_conv3d_bwd_weight_xdl_gndhwc_gkzyxc_gndhwk_bf16_instance.cpp
     xdl/device_grouped_conv3d_bwd_weight_xdl_ndhwgc_gkzyxc_ndhwgk_f16_instance.cpp
     xdl/device_grouped_conv3d_bwd_weight_xdl_ndhwgc_gkzyxc_ndhwgk_f32_instance.cpp
     xdl/device_grouped_conv3d_bwd_weight_xdl_ndhwgc_gkzyxc_ndhwgk_bf16_instance.cpp
    xdl/device_grouped_conv3d_bwd_weight_two_stage_xdl_ndhwgc_gkzyxc_ndhwgk_f16_pipev2_instance.cpp
    xdl/device_grouped_conv3d_bwd_weight_two_stage_xdl_ndhwgc_gkzyxc_ndhwgk_f16_pipev5_instance.cpp
    )

if(DL_KERNELS)
    list(APPEND GROUPED_CONV3D_BWD_WEIGHT
        dl/device_grouped_conv3d_bwd_weight_dl_gndhwc_gkzyxc_gndhwk_f16_instance.cpp
        dl/device_grouped_conv3d_bwd_weight_dl_gndhwc_gkzyxc_gndhwk_f32_instance.cpp
        dl/device_grouped_conv3d_bwd_weight_dl_gndhwc_gkzyxc_gndhwk_bf16_instance.cpp
        dl/device_grouped_conv3d_bwd_weight_dl_ndhwgc_gkzyxc_ndhwgk_f16_instance.cpp
        dl/device_grouped_conv3d_bwd_weight_dl_ndhwgc_gkzyxc_ndhwgk_f32_instance.cpp
        dl/device_grouped_conv3d_bwd_weight_dl_ndhwgc_gkzyxc_ndhwgk_bf16_instance.cpp)
endif()

list(APPEND GROUPED_CONV3D_BWD_WEIGHT
    wmma/device_grouped_conv3d_bwd_weight_wmma_gndhwc_gkzyxc_gndhwk_f16_1x1s1p0_instance.cpp
    wmma/device_grouped_conv3d_bwd_weight_wmma_ndhwgc_gkzyxc_ndhwgk_f16_1x1s1p0_instance.cpp
    wmma/device_grouped_conv3d_bwd_weight_wmma_gndhwc_gkzyxc_gndhwk_f16_instance.cpp
    wmma/device_grouped_conv3d_bwd_weight_wmma_ndhwgc_gkzyxc_ndhwgk_f16_instance.cpp
    wmma/device_grouped_conv3d_bwd_weight_wmma_gndhwc_gkzyxc_gndhwk_i8_1x1s1p0_instance.cpp
    wmma/device_grouped_conv3d_bwd_weight_wmma_ndhwgc_gkzyxc_ndhwgk_i8_1x1s1p0_instance.cpp
    wmma/device_grouped_conv3d_bwd_weight_wmma_gndhwc_gkzyxc_gndhwk_i8_instance.cpp
    wmma/device_grouped_conv3d_bwd_weight_wmma_ndhwgc_gkzyxc_ndhwgk_i8_instance.cpp)

if((DTYPES MATCHES "fp8" AND DTYPES MATCHES "bf8" AND DTYPES MATCHES "fp16") OR NOT DEFINED DTYPES)
    list(APPEND GROUPED_CONV3D_BWD_WEIGHT
      xdl/device_grouped_conv3d_bwd_weight_xdl_ndhwgc_gkzyxc_ndhwgk_f16_comp_bf8_fp8_instance.cpp)
endif()

add_instance_library(device_grouped_conv3d_bwd_weight_instance ${GROUPED_CONV3D_BWD_WEIGHT})
