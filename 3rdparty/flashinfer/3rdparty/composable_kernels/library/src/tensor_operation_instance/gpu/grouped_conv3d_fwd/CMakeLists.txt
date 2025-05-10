# ONLY XDL_AND_WMMA_KERNELS
set(GROUPED_CONV3D_FWD
   xdl/device_grouped_conv3d_fwd_xdl_gndhwc_gkzyxc_gndhwk_bf16_instance.cpp
   xdl/device_grouped_conv3d_fwd_xdl_gndhwc_gkzyxc_gndhwk_f16_instance.cpp
   xdl/device_grouped_conv3d_fwd_xdl_gndhwc_gkzyxc_gndhwk_f32_instance.cpp
   xdl/device_grouped_conv3d_fwd_xdl_gndhwc_gkzyxc_gndhwk_int8_instance.cpp
   xdl/device_grouped_conv3d_fwd_xdl_ndhwgc_gkzyxc_ndhwgk_bf16_instance.cpp
   xdl/device_grouped_conv3d_fwd_xdl_ndhwgc_gkzyxc_ndhwgk_f16_instance.cpp
   xdl/device_grouped_conv3d_fwd_xdl_ndhwgc_gkzyxc_ndhwgk_f32_instance.cpp
   xdl/device_grouped_conv3d_fwd_xdl_ndhwgc_gkzyxc_ndhwgk_int8_instance.cpp
   wmma/device_grouped_conv3d_fwd_wmma_gndhwc_gkzyxc_gndhwk_f16_instance.cpp
   wmma/device_grouped_conv3d_fwd_wmma_gndhwc_gkzyxc_gndhwk_i8_instance.cpp
   wmma/device_grouped_conv3d_fwd_wmma_ndhwgc_gkzyxc_ndhwgk_f16_instance.cpp
   wmma/device_grouped_conv3d_fwd_wmma_ndhwgc_gkzyxc_ndhwgk_i8_instance.cpp
   wmma/device_grouped_conv3d_fwd_wmma_gndhwc_gkzyxc_gndhwk_f16_1x1p0_instance.cpp
   wmma/device_grouped_conv3d_fwd_wmma_gndhwc_gkzyxc_gndhwk_i8_1x1p0_instance.cpp
   wmma/device_grouped_conv3d_fwd_wmma_ndhwgc_gkzyxc_ndhwgk_f16_1x1p0_instance.cpp
   wmma/device_grouped_conv3d_fwd_wmma_ndhwgc_gkzyxc_ndhwgk_i8_1x1p0_instance.cpp
   wmma/device_grouped_conv3d_fwd_wmma_gndhwc_gkzyxc_gndhwk_f16_1x1s1p0_instance.cpp
   wmma/device_grouped_conv3d_fwd_wmma_gndhwc_gkzyxc_gndhwk_i8_1x1s1p0_instance.cpp
   wmma/device_grouped_conv3d_fwd_wmma_ndhwgc_gkzyxc_ndhwgk_f16_1x1s1p0_instance.cpp
   wmma/device_grouped_conv3d_fwd_wmma_ndhwgc_gkzyxc_ndhwgk_i8_1x1s1p0_instance.cpp
   wmma/device_grouped_conv3d_fwd_wmma_gndhwc_gkzyxc_gndhwk_f16_oddc_instance.cpp
   wmma/device_grouped_conv3d_fwd_wmma_gndhwc_gkzyxc_gndhwk_i8_oddc_instance.cpp
   wmma/device_grouped_conv3d_fwd_wmma_ndhwgc_gkzyxc_ndhwgk_f16_oddc_instance.cpp
   wmma/device_grouped_conv3d_fwd_wmma_ndhwgc_gkzyxc_ndhwgk_i8_oddc_instance.cpp
)

if((DTYPES MATCHES "fp8" AND DTYPES MATCHES "fp16") OR NOT DEFINED DTYPES)
    list(APPEND GROUPED_CONV3D_FWD
      xdl/device_grouped_conv3d_fwd_xdl_ndhwgc_gkzyxc_ndhwgk_f16_comp_fp8_instance.cpp)
endif()

if(DTYPES MATCHES "fp8" OR NOT DEFINED DTYPES)
    list(APPEND GROUPED_CONV3D_FWD
      xdl/device_grouped_conv3d_fwd_xdl_ndhwgc_gkzyxc_ndhwgk_fp8_instance.cpp)
endif()

if(DTYPES MATCHES "bf8" OR NOT DEFINED DTYPES)
    list(APPEND GROUPED_CONV3D_FWD
      xdl/device_grouped_conv3d_fwd_xdl_ndhwgc_gkzyxc_ndhwgk_bf8_instance.cpp)
endif()

if((DTYPES MATCHES "fp8" AND DTYPES MATCHES "bf8") OR NOT DEFINED DTYPES)
    list(APPEND GROUPED_CONV3D_FWD
      xdl/device_grouped_conv3d_fwd_xdl_ndhwgc_gkzyxc_ndhwgk_fp8_bf8_instance.cpp)
    list(APPEND GROUPED_CONV3D_FWD
      xdl/device_grouped_conv3d_fwd_xdl_ndhwgc_gkzyxc_ndhwgk_bf8_fp8_instance.cpp)
endif()

add_instance_library(device_grouped_conv3d_fwd_instance ${GROUPED_CONV3D_FWD})
