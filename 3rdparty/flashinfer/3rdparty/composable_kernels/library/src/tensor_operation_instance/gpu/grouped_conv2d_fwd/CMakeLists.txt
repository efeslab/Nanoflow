# XDL_DL_WMMA_KERNELS
add_instance_library(device_grouped_conv2d_fwd_instance
   #xdl
   # GNHWC, GKYXC, GNHWK
   xdl/device_grouped_conv2d_fwd_xdl_gnhwc_gkyxc_gnhwk_bf16_instance.cpp
   xdl/device_grouped_conv2d_fwd_xdl_gnhwc_gkyxc_gnhwk_f16_instance.cpp
   xdl/device_grouped_conv2d_fwd_xdl_gnhwc_gkyxc_gnhwk_f32_instance.cpp
   # NHWGC, GKYXC, NHWGK
   xdl/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_bf16_instance.cpp
   xdl/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_f16_instance.cpp
   xdl/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_f32_instance.cpp
   #dl
   # GNHWC, GKYXC, GNHWK
   dl/device_grouped_conv2d_fwd_dl_gnhwc_gkyxc_gnhwk_f16_instance.cpp
   dl/device_grouped_conv2d_fwd_dl_gnhwc_gkyxc_gnhwk_f32_instance.cpp
   # NHWGC, GKYXC, NHWGK
   dl/device_grouped_conv2d_fwd_dl_nhwgc_gkyxc_nhwgk_f16_instance.cpp
   dl/device_grouped_conv2d_fwd_dl_nhwgc_gkyxc_nhwgk_f32_instance.cpp
   # WMMA
   # GNHWC, GKYXC, GNHWK
  wmma/device_grouped_conv2d_fwd_wmma_gnhwc_gkyxc_gnhwk_f16_instance.cpp
  wmma/device_grouped_conv2d_fwd_wmma_gnhwc_gkyxc_gnhwk_i8_instance.cpp
  wmma/device_grouped_conv2d_fwd_wmma_gnhwc_gkyxc_gnhwk_f16_1x1p0_instance.cpp
  wmma/device_grouped_conv2d_fwd_wmma_gnhwc_gkyxc_gnhwk_i8_1x1p0_instance.cpp
  wmma/device_grouped_conv2d_fwd_wmma_gnhwc_gkyxc_gnhwk_f16_1x1s1p0_instance.cpp
  wmma/device_grouped_conv2d_fwd_wmma_gnhwc_gkyxc_gnhwk_i8_1x1s1p0_instance.cpp
  wmma/device_grouped_conv2d_fwd_wmma_gnhwc_gkyxc_gnhwk_f16_oddc_instance.cpp
  wmma/device_grouped_conv2d_fwd_wmma_gnhwc_gkyxc_gnhwk_i8_oddc_instance.cpp
  ## NHWGC, GKYXC, NHWGK
  wmma/device_grouped_conv2d_fwd_wmma_nhwgc_gkyxc_nhwgk_f16_instance.cpp
  wmma/device_grouped_conv2d_fwd_wmma_nhwgc_gkyxc_nhwgk_i8_instance.cpp
  wmma/device_grouped_conv2d_fwd_wmma_nhwgc_gkyxc_nhwgk_f16_1x1p0_instance.cpp
  wmma/device_grouped_conv2d_fwd_wmma_nhwgc_gkyxc_nhwgk_i8_1x1p0_instance.cpp
  wmma/device_grouped_conv2d_fwd_wmma_nhwgc_gkyxc_nhwgk_f16_1x1s1p0_instance.cpp
  wmma/device_grouped_conv2d_fwd_wmma_nhwgc_gkyxc_nhwgk_i8_1x1s1p0_instance.cpp
  wmma/device_grouped_conv2d_fwd_wmma_nhwgc_gkyxc_nhwgk_f16_oddc_instance.cpp
  wmma/device_grouped_conv2d_fwd_wmma_nhwgc_gkyxc_nhwgk_i8_oddc_instance.cpp
)
