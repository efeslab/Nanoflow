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
   xdl/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_int8_instance.cpp
   # NGCHW, GKYXC, NGKHW
   xdl/device_grouped_conv2d_fwd_xdl_ngchw_gkyxc_ngkhw_bf16_instance.cpp
   xdl/device_grouped_conv2d_fwd_xdl_ngchw_gkyxc_ngkhw_f16_instance.cpp
   xdl/device_grouped_conv2d_fwd_xdl_ngchw_gkyxc_ngkhw_f32_instance.cpp
   xdl/device_grouped_conv2d_fwd_xdl_ngchw_gkyxc_ngkhw_int8_instance.cpp
   # large tensor
   # NHWGC, GKYXC, NHWGK
   xdl/large_tensor/device_grouped_conv2d_fwd_xdl_large_tensor_nhwgc_gkyxc_nhwgk_bf16_instance.cpp
   xdl/large_tensor/device_grouped_conv2d_fwd_xdl_large_tensor_nhwgc_gkyxc_nhwgk_f16_instance.cpp
   xdl/large_tensor/device_grouped_conv2d_fwd_xdl_large_tensor_nhwgc_gkyxc_nhwgk_f32_instance.cpp
   xdl/large_tensor/device_grouped_conv2d_fwd_xdl_large_tensor_nhwgc_gkyxc_nhwgk_int8_instance.cpp
   # merged groups
   # NHWGC, GKYXC, NHWGK
   xdl/merged_groups/device_grouped_conv2d_fwd_xdl_merged_groups_nhwgc_gkyxc_nhwgk_bf16_instance.cpp
   xdl/merged_groups/device_grouped_conv2d_fwd_xdl_merged_groups_nhwgc_gkyxc_nhwgk_f16_instance.cpp
   xdl/merged_groups/device_grouped_conv2d_fwd_xdl_merged_groups_nhwgc_gkyxc_nhwgk_f32_instance.cpp
   xdl/merged_groups/device_grouped_conv2d_fwd_xdl_merged_groups_nhwgc_gkyxc_nhwgk_int8_instance.cpp
   # NGCHW, GKYXC, NGKHW
   xdl/merged_groups/device_grouped_conv2d_fwd_xdl_merged_groups_ngchw_gkyxc_ngkhw_bf16_instance.cpp
   xdl/merged_groups/device_grouped_conv2d_fwd_xdl_merged_groups_ngchw_gkyxc_ngkhw_f16_instance.cpp
   xdl/merged_groups/device_grouped_conv2d_fwd_xdl_merged_groups_ngchw_gkyxc_ngkhw_f32_instance.cpp
   xdl/merged_groups/device_grouped_conv2d_fwd_xdl_merged_groups_ngchw_gkyxc_ngkhw_int8_instance.cpp
   #mem
   # NHWGC, GKYXC, NHWGK
   xdl/mem/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_bf16_mem_intra_instance.cpp
   xdl/mem/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_f16_mem_intra_instance.cpp
   xdl/mem/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_f32_mem_intra_instance.cpp
   xdl/mem/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_int8_mem_intra_instance.cpp
   # NHWGC, GKYXC, NHWGK
   xdl/mem/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_bf16_mem_inter_instance.cpp
   xdl/mem/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_f16_mem_inter_instance.cpp
   xdl/mem/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_f32_mem_inter_instance.cpp
   xdl/mem/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_int8_mem_inter_instance.cpp
   # NGCHW, GKYXC, NGKHW
   xdl/mem/device_grouped_conv2d_fwd_xdl_ngchw_gkyxc_ngkhw_bf16_mem_intra_instance.cpp
   xdl/mem/device_grouped_conv2d_fwd_xdl_ngchw_gkyxc_ngkhw_f16_mem_intra_instance.cpp
   xdl/mem/device_grouped_conv2d_fwd_xdl_ngchw_gkyxc_ngkhw_f32_mem_intra_instance.cpp
   xdl/mem/device_grouped_conv2d_fwd_xdl_ngchw_gkyxc_ngkhw_int8_mem_intra_instance.cpp
   # NGCHW, GKYXC, NGKHW
   xdl/mem/device_grouped_conv2d_fwd_xdl_ngchw_gkyxc_ngkhw_bf16_mem_inter_instance.cpp
   xdl/mem/device_grouped_conv2d_fwd_xdl_ngchw_gkyxc_ngkhw_f16_mem_inter_instance.cpp
   xdl/mem/device_grouped_conv2d_fwd_xdl_ngchw_gkyxc_ngkhw_f32_mem_inter_instance.cpp
   xdl/mem/device_grouped_conv2d_fwd_xdl_ngchw_gkyxc_ngkhw_int8_mem_inter_instance.cpp
   #comp
   # NHWGC, GKYXC, NHWGK
   xdl/comp/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_bf16_comp_instance.cpp
   xdl/comp/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_f16_comp_instance.cpp
   xdl/comp/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_f32_comp_instance.cpp
   xdl/comp/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_int8_comp_instance.cpp
   # NGCHW, GKYXC, NGKHW
   xdl/comp/device_grouped_conv2d_fwd_xdl_ngchw_gkyxc_ngkhw_bf16_comp_instance.cpp
   xdl/comp/device_grouped_conv2d_fwd_xdl_ngchw_gkyxc_ngkhw_f16_comp_instance.cpp
   xdl/comp/device_grouped_conv2d_fwd_xdl_ngchw_gkyxc_ngkhw_f32_comp_instance.cpp
   xdl/comp/device_grouped_conv2d_fwd_xdl_ngchw_gkyxc_ngkhw_int8_comp_instance.cpp
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
