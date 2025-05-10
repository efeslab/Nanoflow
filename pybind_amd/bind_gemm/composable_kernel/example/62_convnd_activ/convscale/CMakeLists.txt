list(APPEND gpu_list gfx908 gfx90a gfx940 gfx941 gfx942)
set(target 0)
foreach(gpu IN LISTS GPU_TARGETS)
 if(gpu IN_LIST gpu_list AND target EQUAL 0)
     add_custom_target(example_convnd_activ_xdl_convscale)
     add_example_executable(example_convnd_fwd_xdl_convscale_fp8 convnd_fwd_xdl_convscale_fp8.cpp)
     add_example_dependencies(example_convnd_activ_xdl_convscale example_convnd_fwd_xdl_convscale_fp8 )

     add_example_executable(example_convnd_fwd_xdl_convscale_bf8 convnd_fwd_xdl_convscale_bf8.cpp)
     add_example_dependencies(example_convnd_activ_xdl_convscale example_convnd_fwd_xdl_convscale_bf8)

     add_example_executable(example_convnd_fwd_xdl_convscale_fp8_bf8 convnd_fwd_xdl_convscale_fp8_bf8.cpp)
     add_example_dependencies(example_convnd_activ_xdl_convscale example_convnd_fwd_xdl_convscale_fp8_bf8)

     add_example_executable(example_convnd_fwd_xdl_convscale_bf8_fp8 convnd_fwd_xdl_convscale_bf8_fp8.cpp)
     add_example_dependencies(example_convnd_activ_xdl_convscale example_convnd_fwd_xdl_convscale_bf8_fp8)

     set(target 1)
 endif()
endforeach()
