# Currently ck_tile is only built on gfx9
if(GPU_TARGETS MATCHES "gfx9")
    add_gtest_executable(test_ck_tile_gemm_pipeline test_gemm_pipeline.cpp)
endif()
