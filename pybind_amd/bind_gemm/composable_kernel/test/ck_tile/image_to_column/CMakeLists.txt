# Currently ck_tile is only built on gfx9
if(GPU_TARGETS MATCHES "gfx9")
    add_gtest_executable(test_tile_image_to_column test_tile_image_to_column.cpp)
endif()
