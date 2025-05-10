# generate a list of kernels, but not actually emit files at config stage
execute_process(
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/generate.py
  --list_blobs ${CMAKE_CURRENT_BINARY_DIR}/blob_list.txt
)

# NOTE: for cmake, the FMHA_FWD_GEN_BLOBS files must be in the same directory
#       as current cmake list, otherwise will not figure out the dependency properly
file(STRINGS ${CMAKE_CURRENT_BINARY_DIR}/blob_list.txt FMHA_FWD_GEN_BLOBS)

add_custom_command(
  OUTPUT ${FMHA_FWD_GEN_BLOBS}
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/generate.py
  --output_dir ${CMAKE_CURRENT_BINARY_DIR}
)

set(EXAMPLE_FMHA_FWD "tile_example_fmha_fwd")
# not using add_example_executable() to add this target, since we don't want this to have
# to be included in "make all/install/check"
message("adding example ${EXAMPLE_FMHA_FWD}")
add_executable(${EXAMPLE_FMHA_FWD} EXCLUDE_FROM_ALL fmha_fwd.cpp)
target_include_directories(${EXAMPLE_FMHA_FWD} PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_sources(${EXAMPLE_FMHA_FWD} PRIVATE ${FMHA_FWD_GEN_BLOBS})

# NOTE: this is dangerous since will change the whole kernel to flush denormals
#       WIP with compiler team for an exp2 intrinsic..., then remove this
if(NOT DEFINED FMHA_FWD_FAST_EXP2)
    set(FMHA_FWD_FAST_EXP2 true)
endif()

set(EXAMPLE_FMHA_FWD_COMPILE_OPTIONS)

# NOTE: we turn off undefined-func-template to let source compile without explicit declare function specializations
#       ... because they are auto-generated
if(FMHA_FWD_FAST_EXP2)
	list(APPEND EXAMPLE_FMHA_FWD_COMPILE_OPTIONS -Wno-undefined-func-template -DCK_TILE_FMHA_FWD_FAST_EXP2=1 -fgpu-flush-denormals-to-zero)
else()
	list(APPEND EXAMPLE_FMHA_FWD_COMPILE_OPTIONS -Wno-undefined-func-template -DCK_TILE_FMHA_FWD_FAST_EXP2=0)
endif()

# Allow comparing floating points directly in order to check sentinel values
list(APPEND EXAMPLE_FMHA_FWD_COMPILE_OPTIONS -Wno-float-equal)

target_compile_options(${EXAMPLE_FMHA_FWD} PRIVATE ${EXAMPLE_FMHA_FWD_COMPILE_OPTIONS})
