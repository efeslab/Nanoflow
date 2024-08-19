cmake_minimum_required(VERSION 3.26.1)

set(CMAKE_C_COMPILER "/usr/bin/gcc-11")
set(CMAKE_CXX_COMPILER "/usr/bin/g++-11")
set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 80)

project(small_gemv LANGUAGES CUDA CXX)

get_filename_component(PARENT_DIR "${CMAKE_CURRENT_SOURCE_DIR}" DIRECTORY)

# ------------- Add FP8 Macro for compilation -----------------#
if(CMAKE_CUDA_ARCHITECTURES GREATER_EQUAL 89)
  add_compile_definitions(FLASHINFER_ENABLE_FP8)
endif()



# ------------- add nvbench and gtest -----------------#
add_subdirectory(${PARENT_DIR}/3rdparty/nvbench ${CMAKE_BINARY_DIR}/nvbench)
add_subdirectory(${PARENT_DIR}/3rdparty/gtest ${CMAKE_BINARY_DIR}/gtest_build)

# ------------- config template -----------------#
set (GROUP_SIZES 1 4 6 8)
set (HEAD_DIMS 64 128 256)
set (KV_LAYOUTS 0 1)
set (POS_ENCODING_MODES 0 1 2)
set (ALLOW_FP16_QK_REDUCTIONS "false" "true")
set (CAUSALS "false" "true")
set (DECODE_DTYPES "f16")
set (PREFILL_DTYPES "f16")
set (IDTYPES "i32")
set (LAUNCH_TYPES "all" "small")

file(MAKE_DIRECTORY ${PROJECT_SOURCE_DIR}/src/generated)

# ------------- generate batch decode inst -----------------#
foreach(group_size IN LISTS GROUP_SIZES)
  foreach(head_dim IN LISTS HEAD_DIMS)
    foreach(kv_layout IN LISTS KV_LAYOUTS)
      foreach(pos_encoding_mode IN LISTS POS_ENCODING_MODES)
        # paged kv-cache
        foreach(idtype IN LISTS IDTYPES)
          foreach(dtype IN LISTS DECODE_DTYPES)
            foreach(ltype IN LISTS LAUNCH_TYPES)
              set(generated_kernel_src ${PROJECT_SOURCE_DIR}/src/generated/batch_paged_decode_group_${group_size}_head_${head_dim}_layout_${kv_layout}_posenc_${pos_encoding_mode}_dtypein_${dtype}_dtypeout_${dtype}_idtype_${idtype}_launchtype_${ltype}.cu)
              add_custom_command(
                OUTPUT ${generated_kernel_src}
                COMMAND python ${PROJECT_SOURCE_DIR}/python/generate_batch_paged_decode_inst.py ${generated_kernel_src}
                DEPENDS ${PROJECT_SOURCE_DIR}/python/generate_batch_paged_decode_inst.py
                COMMENT "Generating additional source file ${generated_kernel_src}"
                VERBATIM
              )
              list(APPEND batch_decode_kernels_src ${generated_kernel_src})
            endforeach(ltype)
          endforeach(dtype)
        endforeach(idtype)
      endforeach(pos_encoding_mode)
    endforeach(kv_layout)
  endforeach(head_dim)
endforeach(group_size)

add_library(decode_kernels STATIC ${batch_decode_kernels_src})
target_include_directories(decode_kernels PRIVATE ${CMAKE_SOURCE_DIR}/include ${PARENT_DIR}/3rdparty/flashinfer/include)

# ------------- generate batch prefill inst -----------------#
foreach(group_size IN LISTS GROUP_SIZES)
  foreach(head_dim IN LISTS HEAD_DIMS)
    foreach(kv_layout IN LISTS KV_LAYOUTS)
        foreach(allow_fp16_qk_reduction IN LISTS ALLOW_FP16_QK_REDUCTIONS)
          foreach(causal IN LISTS CAUSALS)
            foreach(dtype IN LISTS PREFILL_DTYPES)
              foreach(idtype IN LISTS IDTYPES)
                foreach(ltype IN LISTS LAUNCH_TYPES)
                  set(generated_kernel_src ${PROJECT_SOURCE_DIR}/src/generated/batch_paged_prefill_group_${group_size}_head_${head_dim}_layout_${kv_layout}_fp16qkred_${allow_fp16_qk_reduction}_causal_${causal}_dtypein_${dtype}_dtypeout_${dtype}_idtype_${idtype}_launchtype_${ltype}.cu)
                  add_custom_command(
                    OUTPUT ${generated_kernel_src}
                    COMMAND python ${PROJECT_SOURCE_DIR}/python/generate_batch_paged_prefill_inst.py ${generated_kernel_src}
                    DEPENDS ${PROJECT_SOURCE_DIR}/python/generate_batch_paged_prefill_inst.py
                    COMMENT "Generating additional source file ${generated_kernel_src}"
                    VERBATIM
                  )
                  list(APPEND batch_paged_prefill_kernels_src ${generated_kernel_src})   
                endforeach(ltype) 
              endforeach(idtype)
            endforeach(dtype)
          endforeach(causal)
        endforeach(allow_fp16_qk_reduction)
    endforeach(kv_layout)
  endforeach(head_dim)
endforeach(group_size)

add_library(prefill_kernels STATIC ${batch_paged_prefill_kernels_src})
target_include_directories(prefill_kernels PRIVATE ${CMAKE_SOURCE_DIR}/include ${PARENT_DIR}/3rdparty/flashinfer/include)
message(STATUS ${PARENT_DIR}/3rdparty/flashinfer/include)

# ------------- compile bench_batch_decode -----------------#
message(STATUS "Compile batch_decode kernel benchmarks.")
add_executable(bench_batch_decode ${CMAKE_SOURCE_DIR}/src/bench_batch_decode.cu)
target_include_directories(bench_batch_decode PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(bench_batch_decode PRIVATE ${PARENT_DIR}/3rdparty/nvbench)
target_include_directories(bench_batch_decode PRIVATE ${PARENT_DIR}/3rdparty/flashinfer/include)
target_link_libraries(bench_batch_decode PRIVATE nvbench::main decode_kernels prefill_kernels)

# ------------- compile bench_batch_prefill -----------------#
message(STATUS "Compile batch_prefill kernel benchmarks.")
add_executable(bench_batch_prefill ${CMAKE_SOURCE_DIR}/src/bench_batch_prefill.cu)
target_include_directories(bench_batch_prefill PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(bench_batch_prefill PRIVATE ${PARENT_DIR}/3rdparty/nvbench)
target_include_directories(bench_batch_prefill PRIVATE ${PARENT_DIR}/3rdparty/flashinfer/include)
target_link_libraries(bench_batch_prefill PRIVATE nvbench::main prefill_kernels)

# ------------- compile test_batch_decode -----------------#
add_executable(test_batch_decode ${CMAKE_SOURCE_DIR}/src/test_batch_decode.cu)
message(STATUS "Compile batch_decode kernel testcases.")
target_include_directories(test_batch_decode PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(test_batch_decode PRIVATE ${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
target_include_directories(test_batch_decode PRIVATE ${PARENT_DIR}/3rdparty/flashinfer/include)
target_link_libraries(test_batch_decode PRIVATE gtest gtest_main decode_kernels)

# # ------------- compile test_batch_prefill -----------------#
add_executable(test_batch_prefill ${CMAKE_SOURCE_DIR}/src/test_batch_prefill.cu)
message(STATUS "Compile batch_prefill kernel testcases.")
target_include_directories(test_batch_prefill PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(test_batch_prefill PRIVATE ${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
target_include_directories(test_batch_prefill PRIVATE ${PARENT_DIR}/3rdparty/flashinfer/include)
target_link_libraries(test_batch_prefill PRIVATE gtest gtest_main prefill_kernels)

# set(COMPUTE_QK_OPTIONS 0 1)
# set(UPDATE_MDO_OPTIONS 0 1)
# set(COMPUTE_SFM_OPTIONS 0 1)
  
# foreach(compute_qk IN LISTS COMPUTE_QK_OPTIONS)
#   foreach(update_mdo IN LISTS UPDATE_MDO_OPTIONS)
#     foreach(compute_sfm IN LISTS COMPUTE_SFM_OPTIONS)
#       set(LIB_NAME prefill_kernels_${compute_qk}_${update_mdo}_${compute_sfm})
#       add_library(${LIB_NAME} STATIC ${batch_paged_prefill_kernels_src})
#       if(NOT compute_qk EQUAL 1)
#         target_compile_definitions(${LIB_NAME} PRIVATE -DNOT_COMPUTE_QK)
#       endif()
#       if(NOT update_mdo EQUAL 1)
#         target_compile_definitions(${LIB_NAME} PRIVATE -DNOT_UPDATE_MDO)
#       endif()
#       if(NOT compute_sfm EQUAL 1)
#         target_compile_definitions(${LIB_NAME} PRIVATE -DNOT_COMPUTE_SFM)
#       endif()
#       target_include_directories(${LIB_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/3rdparty/flashinfer/include)

#       set(EXE_NAME bench_batch_decode_${compute_qk}_${update_mdo}_${compute_sfm})
#       add_executable(${EXE_NAME} ${CMAKE_SOURCE_DIR}/src/bench_batch_decode.cu)
#       target_include_directories(${EXE_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/include)
#       target_include_directories(${EXE_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/3rdparty/nvbench)
#       target_include_directories(${EXE_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/3rdparty/flashinfer/include)
#       target_link_libraries(${EXE_NAME} PRIVATE nvbench::main decode_kernels ${LIB_NAME})

#     endforeach(compute_sfm)
#   endforeach(update_mdo)
# endforeach(compute_qk)