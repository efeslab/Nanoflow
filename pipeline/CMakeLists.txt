cmake_minimum_required(VERSION 3.26.1)

#set(CMAKE_C_COMPILER "/usr/bin/gcc-11")
#set(CMAKE_CXX_COMPILER "/usr/bin/g++-11")
set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 80)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

get_filename_component(PARENT_DIR ${CMAKE_SOURCE_DIR} DIRECTORY)

set(NVBENCH_DIR ${PARENT_DIR}/3rdparty/nvbench CACHE PATH "Path to nvbench src")
set(GTEST_DIR ${PARENT_DIR}/3rdparty/googletest CACHE PATH "Path to googletest src")
set(CUTLASS_DIR ${PARENT_DIR}/3rdparty/cutlass CACHE PATH "Path to modified cutlass src")
set(SPDLOG_DIR  ${PARENT_DIR}/3rdparty/spdlog CACHE PATH "Path to spdlog src")
set(MSCCLPP_DIR /usr/local/mscclpp CACHE PATH "Path to mscclpp install")
set(FLASHINFER_DIR ${PARENT_DIR}/3rdparty/flashinfer CACHE PATH "Path to flashinfer src")
# override by SMALL_FLASHINFER
#set(FLASHINFER_DIR ${PARENT_DIR}/3rdparty/flashinfer CACHE PATH "Path to flashinfer src")

project(small_gemv LANGUAGES CUDA CXX)

# ------------- Add FP8 Macro for compilation -----------------#
if(CMAKE_CUDA_ARCHITECTURES GREATER_EQUAL 89)
  add_compile_definitions(FLASHINFER_ENABLE_FP8)
endif()


# ------------- Configure libraries -----------------#

find_package(MPI QUIET)
if (NOT MPI_FOUND)
    message(FATAL_ERROR "MPI not found, try apt install libopenmpi-dev")
endif()

add_subdirectory(${SPDLOG_DIR} spdlog)

find_package(pybind11 REQUIRED)
if (NOT pybind11_FOUND)
    message(FATAL_ERROR "pybind11 not found, try apt install python3-pybind11")
endif()

find_program(STUBGEN_EXECUTABLE NAMES stubgen)

add_compile_options(-w)
include_directories(${MSCCLPP_DIR}/include)

include_directories(${CUTLASS_DIR}/include)
include_directories(${CUTLASS_DIR}/tools/util/include)
# override by SMALL_FLASHINFER
#include_directories(${FLASHINFER_DIR}/include)
# ------------- Build Network Test -----------------#

find_library(MSCCLPP_LIBRARY NAMES mscclpp PATHS ${MSCCLPP_DIR}/lib NO_DEFAULT_PATH)

add_executable(test_comm "${CMAKE_SOURCE_DIR}/src/comm_test.cu" "${CMAKE_SOURCE_DIR}/src/comm.cu")
target_include_directories(test_comm PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(test_comm PRIVATE ${PARENT_DIR}/3rdparty/cutlass/include)
target_link_libraries(test_comm PRIVATE MPI::MPI_CXX)
target_link_libraries(test_comm PRIVATE ${MSCCLPP_LIBRARY})
target_link_libraries(test_comm PRIVATE spdlog::spdlog)
target_compile_definitions(test_comm PRIVATE -DENABLE_MPI)

# ------------- Configure FlashInfer Library -----------------#
# Note that codes are stored at $PARENT_DIR/new-small-gemv
set(SMALL_FLASHINFER_DIR ${PARENT_DIR}/gemv)

# Config Template Expansion
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

# Generated template kernels
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
                COMMAND python3 ${SMALL_FLASHINFER_DIR}/python/generate_batch_paged_decode_inst.py ${generated_kernel_src}
                DEPENDS ${SMALL_FLASHINFER_DIR}/python/generate_batch_paged_decode_inst.py
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
target_include_directories(decode_kernels PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(decode_kernels PRIVATE ${SMALL_FLASHINFER_DIR}/include ${FLASHINFER_DIR}/include)
target_link_libraries(decode_kernels PRIVATE spdlog::spdlog)

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
                  COMMAND python3 ${SMALL_FLASHINFER_DIR}/python/generate_batch_paged_prefill_inst.py ${generated_kernel_src}
                  DEPENDS ${SMALL_FLASHINFER_DIR}/python/generate_batch_paged_prefill_inst.py
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
target_include_directories(prefill_kernels PRIVATE ${SMALL_FLASHINFER_DIR}/include ${FLASHINFER_DIR}/include)
target_include_directories(prefill_kernels PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(prefill_kernels PRIVATE spdlog::spdlog)
message(STATUS "add include ${CMAKE_SOURCE_DIR}/include")

# ------------- Manage Source Files -----------------#



set(KEY_SRC "${CMAKE_SOURCE_DIR}/src/comm.cu" 
            "${CMAKE_SOURCE_DIR}/src/computeBound.cu" 
            "${CMAKE_SOURCE_DIR}/src/networkManager.cu" 
            "${CMAKE_SOURCE_DIR}/src/pipeline.cu" 
            "${CMAKE_SOURCE_DIR}/src/pipeline_nonoverlap.cu"
            "${CMAKE_SOURCE_DIR}/src/pipeline_nonoverlap_nanobatch.cu"
            "${CMAKE_SOURCE_DIR}/src/sleep.cu" 
            "${CMAKE_SOURCE_DIR}/src/vortexData.cu"
            "${CMAKE_SOURCE_DIR}/src/offloadKernel.cu"
            "${CMAKE_SOURCE_DIR}/src/gemvDependency.cu"
            "${CMAKE_SOURCE_DIR}/src/small_cuda_operator.cu"
            "${CMAKE_SOURCE_DIR}/src/tensorLogger.cu"
            )

set(C_SRC "${CMAKE_SOURCE_DIR}/src/computeMain.cu")
set(PYBIND_SRC "${CMAKE_SOURCE_DIR}/src/pybind.cu")

set(C_ALL ${KEY_SRC} ${C_SRC})
set(PYBIND_ALL ${KEY_SRC} ${PYBIND_SRC})



# --------------- Build GEMM Libraries -----------------#

set (GEMM_BASE_SRC "${CMAKE_SOURCE_DIR}/src/generate-gemm")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${GEMM_BASE_SRC}/genGEMM.py ${GEMM_BASE_SRC}/gemmFactory.in)
execute_process(COMMAND make WORKING_DIRECTORY ${GEMM_BASE_SRC})
file(GLOB GEMM_SRC "${GEMM_BASE_SRC}/*.cu")
add_library(gemm_lib STATIC ${GEMM_SRC})
target_include_directories(gemm_lib PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(gemm_lib PRIVATE ${PARENT_DIR}/3rdparty/cutlass/include)
target_include_directories(gemm_lib PRIVATE ${PARENT_DIR}/3rdparty/cutlass/tools/util/include)
target_link_libraries(gemm_lib PRIVATE spdlog::spdlog)


# --------------- Build Common Libraries -----------------#

add_library(shared_lib STATIC ${KEY_SRC})
target_include_directories(shared_lib PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(shared_lib PRIVATE ${PARENT_DIR}/3rdparty/cutlass/include)
target_include_directories(shared_lib PRIVATE ${PARENT_DIR}/3rdparty/cutlass/tools/util/include)
target_include_directories(shared_lib PRIVATE ${SMALL_FLASHINFER_DIR}/include ${FLASHINFER_DIR}/include)
target_link_libraries(shared_lib PRIVATE decode_kernels prefill_kernels)
target_link_libraries(shared_lib PRIVATE ${MSCCLPP_LIBRARY})
target_link_libraries(shared_lib PRIVATE MPI::MPI_CXX)
target_link_libraries(shared_lib PRIVATE spdlog::spdlog)
target_link_libraries(shared_lib PRIVATE gemm_lib)


# ------------- Build CUDA side -----------------#

add_executable(test_compute ${C_ALL})
# Include Cutlass Libraries
target_include_directories(test_compute PRIVATE ${CMAKE_SOURCE_DIR}/include)
#target_include_directories(test_compute PRIVATE ${PARENT_DIR}/3rdparty/cutlass/include)
#target_include_directories(test_compute PRIVATE ${PARENT_DIR}/3rdparty/cutlass/tools/util/include)
# Include FlashInfer Libraries
target_include_directories(test_compute PRIVATE ${SMALL_FLASHINFER_DIR}/include ${FLASHINFER_DIR}/include)
target_link_libraries(test_compute PRIVATE decode_kernels prefill_kernels)
# Include Network Libraries
target_link_libraries(test_compute PRIVATE ${MSCCLPP_LIBRARY})
target_link_libraries(test_compute PRIVATE MPI::MPI_CXX)
# Include Log Libraries
target_link_libraries(test_compute PRIVATE spdlog::spdlog)
# # include shared library
# target_link_libraries(test_compute PRIVATE shared_lib)
target_link_libraries(test_compute PRIVATE gemm_lib)

# ------------- Build Pybind side -----------------#

pybind11_add_module(pllm_python ${PYBIND_ALL})
# Include Cutlass Libraries
target_include_directories(pllm_python PRIVATE ${CMAKE_SOURCE_DIR}/include)
#target_include_directories(pllm_python PRIVATE ${PARENT_DIR}/3rdparty/cutlass/include)
#target_include_directories(pllm_python PRIVATE ${PARENT_DIR}/3rdparty/cutlass/tools/util/include)
# Include FlashInfer Libraries
target_include_directories(pllm_python PRIVATE ${SMALL_FLASHINFER_DIR}/include ${FLASHINFER_DIR}/include)
target_link_libraries(pllm_python PRIVATE decode_kernels prefill_kernels)
# Include Network Libraries
target_link_libraries(pllm_python PRIVATE ${MSCCLPP_LIBRARY})
target_link_libraries(pllm_python PRIVATE MPI::MPI_CXX)
# Include Log Libraries
target_link_libraries(pllm_python PRIVATE spdlog::spdlog)
# # include shared library
# target_link_libraries(pllm_python PRIVATE shared_lib)
target_link_libraries(pllm_python PRIVATE gemm_lib)


# ------------- Generate Stub -----------------#
if(NOT STUBGEN_EXECUTABLE)
    message(WARNING "stubgen not found. Please ensure MyPy is installed to generate stub files.")
else()
    # Add a custom command to run stubgen after the target is built
    add_custom_command(
        TARGET pllm_python POST_BUILD
        COMMAND ${STUBGEN_EXECUTABLE} -m pllm_python -o .
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating Python stubs for pllm_python"
    )
endif()
