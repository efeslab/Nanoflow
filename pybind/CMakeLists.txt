cmake_minimum_required(VERSION 3.26.1)
project(Nanoflow-python LANGUAGES CXX CUDA)

# Language standards and compiler settings
set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 90)

if(CMAKE_CUDA_ARCHITECTURES STREQUAL "90")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -DNDEBUG --generate-code=arch=compute_90a,code=[compute_90a,sm_90a] -Xcompiler=-fPIE -Xcompiler=-Wconversion -Xcompiler=-fno-strict-aliasing")
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

#set(CMAKE_C_COMPILER "/usr/bin/gcc-11")
#set(CMAKE_CXX_COMPILER "/usr/bin/g++-11")

# Check for CUDA Toolkit location
if(EXISTS "/opt/nvidia/hpc_sdk/Linux_x86_64/23.11/cuda/12.3")
  set(CUDA_TOOLKIT_ROOT_DIR "/opt/nvidia/hpc_sdk/Linux_x86_64/23.11/cuda/12.3")
endif()

# Find required packages
find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)
find_program(PYBIND11_STUBGEN_EXECUTABLE NAMES pybind11-stubgen)

# Get Torch CMake path
execute_process(
  COMMAND python -c "import torch; print(torch.utils.cmake_prefix_path)"
  OUTPUT_VARIABLE TorchCMakePrefixPath
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(Torch_DIR "${TorchCMakePrefixPath}/Torch")

# Set paths
list(APPEND CMAKE_PREFIX_PATH "${TorchCMakePrefixPath}")
get_filename_component(PARENT_DIR ${CMAKE_SOURCE_DIR} DIRECTORY)

# Third-party library paths
set(CUTLASS_DIR ${PARENT_DIR}/3rdparty/cutlass CACHE PATH "Path to modified cutlass src")
set(SPDLOG_DIR  ${PARENT_DIR}/3rdparty/spdlog CACHE PATH "Path to spdlog src")
set(FLASHINFER_DIR ${PARENT_DIR}/3rdparty/flashinfer CACHE PATH "Path to flashinfer src")
set(MSCCLPP_DIR /usr/local/mscclpp CACHE PATH "Path to mscclpp install")

# Include directories
include_directories(${MSCCLPP_DIR}/include)
include_directories(${CUTLASS_DIR}/include)
include_directories(${CUTLASS_DIR}/tools/util/include)
include_directories(${FLASHINFER_DIR}/include)

# Find libraries
set(Torch_KINETO_LIBRARIES "")
find_package(Torch REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")
find_library(MSCCLPP_LIBRARY NAMES mscclpp PATHS ${MSCCLPP_DIR}/lib NO_DEFAULT_PATH)

# ------------- Add FP8 Macro for compilation -----------------#
if(CMAKE_CUDA_ARCHITECTURES GREATER_EQUAL 89)
  add_compile_definitions(FLASHINFER_ENABLE_FP8)
endif()

add_compile_options(-Wall)
add_compile_options(-Wunused-variable)

# --------------- Build GEMM Libraries -----------------#

set (GEMM_BASE_SRC "${CMAKE_SOURCE_DIR}/src/generate-gemm")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${GEMM_BASE_SRC}/genGEMM.py ${GEMM_BASE_SRC}/gemmFactory.in)
execute_process(COMMAND make WORKING_DIRECTORY ${GEMM_BASE_SRC})
file(GLOB GEMM_SRC "${GEMM_BASE_SRC}/*.cu")
add_library(gemm_lib STATIC ${GEMM_SRC})
target_include_directories(gemm_lib PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(gemm_lib PRIVATE ${PARENT_DIR}/3rdparty/cutlass/include)
target_include_directories(gemm_lib PRIVATE ${PARENT_DIR}/3rdparty/cutlass/tools/util/include)

# -------------- Build FP16 loader -----------------#
find_package(Torch REQUIRED)            # only headers
find_library(URING_LIB uring REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")

pybind11_add_module(
  fast_uring
  src/fast_uring.cpp)

target_include_directories(fast_uring PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(fast_uring PRIVATE Python::Python ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY} ${URING_LIB})

# Helper function to create Python modules
function(add_pybind_module TARGET_NAME)
    cmake_parse_arguments(PARSE_ARGV 1 ARG "" "" "SOURCES;LIBS")
    
    # Create the Python module
    pybind11_add_module(${TARGET_NAME} ${ARG_SOURCES})
    target_link_libraries(${TARGET_NAME} PRIVATE 
        Python::Python 
        ${TORCH_LIBRARIES} 
        ${TORCH_PYTHON_LIBRARY} 
        ${ARG_LIBS}
    )
    target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/include)
    
    # Stub generation
    if(PYBIND11_STUBGEN_EXECUTABLE)
        add_custom_command(
            TARGET ${TARGET_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E env 
                PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}:$ENV{PYTHONPATH}
                ${PYBIND11_STUBGEN_EXECUTABLE}
                    --output-dir ${CMAKE_CURRENT_BINARY_DIR}
                    ${TARGET_NAME}
            COMMENT "Generating pybind11 stubs for ${TARGET_NAME}"
        )
    endif()
endfunction()

# Add modules
add_pybind_module(example_module SOURCES src/test.cu)
add_pybind_module(bind_genEmbedding SOURCES src/bind_genEmbedding.cu)
add_pybind_module(bind_rms_norm SOURCES src/bind_rms_norm.cu)
add_pybind_module(bind_silu_multiply SOURCES src/bind_silu_multiply.cu)
add_pybind_module(bind_sample SOURCES src/bind_sample.cu)
add_pybind_module(bind_gemm SOURCES src/bind_gemm.cu)
target_link_libraries(bind_gemm PRIVATE gemm_lib)
add_pybind_module(bind_green_ctx SOURCES src/bind_green_ctx.cu)
target_link_libraries(bind_green_ctx PRIVATE cuda)
add_pybind_module(bind_ropeappend SOURCES src/bind_ropeappend.cu)
# add_pybind_module(bind_init_net SOURCES src/bind_init_net.cu)
# target_link_libraries(bind_init_net PRIVATE ${MSCCLPP_LIBRARY})