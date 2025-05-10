cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(example_gemm_xdl_fp16_v3 LANGUAGES CXX HIP)

#
# --- C++ standard settings ---
#
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release) # Key to not segfaulting
endif()

# Execute Python to get the Torch CMake prefix path
execute_process(
  COMMAND python -c "import torch; print(torch.utils.cmake_prefix_path)"
  OUTPUT_VARIABLE TorchCMakePrefixPath
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
# Force usage of the old C++ ABI (ABI0)
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=1)

# Append the Torch CMake prefix to the existing CMAKE_PREFIX_PATH
# You can also prepend it depending on your needs.
list(APPEND CMAKE_PREFIX_PATH "${TorchCMakePrefixPath}")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_HIP_EXTENSIONS ON)   # HIP is an extension of Clang under the hood
set(CMAKE_CXX_EXTENSIONS ON)
#
# --- HIP setup (Clang + AMD) ---
#
# If you have a different HIP installation layout, adjust as necessary.
#
find_package(HIP REQUIRED)
# print hip directories

set(CMAKE_HIP_PLATFORM "AMD")
set(CMAKE_HIP_ARCHITECTURES "gfx90a")  # You can specify a target arch if desired, e.g. gfx90a

#
# --- Point to composable-kernel headers and library ---
#

set(COMPOSIBLE_KERNEL_DIR "/work1/kasikci/kanzhu/myCK/3rdparty/composable_kernel")
# set(COMPOSIBLE_KERNEL_DIR "{CMAKE_CURRENT_SOURCE_DIR}/composable_kernel")
# Adjust these to match where you placed composable-kernel
include_directories(
    "${COMPOSIBLE_KERNEL_DIR}/include"
    "${COMPOSIBLE_KERNEL_DIR}/library/include"
    "${COMPOSIBLE_KERNEL_DIR}/build/include"
    ${HIP_INCLUDE_DIR}
)

link_directories("${COMPOSIBLE_KERNEL_DIR}/build/lib")

# add -fPIC
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

#
# --- Build our single example ---
#

set_source_files_properties(gemm_lib.cpp PROPERTIES LANGUAGE HIP)
add_library(gemm_lib SHARED
    gemm_lib.cpp
)
target_link_libraries(
    gemm_lib PRIVATE
    utility
    device_gemm_operations
)


set_source_files_properties(gemm_lib_universal.cpp PROPERTIES LANGUAGE HIP)
add_library(gemm_lib_universal SHARED
    gemm_lib_universal.cpp
)
target_link_libraries(
    gemm_lib_universal PRIVATE
    utility
    device_gemm_operations
)

execute_process(
  COMMAND python -m pybind11 --cmakedir
  OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

find_program(PYBIND11_STUBGEN_EXECUTABLE NAMES pybind11-stubgen)

# Append the directory to CMake's search path
list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")

find_package(pybind11 REQUIRED)
message(STATUS "pybind11 include: ${pybind11_INCLUDE_DIRS}")

find_package(Torch REQUIRED)
message(STATUS "Torch include: ${TORCH_INCLUDE_DIRS}")
find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")

function(add_pybind_module TARGET_NAME)
    cmake_parse_arguments(PARSE_ARGV 1 ARG "" "" "SOURCES;LIBS")
    
    # Create the Python module
    pybind11_add_module(${TARGET_NAME} ${ARG_SOURCES})
    target_link_libraries(${TARGET_NAME} PRIVATE 
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

# add_pybind_module(test_bind_torch
#     SOURCES test_bind_torch.cpp
# )

set_source_files_properties(bind_ck.cpp PROPERTIES LANGUAGE HIP)
add_pybind_module(bind_ck
    SOURCES bind_ck.cpp
    LIBS gemm_lib gemm_lib_universal
)
