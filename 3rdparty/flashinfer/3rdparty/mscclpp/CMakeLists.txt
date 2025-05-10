# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

set(MSCCLPP_MAJOR "0")
set(MSCCLPP_MINOR "5")
set(MSCCLPP_PATCH "1")

set(MSCCLPP_SOVERSION ${MSCCLPP_MAJOR})
set(MSCCLPP_VERSION "${MSCCLPP_MAJOR}.${MSCCLPP_MINOR}.${MSCCLPP_PATCH}")

cmake_minimum_required(VERSION 3.25)
enable_language(CXX)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Options
option(ENABLE_TRACE "Enable tracing" OFF)
option(USE_NPKIT "Use NPKIT" ON)
option(BUILD_TESTS "Build tests" ON)
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)
option(USE_CUDA "Use NVIDIA/CUDA." OFF)
option(USE_ROCM "Use AMD/ROCm." OFF)
option(BYPASS_GPU_CHECK "Bypass GPU check." OFF)

if(BYPASS_GPU_CHECK)
    if(USE_CUDA)
        message("Bypassing GPU check: using NVIDIA/CUDA.")
        find_package(CUDAToolkit REQUIRED)
    elseif(USE_ROCM)
        message("Bypassing GPU check: using AMD/ROCm.")
        # Temporal fix for rocm5.6
        set(CMAKE_PREFIX_PATH "/opt/rocm;${CMAKE_PREFIX_PATH}")
        find_package(hip REQUIRED)
    else()
        message(FATAL_ERROR "Bypassing GPU check: neither NVIDIA/CUDA nor AMD/ROCm is specified.")
    endif()
else()
    # Detect GPUs
    include(CheckNvidiaGpu)
    include(CheckAmdGpu)
    if(NVIDIA_FOUND AND AMD_FOUND)
        message("Detected NVIDIA/CUDA and AMD/ROCm: prioritizing NVIDIA/CUDA.")
        set(USE_CUDA ON)
        set(USE_ROCM OFF)
    elseif(NVIDIA_FOUND)
        message("Detected NVIDIA/CUDA.")
        set(USE_CUDA ON)
        set(USE_ROCM OFF)
    elseif(AMD_FOUND)
        message("Detected AMD/ROCm.")
        set(USE_CUDA OFF)
        set(USE_ROCM ON)
    else()
        message(FATAL_ERROR "Neither NVIDIA/CUDA nor AMD/ROCm is found.")
    endif()
endif()

# Declare project
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
if(USE_CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall,-Wextra")
    project(mscclpp LANGUAGES CXX CUDA)

    # CUDA 11 or higher is required
    if(CUDAToolkit_VERSION_MAJOR LESS 11)
        message(FATAL_ERROR "CUDA 11 or higher is required but detected ${CUDAToolkit_VERSION}")
    endif()

    # Set CUDA architectures
    if(CUDAToolkit_VERSION_MAJOR GREATER_EQUAL 11)
        set(CMAKE_CUDA_ARCHITECTURES 80)
    endif()

    # Hopper architecture
    if(CUDAToolkit_VERSION_MAJOR GREATER_EQUAL 12)
        set(CMAKE_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES} 90)
    endif()

    set(GPU_LIBRARIES CUDA::cudart CUDA::cuda_driver)
    set(GPU_INCLUDE_DIRS ${CUDAToolkit_INCLUDE_DIRS})
else()
    set(CMAKE_HIP_STANDARD 17)
    set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -Wall -Wextra")
    project(mscclpp LANGUAGES CXX)

    set(CMAKE_HIP_ARCHITECTURES gfx90a gfx941 gfx942)

    set(GPU_LIBRARIES hip::device)
    set(GPU_INCLUDE_DIRS ${hip_INCLUDE_DIRS})
endif()

# Format targets
include(${PROJECT_SOURCE_DIR}/cmake/AddFormatTargets.cmake)

# Find ibverbs and libnuma
find_package(IBVerbs REQUIRED)
find_package(NUMA REQUIRED)
find_package(Threads REQUIRED)

include(FetchContent)
FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
FetchContent_MakeAvailable(json)

add_library(mscclpp_obj OBJECT)
target_include_directories(mscclpp_obj
    SYSTEM PRIVATE
    ${GPU_INCLUDE_DIRS}
    ${IBVERBS_INCLUDE_DIRS}
    ${NUMA_INCLUDE_DIRS})
target_link_libraries(mscclpp_obj PRIVATE ${GPU_LIBRARIES} ${NUMA_LIBRARIES} ${IBVERBS_LIBRARIES} nlohmann_json::nlohmann_json Threads::Threads)
set_target_properties(mscclpp_obj PROPERTIES LINKER_LANGUAGE CXX POSITION_INDEPENDENT_CODE 1 VERSION ${MSCCLPP_VERSION} SOVERSION ${MSCCLPP_SOVERSION})
if(USE_CUDA)
    target_compile_definitions(mscclpp_obj PRIVATE USE_CUDA)
elseif(USE_ROCM)
    target_compile_definitions(mscclpp_obj PRIVATE USE_ROCM)
endif()
if(ENABLE_TRACE)
    target_compile_definitions(mscclpp_obj PRIVATE ENABLE_TRACE)
endif()
if(USE_NPKIT)
    target_compile_definitions(mscclpp_obj PRIVATE ENABLE_NPKIT)
endif()

# libmscclpp
add_library(mscclpp SHARED)
target_link_libraries(mscclpp PUBLIC mscclpp_obj)
set_target_properties(mscclpp PROPERTIES VERSION ${MSCCLPP_VERSION} SOVERSION ${MSCCLPP_SOVERSION})
add_library(mscclpp_static STATIC)
target_link_libraries(mscclpp_static PUBLIC mscclpp_obj)
set_target_properties(mscclpp_static PROPERTIES VERSION ${MSCCLPP_VERSION} SOVERSION ${MSCCLPP_SOVERSION})

add_subdirectory(include)
add_subdirectory(src)

if("${INSTALL_PREFIX}" STREQUAL "")
    set(INSTALL_PREFIX "./")
endif()

install(TARGETS mscclpp_obj
    FILE_SET HEADERS DESTINATION ${INSTALL_PREFIX}/include)
install(TARGETS mscclpp
    LIBRARY DESTINATION ${INSTALL_PREFIX}/lib)
install(TARGETS mscclpp_static
    ARCHIVE DESTINATION ${INSTALL_PREFIX}/lib)

# Tests
if(BUILD_TESTS)
    enable_testing() # Called here to allow ctest from the build directory
    add_subdirectory(test)
endif()

# Python bindings
if(BUILD_PYTHON_BINDINGS)
    add_subdirectory(python)
endif()
