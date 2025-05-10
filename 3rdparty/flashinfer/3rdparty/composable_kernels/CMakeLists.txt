cmake_minimum_required(VERSION 3.14)
if(POLICY CMP0140)
  # policies CMP0140 not known to CMake until 3.25
  cmake_policy(SET CMP0140 NEW)
endif()

get_property(_GENERATOR_IS_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)

# This has to be initialized before the project() command appears
# Set the default of CMAKE_BUILD_TYPE to be release, unless user specifies with -D.  MSVC_IDE does not use CMAKE_BUILD_TYPE
if(_GENERATOR_IS_MULTI_CONFIG)
    set(CMAKE_CONFIGURATION_TYPES "Debug;Release;RelWithDebInfo;MinSizeRel" CACHE STRING
            "Available build types (configurations) on multi-config generators")
else()
    set(CMAKE_BUILD_TYPE Release CACHE STRING
            "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel.")
endif()

# Default installation path
if(NOT WIN32)
    set(CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH "")
endif()

set(version 1.1.0)
# Check support for CUDA/HIP in Cmake
project(composable_kernel VERSION ${version} LANGUAGES CXX HIP)
include(CTest)

find_package(Python3 3.6 COMPONENTS Interpreter REQUIRED)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

if (DTYPES)
    add_definitions(-DDTYPES)
    if (DTYPES MATCHES "int8")
        add_definitions(-DCK_ENABLE_INT8)
        set(CK_ENABLE_INT8 "ON")
    endif()
    if (DTYPES MATCHES "fp8")
        add_definitions(-DCK_ENABLE_FP8)
        set(CK_ENABLE_FP8 "ON")
    endif()
    if (DTYPES MATCHES "bf8")
        add_definitions(-DCK_ENABLE_BF8)
        set(CK_ENABLE_BF8 "ON")
    endif()
    if (DTYPES MATCHES "fp16")
        add_definitions(-DCK_ENABLE_FP16)
        set(CK_ENABLE_FP16 "ON")
    endif()
    if (DTYPES MATCHES "fp32")
        add_definitions(-DCK_ENABLE_FP32)
        set(CK_ENABLE_FP32 "ON")
    endif()
    if (DTYPES MATCHES "fp64")
        add_definitions(-DCK_ENABLE_FP64)
        set(CK_ENABLE_FP64 "ON")
    endif()
    if (DTYPES MATCHES "bf16")
        add_definitions(-DCK_ENABLE_BF16)
        set(CK_ENABLE_BF16 "ON")
    endif()
    message("DTYPES macro set to ${DTYPES}")
else()
    add_definitions(-DCK_ENABLE_INT8 -DCK_ENABLE_FP8 -DCK_ENABLE_BF8 -DCK_ENABLE_FP16 -DCK_ENABLE_FP32 -DCK_ENABLE_FP64 -DCK_ENABLE_BF16)
    set(CK_ENABLE_ALL_DTYPES "ON")
endif()

#for f8/bf8_t type
add_compile_options(-Wno-bit-int-extension)
add_compile_options(-Wno-pass-failed)
add_compile_options(-Wno-switch-default)

if(DL_KERNELS)
    add_definitions(-DDL_KERNELS)
    set(CK_ENABLE_DL_KERNELS "ON")
endif()

if(INSTANCES_ONLY)
    add_definitions(-DINSTANCES_ONLY)
    set(CK_ENABLE_INSTANCES_ONLY "ON")
endif()

include(getopt)

# CK version file to record release version as well as git commit hash
find_package(Git REQUIRED)
execute_process(COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD OUTPUT_VARIABLE COMMIT_ID OUTPUT_STRIP_TRAILING_WHITESPACE)
configure_file(include/ck/version.h.in ${CMAKE_CURRENT_BINARY_DIR}/include/ck/version.h)

set(ROCM_SYMLINK_LIBS OFF)
find_package(ROCM REQUIRED PATHS /opt/rocm)

include(ROCMInstallTargets)
include(ROCMPackageConfigHelpers)
include(ROCMSetupVersion)
include(ROCMInstallSymlinks)
include(ROCMCreatePackage)
include(CheckCXXCompilerFlag)
include(ROCMCheckTargetIds)
include(TargetFlags)

rocm_setup_version(VERSION ${version})

list(APPEND CMAKE_PREFIX_PATH ${CMAKE_INSTALL_PREFIX} ${CMAKE_INSTALL_PREFIX}/llvm ${CMAKE_INSTALL_PREFIX}/hip /opt/rocm /opt/rocm/llvm /opt/rocm/hip "$ENV{ROCM_PATH}" "$ENV{HIP_PATH}")

message("GPU_TARGETS= ${GPU_TARGETS}")

message("checking which targets are supported")
#This is the list of targets to be used in case GPU_TARGETS is not set on command line
#These targets will be filtered and only supported ones will be used
#Setting GPU_TARGETS on command line will override this list
if(NOT PROFILER_ONLY)
    rocm_check_target_ids(DEFAULT_GPU_TARGETS
        TARGETS "gfx900;gfx906;gfx908;gfx90a;gfx940;gfx941;gfx942;gfx1030;gfx1100;gfx1101;gfx1102")
else()
    add_definitions(-DPROFILER_ONLY)
    set(GPU_TARGETS "" CACHE STRING "" FORCE)
    if(GPU_TARGETS)
        message(FATAL_ERROR "For PROFILE_ONLY build, please do not set GPU_TARGETS, use GPU_ARCH = gfx90, gfx94, gfx10, or gfx11")
    endif()
    if(GPU_ARCH MATCHES "gfx90")
        rocm_check_target_ids(DEFAULT_GPU_TARGETS TARGETS "gfx908;gfx90a")
    elseif(GPU_ARCH MATCHES "gfx94")
        rocm_check_target_ids(DEFAULT_GPU_TARGETS TARGETS "gfx940;gfx941;gfx942")
    elseif(GPU_ARCH MATCHES "gfx10")
        rocm_check_target_ids(DEFAULT_GPU_TARGETS TARGETS "gfx1030")
    elseif(GPU_ARCH MATCHES "gfx11")
        rocm_check_target_ids(DEFAULT_GPU_TARGETS TARGETS "gfx1100;gfx1101;gfx1102")
    else()
        message(FATAL_ERROR "For PROFILE_ONLY build, please specify GPU_ARCH as gfx90, gfx94, gfx10, or gfx11")
    endif()
    set(GPU_TARGETS "${DEFAULT_GPU_TARGETS}" CACHE STRING " " FORCE)
endif()

message("Supported GPU_TARGETS= ${DEFAULT_GPU_TARGETS}")

if(GPU_TARGETS)
    message("Building CK for the following targets: ${GPU_TARGETS}")
else()
    message("Building CK for the default targets: ${DEFAULT_GPU_TARGETS}")
endif()

if (GPU_TARGETS)
    if (GPU_TARGETS MATCHES "gfx9")
        add_definitions(-DCK_USE_XDL)
        set(CK_USE_XDL "ON")
    endif()
    if (GPU_TARGETS MATCHES "gfx11")
        add_definitions(-DCK_USE_WMMA)
        set(CK_USE_WMMA "ON")
    endif()
else()
    add_definitions(-DCK_USE_WMMA -DCK_USE_XDL)
    set(CK_USE_XDL "ON")
    set(CK_USE_WMMA "ON")
endif()

# CK config file to record supported datatypes, etc.
configure_file(include/ck/config.h.in ${CMAKE_CURRENT_BINARY_DIR}/include/ck/config.h)

find_package(hip)
# No assumption that HIP kernels are launched with uniform block size for backward compatibility
# SWDEV-413293 and https://reviews.llvm.org/D155213
math(EXPR hip_VERSION_FLAT "(${hip_VERSION_MAJOR} * 1000 + ${hip_VERSION_MINOR}) * 100000 + ${hip_VERSION_PATCH}")
message("hip_version_flat=${hip_VERSION_FLAT}")
if(NOT WIN32 AND ${hip_VERSION_FLAT} GREATER 500723302)
   message("Adding the fno-offload-uniform-block compiler flag")
   add_compile_options(-fno-offload-uniform-block)
endif()
if(NOT WIN32 AND ${hip_VERSION_FLAT} GREATER 600140090)
   message("Adding the enable-post-misched=0 compiler flag")
   add_compile_options(-mllvm -enable-post-misched=0)
endif()
#
# Seperate linking jobs from compiling
# Too many concurrent linking jobs can break the build
# Copied from LLVM
set(CK_PARALLEL_LINK_JOBS "" CACHE STRING
  "Define the maximum number of concurrent link jobs (Ninja only).")
if(CMAKE_GENERATOR MATCHES "Ninja")
  if(CK_PARALLEL_LINK_JOBS)
    set_property(GLOBAL APPEND PROPERTY JOB_POOLS link_job_pool=${CK_PARALLEL_LINK_JOBS})
    set(CMAKE_JOB_POOL_LINK link_job_pool)
  endif()
elseif(CK_PARALLEL_LINK_JOBS)
  message(WARNING "Job pooling is only available with Ninja generators.")
endif()
# Similar for compiling
set(CK_PARALLEL_COMPILE_JOBS "" CACHE STRING
  "Define the maximum number of concurrent compile jobs (Ninja only).")
if(CMAKE_GENERATOR MATCHES "Ninja")
  if(CK_PARALLEL_COMPILE_JOBS)
    set_property(GLOBAL APPEND PROPERTY JOB_POOLS compile_job_pool=${CK_PARALLEL_COMPILE_JOBS})
    set(CMAKE_JOB_POOL_COMPILE compile_job_pool)
  endif()
elseif(CK_PARALLEL_COMPILE_JOBS)
  message(WARNING "Job pooling is only available with Ninja generators.")
endif()


option(USE_BITINT_EXTENSION_INT4 "Whether to enable clang's BitInt extension to provide int4 data type." OFF)
option(USE_OPT_GFX11 "Whether to enable LDS cumode and Wavefront32 mode for GFX11 silicons." OFF)

if(USE_BITINT_EXTENSION_INT4)
    add_compile_definitions(CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4)
    add_compile_options(-Wno-bit-int-extension)
    message("CK compiled with USE_BITINT_EXTENSION_INT4 set to ${USE_BITINT_EXTENSION_INT4}")
endif()

if(USE_OPT_GFX11)
    add_compile_options(-mcumode)
    add_compile_options(-mno-wavefrontsize64)
    message("CK compiled with USE_OPT_GFX11 set to ${USE_OPT_GFX11}")
endif()

## Threads
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
link_libraries(Threads::Threads)

## C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
message("CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")

## HIP
set(CMAKE_HIP_PLATFORM amd)
set(CMAKE_HIP_COMPILER ${CMAKE_CXX_COMPILER})
set(CMAKE_HIP_EXTENSIONS ON)
message("CMAKE_HIP_COMPILER: ${CMAKE_HIP_COMPILER}")

## OpenMP
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
	# workaround issue hipcc in rocm3.5 cannot find openmp
	set(OpenMP_CXX "${CMAKE_CXX_COMPILER}")
	set(OpenMP_CXX_FLAGS "-fopenmp=libomp -Wno-unused-command-line-argument")
	set(OpenMP_CXX_LIB_NAMES "libomp" "libgomp" "libiomp5")
	set(OpenMP_libomp_LIBRARY ${OpenMP_CXX_LIB_NAMES})
	set(OpenMP_libgomp_LIBRARY ${OpenMP_CXX_LIB_NAMES})
	set(OpenMP_libiomp5_LIBRARY ${OpenMP_CXX_LIB_NAMES})
else()
	find_package(OpenMP REQUIRED)
endif()

message("OpenMP_CXX_LIB_NAMES: ${OpenMP_CXX_LIB_NAMES}")
message("OpenMP_gomp_LIBRARY: ${OpenMP_gomp_LIBRARY}")
message("OpenMP_pthread_LIBRARY: ${OpenMP_pthread_LIBRARY}")
message("OpenMP_CXX_FLAGS: ${OpenMP_CXX_FLAGS}")

link_libraries(${OpenMP_gomp_LIBRARY})
link_libraries(${OpenMP_pthread_LIBRARY})

## HIP
find_package(HIP REQUIRED)
# Override HIP version in config.h, if necessary.
# The variables set by find_package() can't be overwritten,
# therefore let's use intermediate variables.
set(CK_HIP_VERSION_MAJOR "${HIP_VERSION_MAJOR}")
set(CK_HIP_VERSION_MINOR "${HIP_VERSION_MINOR}")
set(CK_HIP_VERSION_PATCH "${HIP_VERSION_PATCH}")
if( DEFINED CK_OVERRIDE_HIP_VERSION_MAJOR )
    set(CK_HIP_VERSION_MAJOR "${CK_OVERRIDE_HIP_VERSION_MAJOR}")
    message(STATUS "CK_HIP_VERSION_MAJOR overriden with ${CK_OVERRIDE_HIP_VERSION_MAJOR}")
endif()
if( DEFINED CK_OVERRIDE_HIP_VERSION_MINOR )
    set(CK_HIP_VERSION_MINOR "${CK_OVERRIDE_HIP_VERSION_MINOR}")
    message(STATUS "CK_HIP_VERSION_MINOR overriden with ${CK_OVERRIDE_HIP_VERSION_MINOR}")
endif()
if( DEFINED CK_OVERRIDE_HIP_VERSION_PATCH )
    set(CK_HIP_VERSION_PATCH "${CK_OVERRIDE_HIP_VERSION_PATCH}")
    message(STATUS "CK_HIP_VERSION_PATCH overriden with ${CK_OVERRIDE_HIP_VERSION_PATCH}")
endif()
message(STATUS "Build with HIP ${HIP_VERSION}")
link_libraries(hip::device)
if(CK_hip_VERSION VERSION_GREATER_EQUAL 6.0.23494)
    add_compile_definitions(__HIP_PLATFORM_AMD__=1)
else()
    add_compile_definitions(__HIP_PLATFORM_HCC__=1)
endif()

## tidy
include(EnableCompilerWarnings)
set(CK_TIDY_ERRORS ERRORS * -readability-inconsistent-declaration-parameter-name)
if(CMAKE_CXX_COMPILER MATCHES ".*hcc" OR CMAKE_CXX_COMPILER MATCHES ".*clang\\+\\+")
    set(CK_TIDY_CHECKS -modernize-use-override -readability-non-const-parameter)
# Enable tidy on hip
elseif(CK_BACKEND STREQUAL "HIP" OR CK_BACKEND STREQUAL "HIPNOGPU")
    set(CK_TIDY_ERRORS ALL)
endif()


include(ClangTidy)
enable_clang_tidy(
    CHECKS
        *
        -abseil-*
        -android-cloexec-fopen
        # Yea we shouldn't be using rand()
        -cert-msc30-c
        -bugprone-exception-escape
        -bugprone-macro-parentheses
        -cert-env33-c
        -cert-msc32-c
        -cert-msc50-cpp
        -cert-msc51-cpp
        -cert-dcl37-c
        -cert-dcl51-cpp
        -clang-analyzer-alpha.core.CastToStruct
        -clang-analyzer-optin.performance.Padding
        -clang-diagnostic-deprecated-declarations
        -clang-diagnostic-extern-c-compat
        -clang-diagnostic-unused-command-line-argument
        -cppcoreguidelines-avoid-c-arrays
        -cppcoreguidelines-avoid-magic-numbers
        -cppcoreguidelines-explicit-virtual-functions
        -cppcoreguidelines-init-variables
        -cppcoreguidelines-macro-usage
        -cppcoreguidelines-non-private-member-variables-in-classes
        -cppcoreguidelines-pro-bounds-array-to-pointer-decay
        -cppcoreguidelines-pro-bounds-constant-array-index
        -cppcoreguidelines-pro-bounds-pointer-arithmetic
        -cppcoreguidelines-pro-type-member-init
        -cppcoreguidelines-pro-type-reinterpret-cast
        -cppcoreguidelines-pro-type-union-access
        -cppcoreguidelines-pro-type-vararg
        -cppcoreguidelines-special-member-functions
        -fuchsia-*
        -google-explicit-constructor
        -google-readability-braces-around-statements
        -google-readability-todo
        -google-runtime-int
        -google-runtime-references
        -hicpp-vararg
        -hicpp-braces-around-statements
        -hicpp-explicit-conversions
        -hicpp-named-parameter
        -hicpp-no-array-decay
        # We really shouldn't use bitwise operators with signed integers, but
        # opencl leaves us no choice
        -hicpp-avoid-c-arrays
        -hicpp-signed-bitwise
        -hicpp-special-member-functions
        -hicpp-uppercase-literal-suffix
        -hicpp-use-auto
        -hicpp-use-equals-default
        -hicpp-use-override
        -llvm-header-guard
        -llvm-include-order
        #-llvmlibc-*
        -llvmlibc-restrict-system-libc-headers
        -llvmlibc-callee-namespace
        -llvmlibc-implementation-in-namespace
        -llvm-else-after-return
        -llvm-qualified-auto
        -misc-misplaced-const
        -misc-non-private-member-variables-in-classes
        -misc-no-recursion
        -modernize-avoid-bind
        -modernize-avoid-c-arrays
        -modernize-pass-by-value
        -modernize-use-auto
        -modernize-use-default-member-init
        -modernize-use-equals-default
        -modernize-use-trailing-return-type
        -modernize-use-transparent-functors
        -performance-unnecessary-value-param
        -readability-braces-around-statements
        -readability-else-after-return
        # we are not ready to use it, but very useful
        -readability-function-cognitive-complexity
        -readability-isolate-declaration
        -readability-magic-numbers
        -readability-named-parameter
        -readability-uppercase-literal-suffix
        -readability-convert-member-functions-to-static
        -readability-qualified-auto
        -readability-redundant-string-init
        # too many narrowing conversions in our code
        -bugprone-narrowing-conversions
        -cppcoreguidelines-narrowing-conversions
        -altera-struct-pack-align
        -cppcoreguidelines-prefer-member-initializer
        ${CK_TIDY_CHECKS}
        ${CK_TIDY_ERRORS}
    HEADER_FILTER
        "\.hpp$"
    EXTRA_ARGS
        -DCK_USE_CLANG_TIDY
)

include(CppCheck)
enable_cppcheck(
    CHECKS
        warning
        style
        performance
        portability
    SUPPRESS
        ConfigurationNotChecked
        constStatement
        duplicateCondition
        noExplicitConstructor
        passedByValue
        preprocessorErrorDirective
        shadowVariable
        unusedFunction
        unusedPrivateFunction
        unusedStructMember
        unmatchedSuppression
    FORCE
    SOURCES
        library/src
    INCLUDE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_BINARY_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/library/include
    DEFINE
        CPPCHECK=1
        __linux__=1
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)

# set CK project include directories
include_directories(BEFORE
    ${PROJECT_BINARY_DIR}/include
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/library/include
    ${HIP_INCLUDE_DIRS}
)

SET(BUILD_DEV ON CACHE BOOL "BUILD_DEV")
if(BUILD_DEV)
    add_compile_options(-Werror)
    add_compile_options(-Weverything)
endif()
message("CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")

add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -C ${CMAKE_CFG_INTDIR})

file(GLOB_RECURSE INSTANCE_FILES "${PROJECT_SOURCE_DIR}/*/device_*_instance.cpp")
file(GLOB dir_list RELATIVE ${PROJECT_SOURCE_DIR}/library/src/tensor_operation_instance/gpu ${PROJECT_SOURCE_DIR}/library/src/tensor_operation_instance/gpu/*)
set(CK_DEVICE_INSTANCES)
FOREACH(subdir_path ${dir_list})
set(target_dir)
IF(IS_DIRECTORY "${PROJECT_SOURCE_DIR}/library/src/tensor_operation_instance/gpu/${subdir_path}")
    set(cmake_instance)
    file(READ "${PROJECT_SOURCE_DIR}/library/src/tensor_operation_instance/gpu/${subdir_path}/CMakeLists.txt" cmake_instance)
    set(add_inst 0)
    if(("${cmake_instance}" MATCHES "fp8" OR "${cmake_instance}" MATCHES "_f8") AND DTYPES MATCHES "fp8")
        set(add_inst 1)
    endif()
    if(("${cmake_instance}" MATCHES "bf8" OR "${cmake_instance}" MATCHES "_b8") AND DTYPES MATCHES "bf8")
        set(add_inst 1)
    endif()
    if(("${cmake_instance}" MATCHES "fp16" OR "${cmake_instance}" MATCHES "_f16") AND DTYPES MATCHES "fp16")
        set(add_inst 1)
    endif()
    if(("${cmake_instance}" MATCHES "fp32" OR "${cmake_instance}" MATCHES "_f32") AND DTYPES MATCHES "fp32")
        set(add_inst 1)
    endif()
    if(("${cmake_instance}" MATCHES "fp64" OR "${cmake_instance}" MATCHES "_f64") AND DTYPES MATCHES "fp64")
        set(add_inst 1)
    endif()
    if(("${cmake_instance}" MATCHES "bf16" OR "${cmake_instance}" MATCHES "_b16") AND DTYPES MATCHES "bf16")
        set(add_inst 1)
    endif()
    if(("${cmake_instance}" MATCHES "int8" OR "${cmake_instance}" MATCHES "_i8") AND DTYPES MATCHES "int8")
        set(add_inst 1)
    endif()
    if(NOT "${cmake_instance}" MATCHES "DTYPES")
        set(add_inst 1)
    endif()
    if(add_inst EQUAL 1 OR NOT DEFINED DTYPES)
        list(APPEND CK_DEVICE_INSTANCES device_${subdir_path}_instance)
    endif()
ENDIF()
ENDFOREACH()

add_custom_target(instances DEPENDS utility;${CK_DEVICE_INSTANCES}  SOURCES ${INSTANCE_FILES})
add_subdirectory(library)

if(NOT DEFINED INSTANCES_ONLY)
 if(NOT DEFINED PROFILER_ONLY)
   rocm_package_setup_component(tests
        LIBRARY_NAME composablekernel
        PACKAGE_NAME tests # Prevent -static suffix on package name
   )

   rocm_package_setup_component(examples
        LIBRARY_NAME composablekernel
        PACKAGE_NAME examples
   )
   add_subdirectory(example)
   if(BUILD_TESTING)
      add_subdirectory(test)
   endif()

   rocm_package_setup_component(profiler
        LIBRARY_NAME composablekernel
        PACKAGE_NAME ckprofiler
   )
   add_subdirectory(profiler)
  else()
    #When building PROFILER_ONLY, label the package with GPU_ARCH
    rocm_package_setup_component(profiler
       LIBRARY_NAME composablekernel
       PACKAGE_NAME ckprofiler_${GPU_ARCH}
    )
    add_subdirectory(profiler)
  endif()
endif()

#Create an interface target for the include only files and call it "composablekernels"
include(CMakePackageConfigHelpers)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/composable_kernelConfigVersion.cmake"
    VERSION "${version}"
    COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/composable_kernelConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/composable_kernel
    NO_CHECK_REQUIRED_COMPONENTS_MACRO
)

rocm_install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/composable_kernelConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/composable_kernelConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/composable_kernel
)

# Install CK version and configuration files
rocm_install(FILES
    ${PROJECT_BINARY_DIR}/include/ck/version.h
    ${PROJECT_BINARY_DIR}/include/ck/config.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ck/
)

set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set(CPACK_RPM_PACKAGE_LICENSE "MIT")

rocm_create_package(
    NAME composablekernel
    DESCRIPTION "High Performance Composable Kernel for AMD GPUs"
    MAINTAINER "MIOpen Kernels Dev Team <dl.MIOpen@amd.com>"
    LDCONFIG
    HEADER_ONLY
)
