# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

find_package(MPI)

set(TEST_LIBS_COMMON mscclpp ${GPU_LIBRARIES} ${NUMA_LIBRARIES} ${IBVERBS_LIBRARIES} Threads::Threads)
set(TEST_LIBS_GTEST GTest::gtest_main GTest::gmock_main)
set(TEST_INC_COMMON PRIVATE ${PROJECT_SOURCE_DIR}/include SYSTEM PRIVATE ${GPU_INCLUDE_DIRS})
set(TEST_INC_INTERNAL PRIVATE ${PROJECT_SOURCE_DIR}/src/include)

if(USE_ROCM)
    file(GLOB_RECURSE CU_SOURCES CONFIGURE_DEPENDS *.cu)
    set_source_files_properties(${CU_SOURCES} PROPERTIES LANGUAGE CXX)
endif()

function(add_test_executable name sources)
    add_executable(${name} ${sources})
    target_link_libraries(${name} ${TEST_LIBS_COMMON} MPI::MPI_CXX)
    target_include_directories(${name} ${TEST_INC_COMMON} ${TEST_INC_INTERNAL})
    target_compile_definitions(${name} PRIVATE MSCCLPP_USE_MPI_FOR_TESTS)
    add_test(NAME ${name} COMMAND ${CMAKE_CURRENT_BINARY_DIR}/run_mpi_test.sh ${name} 2)
endfunction()

add_test_executable(allgather_test_cpp allgather_test_cpp.cu)
add_test_executable(allgather_test_host_offloading allgather_test_host_offloading.cu)
add_test_executable(nvls_test nvls_test.cu)

configure_file(run_mpi_test.sh.in run_mpi_test.sh)

include(CTest)
include(FetchContent)
FetchContent_Declare(googletest URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip)
option(INSTALL_GTEST OFF)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)

# Unit tests
add_executable(unit_tests)
target_link_libraries(unit_tests ${TEST_LIBS_COMMON} ${TEST_LIBS_GTEST})
target_include_directories(unit_tests ${TEST_INC_COMMON} ${TEST_INC_INTERNAL})
add_subdirectory(unit)
gtest_discover_tests(unit_tests DISCOVERY_MODE PRE_TEST)

# Multi-process unit tests
add_executable(mp_unit_tests)
target_link_libraries(mp_unit_tests ${TEST_LIBS_COMMON} ${TEST_LIBS_GTEST} MPI::MPI_CXX)
target_include_directories(mp_unit_tests ${TEST_INC_COMMON} ${TEST_INC_INTERNAL})
add_subdirectory(mp_unit)
gtest_discover_tests(mp_unit_tests DISCOVERY_MODE PRE_TEST)

# mscclpp-test
add_subdirectory(mscclpp-test)
