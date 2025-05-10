# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz)
FetchContent_MakeAvailable(json)

function(add_mscclpp_test_executable name sources)
    if(USE_ROCM)
        set_source_files_properties(${sources} PROPERTIES LANGUAGE CXX)
    endif()
    add_executable(${name} ${sources} common.cc)
    target_link_libraries(${name} ${TEST_LIBS_COMMON} MPI::MPI_CXX nlohmann_json::nlohmann_json)
    target_include_directories(${name} ${TEST_INC_COMMON})
endfunction()

add_mscclpp_test_executable(sendrecv_test_perf sendrecv_test.cu)
add_mscclpp_test_executable(allgather_test_perf allgather_test.cu)
add_mscclpp_test_executable(allreduce_test_perf allreduce_test.cu)
add_mscclpp_test_executable(alltoall_test_perf alltoall_test.cu)
