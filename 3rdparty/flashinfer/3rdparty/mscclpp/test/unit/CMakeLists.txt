# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

target_sources(unit_tests PRIVATE
    core_tests.cc
    cuda_utils_tests.cc
    errors_tests.cc
    fifo_tests.cu
    numa_tests.cc
    socket_tests.cc
    utils_tests.cc
    utils_internal_tests.cc
    compile_tests.cu
)
