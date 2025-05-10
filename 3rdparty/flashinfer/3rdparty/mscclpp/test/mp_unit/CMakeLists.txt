# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

target_sources(mp_unit_tests PRIVATE
    mp_unit_tests.cc
    bootstrap_tests.cc
    ib_tests.cu
    communicator_tests.cu
    proxy_channel_tests.cu
    sm_channel_tests.cu
    executor_tests.cc
)
