# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

add_subdirectory(mscclpp)
add_subdirectory(test)

add_custom_target(pytest_lib_copy ALL
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_BINARY_DIR}/mscclpp/_mscclpp.*.so
        ${CMAKE_CURRENT_SOURCE_DIR}/mscclpp
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_BINARY_DIR}/test/_ext.*.so
        ${CMAKE_CURRENT_SOURCE_DIR}/test/_cpp
    DEPENDS mscclpp_py mscclpp_py_test
)
