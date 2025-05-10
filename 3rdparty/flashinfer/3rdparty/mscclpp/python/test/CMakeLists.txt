# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)
include(FetchContent)
FetchContent_Declare(nanobind GIT_REPOSITORY https://github.com/wjakob/nanobind.git GIT_TAG v1.4.0)
FetchContent_MakeAvailable(nanobind)

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS *.cpp)
nanobind_add_module(mscclpp_py_test ${SOURCES})
set_target_properties(mscclpp_py_test PROPERTIES OUTPUT_NAME _ext)
target_link_libraries(mscclpp_py_test PRIVATE mscclpp_static ${GPU_LIBRARIES})
target_include_directories(mscclpp_py_test SYSTEM PRIVATE ${GPU_INCLUDE_DIRS})
