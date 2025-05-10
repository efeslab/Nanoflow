set(EXAMPLE_REDUCE "tile_example_reduce")
# not using add_example_executable() to add this target, since we don't want this to have
# to be included in "make all/install/check"
message("adding example ${EXAMPLE_REDUCE}")

add_executable(${EXAMPLE_REDUCE} EXCLUDE_FROM_ALL reduce.cpp)
target_include_directories(${EXAMPLE_REDUCE} PRIVATE ${CMAKE_CURRENT_LIST_DIR})
set(EXAMPLE_REDUCE_COMPILE_OPTIONS)

# NOTE: we turn off undefined-func-template to let source compile without explicit declare function specializations
list(APPEND EXAMPLE_REDUCE_COMPILE_OPTIONS -Wno-undefined-func-template -Wno-float-equal)

target_compile_options(${EXAMPLE_REDUCE} PRIVATE ${EXAMPLE_REDUCE_COMPILE_OPTIONS})

# TODO: we have to turn off this global prop, otherwise the progress bar generated
# by cmake will print too many files, execvp: /bin/sh: Argument list too long
# however, this property may affect global
# TODO: consider codegen a makefile by us
set_property(GLOBAL PROPERTY RULE_MESSAGES OFF)