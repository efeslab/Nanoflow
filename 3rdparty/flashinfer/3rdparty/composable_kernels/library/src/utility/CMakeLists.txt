add_library(utility STATIC
    device_memory.cpp
    host_tensor.cpp
    convolution_parameter.cpp
)

add_library(composable_kernel::utility ALIAS utility)
set_target_properties(utility PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_options(utility PRIVATE ${CMAKE_COMPILER_WARNINGS})
target_include_directories(utility PUBLIC
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/library/utility>"
)
if(WIN32)
    target_compile_definitions(utility PUBLIC NOMINMAX)
endif()

rocm_install(
    TARGETS utility
    EXPORT utilityTargets
)

rocm_install(
    EXPORT utilityTargets
    FILE composable_kernelutilityTargets.cmake
    NAMESPACE composable_kernel::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/composable_kernel
)

clang_tidy_check(utility)
