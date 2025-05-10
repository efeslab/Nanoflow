set(srcs
  axes_metadata.cxx
  axis_base.cxx
  benchmark_base.cxx
  benchmark_manager.cxx
  blocking_kernel.cu
  criterion_manager.cxx
  csv_printer.cu
  cuda_call.cu
  device_info.cu
  device_manager.cu
  float64_axis.cxx
  int64_axis.cxx
  markdown_printer.cu
  named_values.cxx
  option_parser.cu
  printer_base.cxx
  printer_multiplex.cxx
  runner.cxx
  state.cxx
  stopping_criterion.cxx
  string_axis.cxx
  type_axis.cxx
  type_strings.cxx

  detail/entropy_criterion.cxx
  detail/measure_cold.cu
  detail/measure_hot.cu
  detail/state_generator.cxx
  detail/stdrel_criterion.cxx

  internal/nvml.cxx
)

if (NVBench_ENABLE_CUPTI)
  list(APPEND srcs detail/measure_cupti.cu cupti_profiler.cxx)
endif()

# CUDA 11.0 can't compile json_printer without crashing
# So for that version fall back to C++ with degraded
# output ( no PTX version info )
if(CMAKE_CUDA_COMPILER_ID STREQUAL NVIDIA AND
   CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11.1)
  set(json_printer_impl json_printer.cxx)
  set(json_is_cu FALSE)
else()
  set(json_printer_impl json_printer.cu)
  set(json_is_cu TRUE)
endif()
list(APPEND srcs ${json_printer_impl})

# Generate doc strings from md files:
include("../cmake/FileToString.cmake")
file_to_string("../docs/cli_help.md"
  "${NVBench_BINARY_DIR}/nvbench/internal/cli_help.cuh"
  ""
  cli_help_text
)
file_to_string("../docs/cli_help_axis.md"
  "${NVBench_BINARY_DIR}/nvbench/internal/cli_help_axis.cuh"
  ""
  cli_help_axis_text
)

nvbench_write_config_header(config.cuh.in
  "${NVBench_BINARY_DIR}/nvbench/config.cuh"
)

# nvbench (nvbench::nvbench)
add_library(nvbench ${srcs})
nvbench_config_target(nvbench)
target_include_directories(nvbench PUBLIC
  "$<BUILD_INTERFACE:${NVBench_SOURCE_DIR}>"
  "$<BUILD_INTERFACE:${NVBench_BINARY_DIR}>"
  "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)
target_link_libraries(nvbench
  PUBLIC
    ${ctk_libraries}
  PRIVATE
    fmt::fmt
    nvbench_json
)

# ##################################################################################################
# * conda environment -----------------------------------------------------------------------------
rapids_cmake_support_conda_env(conda_env MODIFY_PREFIX_PATH)
if(TARGET conda_env)
  # When we are inside a conda env the linker will be set to
  # `ld.bfd` which will try to resolve all undefined symbols at link time.
  #
  # Since we could be using a shared library version of fmt we need
  # it on the final link line of consumers
  target_link_libraries(nvbench PRIVATE $<BUILD_INTERFACE:conda_env>)

  # When we are inside a conda env the linker will be set to
  # `ld.bfd` which will try to resolve all undefined symbols at link time.
  #
  # Since we could be using a shared library version of fmt we need
  # it on the final link line of consumers
  if(fmt_is_external)
    target_link_libraries(nvbench PUBLIC fmt::fmt)
  endif()
endif()

target_compile_features(nvbench PUBLIC cuda_std_17 PRIVATE cxx_std_17)
add_dependencies(nvbench.all nvbench)

# nvbench.main (nvbench::main)
add_library(nvbench.main OBJECT main.cu)
nvbench_config_target(nvbench.main)
target_link_libraries(nvbench.main PUBLIC nvbench)
set_target_properties(nvbench.main PROPERTIES EXPORT_NAME main)
add_dependencies(nvbench.all nvbench.main)

# Support add_subdirectory:
add_library(nvbench::nvbench ALIAS nvbench)
add_library(nvbench::main ALIAS nvbench.main)

nvbench_setup_dep_dlls(nvbench)
nvbench_install_libraries(nvbench nvbench.main nvbench.build_interface)

# nvcc emits several unavoidable warnings while compiling nlohmann_json:
if (json_is_cu)
  set_property(SOURCE ${json_printer_impl} APPEND PROPERTY COMPILE_OPTIONS
    # error #186-D: pointless comparison of unsigned integer with zero
    $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:-Xcudafe=--diag_suppress=186>
    # error #940-D: missing return statement at end of non-void function
    # (the end of the function in hash.hpp(114) is unreachable)
    $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:-Xcudafe=--diag_suppress=940>
  )
endif()

# The call to `rapids_cmake_write_git_revision_file` must be in the same
# CMakeLists.txt as the consumer ( nvbench ) for CMake to get the dependency
# graph correct.
rapids_cmake_write_git_revision_file(
  nvbench_git_revision
  "${NVBench_BINARY_DIR}/nvbench/detail/git_revision.cuh"
)
target_link_libraries(nvbench PRIVATE nvbench_git_revision)

if(NOT BUILD_SHARED_LIBS)
  # Need to ensure that for static builds we export the nvbench_git_revision
  # target
  nvbench_install_libraries(nvbench_git_revision)
endif()
