set(test_srcs
  axes_metadata.cu
  benchmark.cu
  create.cu
  cuda_timer.cu
  cpu_timer.cu
  criterion_manager.cu
  criterion_params.cu
  custom_main_custom_args.cu
  custom_main_custom_exceptions.cu
  custom_main_global_state_raii.cu
  enum_type_list.cu
  entropy_criterion.cu
  float64_axis.cu
  int64_axis.cu
  named_values.cu
  option_parser.cu
  range.cu
  ring_buffer.cu
  runner.cu
  state.cu
  statistics.cu
  state_generator.cu
  stdrel_criterion.cu
  string_axis.cu
  type_axis.cu
  type_list.cu
)

# Custom arguments:
# CTest commands+args can't be modified after creation, so we need to rely on substitution.
set(NVBench_TEST_ARGS_nvbench.test.custom_main_custom_args "--quiet" "--my-custom-arg" "--run-once" "-d" "0")
set(NVBench_TEST_ARGS_nvbench.test.custom_main_custom_exceptions "--quiet" "--run-once" "-d" "0")

# Metatarget for all tests:
add_custom_target(nvbench.test.all)
add_dependencies(nvbench.all nvbench.test.all)

foreach(test_src IN LISTS test_srcs)
  get_filename_component(test_name "${test_src}" NAME_WLE)
  string(PREPEND test_name "nvbench.test.")
  add_executable(${test_name} "${test_src}")
  target_include_directories(${test_name} PRIVATE "${CMAKE_CURRENT_LIST_DIR}")
  target_link_libraries(${test_name} PRIVATE nvbench::nvbench fmt)
  set_target_properties(${test_name} PROPERTIES COMPILE_FEATURES cuda_std_17)
  nvbench_config_target(${test_name})
  add_test(NAME ${test_name} COMMAND "$<TARGET_FILE:${test_name}>" ${NVBench_TEST_ARGS_${test_name}})

  add_dependencies(nvbench.test.all ${test_name})
endforeach()

set_tests_properties(nvbench.test.custom_main_custom_exceptions PROPERTIES
  PASS_REGULAR_EXPRESSION "Custom error detected: Expected exception thrown."
)

add_subdirectory(cmake)
add_subdirectory(device)
