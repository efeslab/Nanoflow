cmake_minimum_required(VERSION 3.20.1)
project(NVBenchTestExport CUDA CXX)

message(STATUS "NVBench_DIR=${NVBench_DIR}")
find_package(NVBench REQUIRED)

add_executable(test_bench test_bench.cu)
target_link_libraries(test_bench PRIVATE nvbench::main)
enable_testing()
add_test(NAME test_bench COMMAND "$<TARGET_FILE:test_bench>" --timeout 1)
add_test(NAME nvbench_ctl COMMAND "$<TARGET_FILE:nvbench::ctl>")

# Setup LD_LIBRARY_PATH for testing
if (UNIX)
  set(ctl_lib_path "")
  set(cupti_lib_path "")

  # Need to find installed libnvbench.so for installed nvbench-ctl.
  # Not needed for build_tree test because of RUNPATH.
  if (TEST_TYPE STREQUAL "INSTALL_TREE")
    get_property(nvbench_config TARGET nvbench::nvbench
      PROPERTY IMPORTED_CONFIGURATIONS
    )

    list(LENGTH nvbench_config num_configs)
    if (num_configs GREATER 1)
      message(WARNING
        "Multiple IMPORTED_CONFIGURATIONS for nvbench::nvbench. "
        "Picking the first one. This may cause issues."
      )
      list(GET nvbench_config 0 nvbench_config)
    endif()

    get_property(ctl_lib_path TARGET nvbench::nvbench
      PROPERTY IMPORTED_LOCATION_${nvbench_config}
    )
    cmake_path(GET ctl_lib_path PARENT_PATH ctl_lib_path)
  endif()

  # Need to add the CUPTI path to LD_LIBRARY_PATH to make sure CUPTI libraries
  # are found at runtime:
  if (TARGET nvbench::cupti)
    get_property(cupti_lib_path TARGET nvbench::cupti PROPERTY IMPORTED_LOCATION)
    cmake_path(GET cupti_lib_path PARENT_PATH cupti_lib_path)
  endif()

  set_property(TEST test_bench PROPERTY
    ENVIRONMENT "LD_LIBRARY_PATH=${cupti_lib_path}"
  )
  set_property(TEST nvbench_ctl PROPERTY
    ENVIRONMENT "LD_LIBRARY_PATH=${ctl_lib_path}:${cupti_lib_path}"
  )

endif()
