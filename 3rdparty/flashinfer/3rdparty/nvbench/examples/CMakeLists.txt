set(example_srcs
  auto_throughput.cu
  axes.cu
  custom_criterion.cu
  enums.cu
  exec_tag_sync.cu
  exec_tag_timer.cu
  skip.cu
  stream.cu
  throughput.cu
)

# Metatarget for all examples:
add_custom_target(nvbench.example.all)
add_dependencies(nvbench.all nvbench.example.all)


function (nvbench_add_examples_target target_prefix cuda_std)
  add_custom_target(${target_prefix}.all)
  add_dependencies(nvbench.example.all ${target_prefix}.all)

  foreach(example_src IN LISTS example_srcs)
    get_filename_component(example_name "${example_src}" NAME_WLE)
    string(PREPEND example_name "${target_prefix}.")
    add_executable(${example_name} "${example_src}")
    nvbench_config_target(${example_name})
    target_include_directories(${example_name} PRIVATE "${CMAKE_CURRENT_LIST_DIR}")
    target_link_libraries(${example_name} PRIVATE nvbench::main)
    set_target_properties(${example_name} PROPERTIES COMPILE_FEATURES cuda_std_${cuda_std})
    add_test(NAME ${example_name}
      COMMAND "$<TARGET_FILE:${example_name}>" --timeout 0.1 --min-time 1e-5
    )

    # These should not deadlock. If they do, it may be that the CUDA context was created before
    # setting CUDA_MODULE_LOAD=EAGER in main, see NVIDIA/nvbench#136.
    set_tests_properties(${example_name} PROPERTIES
      FAIL_REGULAR_EXPRESSION "Possible Deadlock Detected"
    )

    add_dependencies(${target_prefix}.all ${example_name})
  endforeach()

  # Silence some warnings from old thrust headers:
  set(thrust_examples
    auto_throughput
    axes
    custom_criterion
    exec_tag_sync
    exec_tag_timer
    skip
    stream
    throughput
  )
  foreach (example IN LISTS thrust_examples)
    if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
      # C4324: structure was padded due to alignment specifier
      nvbench_add_cxx_flag(${target_prefix}.${example} PRIVATE "/wd4324")

      # warning C4201: nonstandard extension used: nameless struct/union:
      # Fixed in Thrust 1.12.0 (CTK 11.4, NV HPC 21.3)
      if (${CUDAToolkit_VERSION} VERSION_LESS 11.4)
        nvbench_add_cxx_flag(${target_prefix}.${example} PRIVATE "/wd4201")
      endif()
    endif()
  endforeach()
endfunction()


foreach (std IN LISTS NVBench_DETECTED_CUDA_STANDARDS)
  nvbench_add_examples_target(nvbench.example.cpp${std} ${std})
endforeach()
