add_executable(nvbench.ctl nvbench-ctl.cu)
nvbench_config_target(nvbench.ctl)
target_link_libraries(nvbench.ctl PRIVATE nvbench)
set_target_properties(nvbench.ctl PROPERTIES
  OUTPUT_NAME nvbench-ctl
  EXPORT_NAME ctl
)
add_dependencies(nvbench.all nvbench.ctl)
nvbench_setup_dep_dlls(nvbench.ctl)
nvbench_install_executables(nvbench.ctl)

if (NVBench_ENABLE_TESTING)
  # Test: nvbench
  add_test(NAME nvbench.ctl.no_args COMMAND "$<TARGET_FILE:nvbench.ctl>")
  # Should print the version without any args:
  set_property(TEST nvbench.ctl.no_args
    PROPERTY PASS_REGULAR_EXPRESSION "NVBench v"
  )

  # Test: nvbench --version
  add_test(NAME nvbench.ctl.version COMMAND "$<TARGET_FILE:nvbench.ctl>" --version)
  # Should print the version without any args:
  set_property(TEST nvbench.ctl.version
    PROPERTY PASS_REGULAR_EXPRESSION "NVBench v"
  )

  # Test: nvbench --list
  add_test(NAME nvbench.ctl.list COMMAND "$<TARGET_FILE:nvbench.ctl>" --list)

  # Test: nvbench -l
  add_test(NAME nvbench.ctl.l COMMAND "$<TARGET_FILE:nvbench.ctl>" -l)

  # Test: nvbench --help
  add_test(NAME nvbench.ctl.help COMMAND "$<TARGET_FILE:nvbench.ctl>" --help)

  # Test: nvbench -h
  add_test(NAME nvbench.ctl.h COMMAND "$<TARGET_FILE:nvbench.ctl>" -h)

  # Test: nvbench --help-axes
  add_test(NAME nvbench.ctl.help_axes COMMAND "$<TARGET_FILE:nvbench.ctl>" --help-axes)

  # Test: nvbench --help-axis
  add_test(NAME nvbench.ctl.help_axis COMMAND "$<TARGET_FILE:nvbench.ctl>" --help-axis)
endif()
