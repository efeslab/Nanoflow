# Test that we're converging to an accurate mean + stdev without timing out:
set(test_name nvbench.test.device.noisy_bench)
add_executable(${test_name} noisy_bench.cu)
target_link_libraries(${test_name} PRIVATE nvbench::main fmt)
nvbench_config_target(${test_name})
add_dependencies(nvbench.test.all ${test_name})

if (NVBench_ENABLE_DEVICE_TESTING)
  add_test(NAME ${test_name} COMMAND "$<TARGET_FILE:${test_name}>")
  set_tests_properties(${test_name} PROPERTIES
    # Any timeouts/warnings are hard failures for this test.
    FAIL_REGULAR_EXPRESSION "Warn;timed out"
  )
endif()
