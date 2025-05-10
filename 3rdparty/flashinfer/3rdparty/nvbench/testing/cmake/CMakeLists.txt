if ("MSVC" STREQUAL "${CMAKE_CXX_COMPILER_ID}")
  # There's a bug that prevents build-and-test from working on MSVC.
  # See NVIDIA/nvbench#43.
  return()
endif()

# Need to escape the semicolons in CUDA_ARCHITECTURES or the tests break:
nvbench_escaped_cuda_arches(arches)

set(cmake_opts
  -D "CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
  -D "CMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}"
  -D "CMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
  -D "CMAKE_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}"
  -D "CMAKE_CUDA_FLAGS=${CMAKE_CUDA_FLAGS}"
  -D "CMAKE_CUDA_ARCHITECTURES=${arches}"
)

# Temporary installation prefix for tests against installed nvbench:
set(tmp_install_prefix "${CMAKE_CURRENT_BINARY_DIR}/test_nvbench_install")

# Add a build-and-test CTest.
# - full_test_name_var will be set to the full name of the test.
# - subdir is the relative path to the test project directory.
# - test_id is used to generate a unique name for this test, allowing the
#   subdir to be reused.
# - Any additional args will be passed to the project configure step.
function(nvbench_add_compile_test full_test_name_var subdir test_id)
  set(test_name nvbench.test.cmake.${subdir}.${test_id})
  set(src_dir "${CMAKE_CURRENT_SOURCE_DIR}/${subdir}")
  set(build_dir "${CMAKE_CURRENT_BINARY_DIR}/${subdir}/${test_id}")
  add_test(NAME ${test_name}
    COMMAND "${CMAKE_CTEST_COMMAND}"
      --build-and-test "${src_dir}" "${build_dir}"
      --build-generator "${CMAKE_GENERATOR}"
      --build-options
        ${cmake_opts}
        ${ARGN}
      --test-command "${CMAKE_CTEST_COMMAND}" --output-on-failure
  )
  set(${full_test_name_var} ${test_name} PARENT_SCOPE)
endfunction()

################################################################################
# Test against build tree export

nvbench_add_compile_test(test_name
  test_export
  build_tree
  -D "NVBench_DIR=${NVBench_BINARY_DIR}"
  -D TEST_TYPE=BUILD_TREE
)

################################################################################
# Test against install tree export

nvbench_add_compile_test(test_name
  test_export
  install_tree
  # "rapids_export() always installs to lib" per rapids_export docs
  -D "NVBench_DIR=${tmp_install_prefix}/lib/cmake/nvbench/"
  -D TEST_TYPE=INSTALL_TREE
)
set_tests_properties(${test_name} PROPERTIES FIXTURES_REQUIRED install_tree)

################################################################################
# Install tree fixtures
add_test(NAME nvbench.test.cmake.install_tree.install
  COMMAND "${CMAKE_COMMAND}"
    --install "${NVBench_BINARY_DIR}"
    --prefix "${tmp_install_prefix}"
)
set_tests_properties(nvbench.test.cmake.install_tree.install PROPERTIES
  FIXTURES_SETUP install_tree
)

add_test(NAME nvbench.test.cmake.install_tree.cleanup
  COMMAND "${CMAKE_COMMAND}" -E rm -rf "${tmp_install_prefix}"
)
set_tests_properties(nvbench.test.cmake.install_tree.cleanup PROPERTIES
  FIXTURES_CLEANUP install_tree
)
