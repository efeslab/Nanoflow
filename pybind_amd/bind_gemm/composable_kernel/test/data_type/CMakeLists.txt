# temporarily disable flaky test for all architectures
add_definitions(-DCK_SKIP_FLAKY_F8_TEST)
set(CK_SKIP_FLAKY_F8_TEST "ON")

if (USE_BITINT_EXTENSION_INT4)
  add_gtest_executable(test_int4 test_int4.cpp)
  if(result EQUAL 0)
    target_link_libraries(test_int4 PRIVATE utility)
  endif()
endif()



add_custom_target(test_fp8)

if (CK_USE_OCP_FP8)
  add_gtest_executable(test_fp8_ocp test_fp8_ocp.cpp)
  if(result EQUAL 0)
    target_link_libraries(test_fp8_ocp PRIVATE utility)
  endif()

  add_gtest_executable(test_bf8_ocp test_bf8_ocp.cpp)
  if(result EQUAL 0)
    target_link_libraries(test_bf8_ocp PRIVATE utility)
  endif()

  add_dependencies(test_fp8 test_fp8_ocp)
  add_dependencies(test_fp8 test_bf8_ocp)
endif()

if (CK_USE_FNUZ_FP8)
  add_gtest_executable(test_fp8_fnuz test_fp8_fnuz.cpp)
  if(result EQUAL 0)
    target_link_libraries(test_fp8_fnuz PRIVATE utility)
  endif()

  add_gtest_executable(test_bf8_fnuz test_bf8_fnuz.cpp)
  if(result EQUAL 0)
    target_link_libraries(test_bf8_fnuz PRIVATE utility)
  endif()

  add_dependencies(test_fp8 test_fp8_fnuz)
  add_dependencies(test_fp8 test_bf8_fnuz)
endif()

add_gtest_executable(test_custom_type test_custom_type.cpp)
if(result EQUAL 0)
  target_link_libraries(test_custom_type PRIVATE utility)
endif()

add_gtest_executable(test_type_convert_const type_convert_const.cpp)
add_gtest_executable(test_bhalf test_bhalf.cpp)
