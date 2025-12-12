add_executable(test_core tests/test_core.c)
target_link_libraries(test_core PRIVATE cmfo_core)

add_executable(test_mat7 tests/test_mat7.c)
target_link_libraries(test_mat7 PRIVATE cmfo_core)

add_executable(test_mat7_mul tests/test_mat7_mul.c)
target_link_libraries(test_mat7_mul PRIVATE cmfo_core)

add_executable(test_mat7_det tests/test_mat7_det.c)
target_link_libraries(test_mat7_det PRIVATE cmfo_core)

add_executable(test_mat7_inv tests/test_mat7_inv.c)
target_link_libraries(test_mat7_inv PRIVATE cmfo_core)

add_executable(test_tensor7 tests/test_tensor7.c)
target_link_libraries(test_tensor7 PRIVATE cmfo_core)

add_test(NAME core COMMAND test_core)
add_test(NAME mat7 COMMAND test_mat7)
add_test(NAME mat7_mul COMMAND test_mat7_mul)
add_test(NAME mat7_det COMMAND test_mat7_det)
add_test(NAME mat7_inv COMMAND test_mat7_inv)
add_test(NAME tensor7 COMMAND test_tensor7)
