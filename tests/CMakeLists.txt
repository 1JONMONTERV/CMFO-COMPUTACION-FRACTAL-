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

# New CMFO Validation Tests (Soliton, Matrix, Tensor Props)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_soliton_collision.c")
    add_executable(test_soliton test_soliton_collision.c)
    target_link_libraries(test_soliton PRIVATE cmfo_core m)
    add_test(NAME SolitonCollisionTest COMMAND test_soliton)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_mat7_stability.c")
    add_executable(test_mat7_stab test_mat7_stability.c)
    target_link_libraries(test_mat7_stab PRIVATE cmfo_core m)
    add_test(NAME Mat7StabilityTest COMMAND test_mat7_stab)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_tensor7_props.c")
    add_executable(test_tensor_props test_tensor7_props.c)
    target_link_libraries(test_tensor_props PRIVATE cmfo_core m)
    add_test(NAME TensorPropsTest COMMAND test_tensor_props)
endif()

# C++ Demo
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/demo_cpp.cpp")
    add_executable(demo_cpp demo_cpp.cpp)
    target_link_libraries(demo_cpp PRIVATE cmfo_core m)
    set_target_properties(demo_cpp PROPERTIES CXX_STANDARD 17)
    add_test(NAME CppInteropDemo COMMAND demo_cpp)
endif()
