add_executable(array-test array-test.cpp)
target_link_libraries(array-test gtest gtest_main)
add_test(NAME multiarray_test COMMAND array-test.cpp)