add_executable(array-test
               array-test.cpp
               bricked-array-test.cpp)
target_link_libraries(array-test
                      gtest
                      gtest_main
                      brick
                      ${LIBRT})
add_test(NAME multiarray_test COMMAND array-test.cpp)
add_test(NAME bricked_array_test COMMAND bricked-array-test.cpp)