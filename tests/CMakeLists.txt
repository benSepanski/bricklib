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

if(CMAKE_CUDA_COMPILER)
    include_directories(
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )
    add_executable(array-cuda-tests
                   copy-to-cuda-test.cu)
    target_link_libraries(array-cuda-tests
                          gtest
                          gtest_main
                          brick
                          ${LIBRT})
    add_test(NAME array_to_cuda_test COMMAND copy-to-cuda-test.cu)
endif()