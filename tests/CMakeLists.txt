add_executable(single-cpu-tests
               array-test.cpp
               bricked-array-test.cpp)
target_link_libraries(single-cpu-tests gtest gtest_main brick ${LIBRT})
add_test(NAME multiarray_test COMMAND array-test.cpp)
add_test(NAME bricked_array_test COMMAND bricked-array-test.cpp)

if(MPI_FOUND)
    add_executable(mpi-cpu-tests mpi-layout-test.cpp)
    target_link_libraries(mpi-cpu-tests gtest gtest_main brick brick-mpi ${MPI_CXX_LIBRARIES} ${LIBRT})
    add_test(NAME mpi_layout COMMAND mpi-layout-test.cpp)
endif()

if(CMAKE_CUDA_COMPILER)
    include_directories(
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )
    add_executable(cuda-tests
                   copy-to-cuda-test.cu)
    target_link_libraries(cuda-tests gtest gtest_main brick ${LIBRT})
    add_test(NAME array_to_cuda_test COMMAND copy-to-cuda-test.cu)
endif()