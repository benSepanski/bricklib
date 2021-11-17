add_executable(single-cpu-tests
               array-test.cpp
               bricked-array-test.cpp index-space-test.cpp)
target_link_libraries(single-cpu-tests gtest gtest_main brick ${LIBRT})
add_test(NAME multiarray_test COMMAND array-test.cpp)
add_test(NAME bricked_array_test COMMAND bricked-array-test.cpp)
add_test(NAME index_space_test COMMAND index-space-test.cpp)

if(MPI_FOUND)
    add_executable(mpi-cpu-tests mpi-layout-test.cpp MPIEnvironment.cpp)
    target_link_libraries(mpi-cpu-tests gtest gtest_main brick brick-mpi ${MPI_CXX_LIBRARIES} ${LIBRT})
    add_test(NAME mpi_cpu COMMAND mpi-layout-test.cpp)
endif()

if(CMAKE_CUDA_COMPILER)
    include_directories(
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )
    add_executable(cuda-tests
            cuda-test.cu)
    target_link_libraries(cuda-tests gtest gtest_main brick ${LIBRT})
    add_test(NAME cuda_test COMMAND cuda-test.cu)
endif()

if(MPI_FOUND AND CMAKE_CUDA_COMPILER)
    add_executable(mpi-cuda-tests mpi-cuda-test.cu MPIEnvironment.cpp)
    target_link_libraries(mpi-cuda-tests gtest gtest_main brick brick-mpi ${MPI_CXX_LIBRARIES} ${LIBRT})
    add_test(NAME mpi_cuda COMMAND mpi-cuda-test.cu)
endif()