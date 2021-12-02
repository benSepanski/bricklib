VSTARGET(N3MPI main.cpp main-out.cpp)
add_executable(weak-cpu ${VSTARGET_N3MPI_OUTPUT} args.cpp)
target_link_libraries(weak-cpu brick brick-mpi brickhelper ${MPI_CXX_LIBRARIES} ${LIBRT})
set_target_properties(weak-cpu
        PROPERTIES OUTPUT_NAME cpu)

VSTARGET(N4MPI main-4d.cpp main-4d-out.cpp)
add_executable(weak-cpu-4d ${VSTARGET_N4MPI_OUTPUT} args.cpp)
target_link_libraries(weak-cpu-4d brick brick-mpi brickhelper ${MPI_CXX_LIBRARIES} ${LIBRT})
set_target_properties(weak-cpu-4d
        PROPERTIES OUTPUT_NAME cpu-4d)

VSTARGET(N3MPI_ONESIDED onesided.cpp onesided-out.cpp)
add_executable(weak-onesided ${VSTARGET_N3MPI_ONESIDED_OUTPUT})
target_link_libraries(weak-onesided brick brick-mpi brickhelper ${MPI_CXX_LIBRARIES} ${LIBRT})
set_target_properties(weak-onesided
        PROPERTIES OUTPUT_NAME onesided)

VSTARGET(N3MPI_SHMEM shmem.cpp shmem-out.cpp)
add_executable(weak-shmem EXCLUDE_FROM_ALL ${VSTARGET_N3MPI_SHMEM_OUTPUT})
message(STATUS ${MPI_CXX_LIBRARIES})
target_link_libraries(weak-shmem brick brick-mpi brickhelper ${MPI_CXX_LIBRARIES} /usr/lib/openmpi/liboshmem.so ${LIBRT})
set_target_properties(weak-shmem
        PROPERTIES OUTPUT_NAME shmem)

if(CMAKE_CUDA_COMPILER)
    include_directories(
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )
    VSTARGET(C3MPI main.cu main-out.cu)
    add_executable(weak-cuda ${VSTARGET_C3MPI_OUTPUT} args.cpp)
    target_compile_definitions(weak-cuda PRIVATE)
    target_link_libraries(weak-cuda brick brick-mpi brickhelper ${MPI_CXX_LIBRARIES} ${LIBRT})
    set_target_properties(weak-cuda
            PROPERTIES OUTPUT_NAME cuda)

    VSTARGET(C3MPIMMAP mmap.cu mmap-out.cu)
    add_executable(weak-cuda-mmap ${VSTARGET_C3MPIMMAP_OUTPUT} args.cpp)
    target_link_libraries(weak-cuda-mmap brick brick-mpi brickhelper ${MPI_CXX_LIBRARIES} ${LIBRT} cuda)
    set_target_properties(weak-cuda-mmap
            PROPERTIES OUTPUT_NAME cuda-mmap)

    if(gtensor_FOUND)
        OPTION(GENE6D_UNALIGN "Use DECOMP_PAGEUNALIGN for weak-gene6d" OFF)
        OPTION(GENE6D_CUDA_AWARE "Use CUDA_AWARE for weak-gene6d" OFF)
        OPTION(GENE6D_USE_TYPES "Use MPI Types for array transfers" OFF)

        add_executable(weak-gene6d gene-6d-main.cu)
        target_gtensor_sources(weak-gene6d PRIVATE gene-6d-main.cu)
        target_link_libraries(weak-gene6d brick brickhelper gtensor::gtensor ${MPI_CXX_LIBRARIES} ${LIBRT} brick-mpi CUDA::cudart)

        set_target_properties(weak-gene6d 
                PROPERTIES OUTPUT_NAME gene6d)
        target_include_directories(weak-gene6d PUBLIC
                "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
                )
        if(GENE6D_UNALIGN)
            target_compile_definitions(weak-gene6d PUBLIC DECOMP_PAGEUNALIGN)
        endif()
        if(GENE6D_CUDA_AWARE)
            target_compile_definitions(weak-gene6d PUBLIC CUDA_AWARE)
        endif()
        if(GENE6D_USE_TYPES)
            target_compile_definitions(weak-gene6d PUBLIC USE_TYPES)
        endif()
    endif ()
endif ()
