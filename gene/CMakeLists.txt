if(gtensor_FOUND AND CMAKE_CUDA_COMPILER)
    if(MPI_FOUND)
        add_executable(mpi-gene6d gene-6d-mpi.cpp gene-6d-stencils.cu mpi-util.cpp)
        target_gtensor_sources(mpi-gene6d PRIVATE gene-6d-mpi.cpp gene6d-gtensor-stencils.h)
        target_link_libraries(mpi-gene6d
                              brick brickhelper
                              gtensor::gtensor
                              ${MPI_CXX_LIBRARIES} ${LIBRT}
                              brick-mpi CUDA::cudart)

        set_target_properties(mpi-gene6d
                PROPERTIES OUTPUT_NAME mpi-gene6d)
        target_include_directories(mpi-gene6d PUBLIC
                "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
                )

        OPTION(GENE6D_UNALIGN "Use DECOMP_PAGEUNALIGN for mpi-gene6d" OFF)
        OPTION(GENE6D_CUDA_AWARE "Use CUDA_AWARE for mpi-gene6d" OFF)
        OPTION(GENE6D_USE_TYPES "Use MPI Types for array transfers" OFF)
        if(GENE6D_UNALIGN)
            target_compile_definitions(mpi-gene6d PUBLIC DECOMP_PAGEUNALIGN)
        endif()
        if(GENE6D_CUDA_AWARE)
            target_compile_definitions(mpi-gene6d PUBLIC CUDA_AWARE)
        endif()
        if(GENE6D_USE_TYPES)
            target_compile_definitions(mpi-gene6d PUBLIC USE_TYPES)
        endif()
    endif()
endif ()
