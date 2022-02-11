if(gtensor_FOUND AND CMAKE_CUDA_COMPILER)
    # capture brick-dim if passed
    function(set_gene6d_brick_dim target_name)
        if(GENE6D_BRICK_DIM)
            # Commas need to be escaped for nvcc preprocessor
            # https://gitlab.kitware.com/cmake/cmake/-/issues/17379
            string(REPLACE , [[\,]] GENE6D_BRICK_DIM_ESCAPED_COMMA "${GENE6D_BRICK_DIM}" )
            target_compile_definitions(${target_name}
                    PUBLIC $<$<COMPILE_LANGUAGE:CXX>:GENE6D_BRICK_DIM=${GENE6D_BRICK_DIM}>
                           $<$<COMPILE_LANGUAGE:CUDA>:GENE6D_BRICK_DIM=${GENE6D_BRICK_DIM_ESCAPED_COMMA}>)
        endif()
    endfunction()

    # Generate kernels
    set(GENE6D_STENCILS_INCLUDE_DIRS
            "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
    if(GENE6D_BRICK_DIM)
        set(GENE6D_STENCILS_COMPILE_OPTIONS "-DGENE6D_BRICK_DIM=${GENE6D_BRICK_DIM}")
    endif()
    VSTARGET(GENE6D_STENCILS brick-stencils.cu brick-stencils-out.cu)

    # mpi computation
    if(MPI_FOUND)
        add_executable(mpi-gene6d mpi-main.cpp ${VSTARGET_GENE6D_STENCILS_OUTPUT} mpi-util.cpp util.cpp)
        target_gtensor_sources(mpi-gene6d PRIVATE mpi-main.cpp gtensor-stencils.h)
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
        set_gene6d_brick_dim(mpi-gene6d)
    endif()

    # single-device computation
    add_executable(single-gene-6d single-main.cpp ${VSTARGET_GENE6D_STENCILS_OUTPUT} single-util.cpp util.cpp)
    target_gtensor_sources(single-gene-6d PRIVATE single-main.cpp gtensor-stencils.h)
    target_link_libraries(single-gene-6d
            brick brickhelper
            gtensor::gtensor
            ${LIBRT}
            CUDA::cudart)

    set_target_properties(single-gene-6d
            PROPERTIES OUTPUT_NAME single-gene6d)
    target_include_directories(single-gene6d PUBLIC
            "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
            )
    set_gene6d_brick_dim(single-gene-6d)

    # FFT computation
    add_executable(single-fft-gene-6d ../gene/fft-gene-6d.cu)
    if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.17")
        set_property(TARGET single-fft-gene-6d PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    endif()
    if(PERLMUTTER)
        if(NOT DEFINED ENV{CUFFT_LINK_DIR} AND NOT DEFINED CUFFT_LINK_DIR)
            message(FATAL_ERROR "Must pass CUFFT_LINK_DIR option (or set as environment variable) for Perlmutter build")
        endif()
        set(CUFFT_LINK_DIR $ENV{CUFFT_LINK_DIR} CACHE PATH "Directory containing libcufft_static.a" )
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -L${CUFFT_LINK_DIR}")
    endif()
    target_link_libraries(single-fft-gene-6d brickhelper ${CUFFT_LIBRARIES} gtensor::gtensor CUDA::nvToolsExt)
    target_include_directories(single-fft-gene-6d PUBLIC
            "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
            )
    set_target_properties(single-fft-gene-6d PROPERTIES OUTPUT_NAME fft-gene-6d)
    # https://developer.nvidia.com/blog/cuda-pro-tip-use-cufft-callbacks-custom-data-processing/
    # need relocatable device code
    target_compile_options(single-fft-gene-6d PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--device-c>)
    # link-time optimization https://developer.nvidia.com/blog/improving-gpu-app-performance-with-cuda-11-2-device-lto/
    if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL "11.2" AND ${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.18")
        message(STATUS "Enabling device link-time optimization of single-fft-gene-6d for cuda")
        target_link_options(single-fft-gene-6d PRIVATE -dlto)
    else()
        message(STATUS "No device link-time optimization for cuda with cuda version ${CMAKE_CUDA_COMPILER_VERSION} and cmake version ${CMAKE_VERSION}")
    endif()
endif ()
