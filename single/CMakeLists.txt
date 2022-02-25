VSTARGET(N3AXIS ../stencils/3axis.cpp ../stencils/3axis-out.cpp)
add_executable(single-cpu cpu.cpp ${VSTARGET_N3AXIS_OUTPUT} ../stencils/copy.cpp)
target_link_libraries(single-cpu brickhelper)
set_target_properties(single-cpu
        PROPERTIES OUTPUT_NAME cpu)

add_executable(single-mpi mpi.cpp ../src/brick-mpi.cpp)
target_link_libraries(single-mpi brick brickhelper ${MPI_CXX_LIBRARIES})
set_target_properties(single-mpi
        PROPERTIES OUTPUT_NAME mpi)

if (HAS_SYCL)
    set(SYCL3AXIS_INCLUDE_DIRS
            "${SYCL_DIR}/include"
            "${SYCL_DIR}/include/sycl")
    set(SYCL3AXIS_COMPILE_OPTIONS
            "-fsycl")
    VSTARGET(SYCL3AXIS ../stencils/3axis_sycl.cpp ../stencils/3axis_sycl-out.cpp)
    add_executable(single-sycl sycl.cpp ${VSTARGET_SYCL3AXIS_OUTPUT})
    target_link_libraries(single-sycl brick brickhelper stdc++ sycl)
    target_include_directories(single-sycl
            PUBLIC "${SYCL_DIR}/include"
            PUBLIC "${SYCL_DIR}/include/sycl"
            PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/stencils")
    target_link_directories(single-sycl PUBLIC "${SYCL_DIR}/lib")
    # sycl need option for both phase
    target_compile_options(single-sycl PUBLIC "-fsycl")
    target_link_options(single-sycl PUBLIC "-fsycl")
    set_target_properties(single-sycl
            PROPERTIES OUTPUT_NAME sycl)
endif ()

if (OpenCL_FOUND)
    add_executable(single-opencl opencl.cpp ../stencils/3axis_cl.cpp)
    target_compile_definitions(single-opencl PRIVATE "CL_KRNL_OPTIONS=\"-I${PROJECT_SOURCE_DIR}/include\"")
    VSTARGET(CL3AXIS ../stencils/3axis_cl_krnl.c ${CMAKE_CURRENT_BINARY_DIR}/3axis_cl_krnl-out.c)
    add_custom_target(cl3axis
            SOURCES ${VSTARGET_CL3AXIS_OUTPUT})
    target_link_libraries(single-opencl brick brickhelper ${OpenCL_LIBRARIES})
    target_include_directories(single-opencl PUBLIC "${OpenCL_INCLUDE_DIRS}")
    add_dependencies(single-opencl cl3axis)
    set_target_properties(single-opencl
            PROPERTIES OUTPUT_NAME opencl)
endif ()

if (CMAKE_CUDA_COMPILER)
    set(C3AXIS_INCLUDE_DIRS
            "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
            )
    VSTARGET(C3AXIS ../stencils/3axis.cu ../stencils/3axis-out.cu)
    add_executable(single-cuda cuda.cpp ${VSTARGET_C3AXIS_OUTPUT})
    target_link_libraries(single-cuda brickhelper)
    set_target_properties(single-cuda
            PROPERTIES OUTPUT_NAME cuda)
    target_include_directories(single-cuda PUBLIC
            "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
            )
endif ()

if (USE_HIP)
    VSTARGET(H3AXIS ../stencils/3axis.hip.cpp ../stencils/3axis-out.hip.cpp)
    set(HIP_SRCS hip.cpp ${VSTARGET_H3AXIS_OUTPUT})
    set_source_files_properties(${HIP_SRCS} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
    hip_add_executable(single-hip ${HIP_SRCS} CLANG_OPTIONS ${CMAKE_CXX_FLAGS})
    target_link_libraries(single-hip brickhelper hip::host)
    set_target_properties(single-hip
            PROPERTIES OUTPUT_NAME hip)
endif ()

if (USE_DPC)
   VSTARGET(D3AXIS ../stencils/3axis_dpc.cpp ../stencils/3axis_dpc-out.cpp)
   set(CMAKE_CXX_COMPILER "dpcpp")
   set(CMAKE_LINKER "dpcpp")
   set(CMAKE_AR "llvm-ar")
   set(CMAKE_RANLIB "llvm-ranlib")
   set(CMAKE_CXX_STANDARD 17)
   add_executable(single-dpc dpc.cpp ${VSTARGET_D3AXIS_OUTPUT})
   target_link_libraries(single-dpc brickhelper)
   target_include_directories(single-dpc PUBLIC "$(which dpcpp)/../../include/sycl/")
   set_target_properties(single-dpc PROPERTIES OUTPUT_NAME dpc)
endif ()

if(gtensor_FOUND)
#     get_target_property(GTENSOR_INCLUDES gtensor::gtensor INTERFACE_INCLUDE_DIRECTORIES)
#     message(STATUS "Linking with gtensor from ${GTENSOR_INCLUDES}")
#     set(GENE5D_INCLUDE_DIRS "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}" "${GTENSOR_INCLUDES}")

#     VSTARGET(GENE5D ../stencils/gene-6d-main.cpp ../stencils/gene-6d-main-out.cpp)
#     add_executable(single-gene6d ${VSTARGET_GENE5D_OUTPUT} ../stencils/gene-6d.cu)
    add_executable(single-gene6d gene-6d-main.cpp ../stencils/gene-6d.cpp ../stencils/gene-6d.cu)
#     target_gtensor_sources(single-gene6d PRIVATE ${VSTARGET_GENE5D_OUTPUT})
    target_gtensor_sources(single-gene6d PRIVATE gene-6d-main.cpp ../stencils/gene-6d.cpp)
    target_link_libraries(single-gene6d brickhelper gtensor::gtensor nvToolsExt)

    set_target_properties(single-gene6d
            PROPERTIES OUTPUT_NAME gene6d)
    target_include_directories(single-gene6d PUBLIC
            "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
            )
endif ()

if(CMAKE_CUDA_COMPILER)
    # sample FFT
    add_executable(single-fft fft.cu)
    if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.17")
        set_property(TARGET single-fft PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    endif()
    if(PERLMUTTER)
        if(NOT DEFINED ENV{CUFFT_LINK_DIR} AND NOT DEFINED CUFFT_LINK_DIR)
            message(FATAL_ERROR "Must pass CUFFT_LINK_DIR option (or set as environment variable) for Perlmutter build")
        endif()
        set(CUFFT_LINK_DIR $ENV{CUFFT_LINK_DIR} CACHE PATH "Directory containing libcufft_static.a" )
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -L${CUFFT_LINK_DIR}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -L${CUFFT_LINK_DIR}")
    endif()
    target_link_libraries(single-fft brickhelper ${CUFFT_LIBRARIES})
    target_include_directories(single-fft PUBLIC
            "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
            )
    # https://developer.nvidia.com/blog/cuda-pro-tip-use-cufft-callbacks-custom-data-processing/
    # need relocatable device code
    target_compile_options(single-fft PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--device-c>)
    # link-time optimization https://developer.nvidia.com/blog/improving-gpu-app-performance-with-cuda-11-2-device-lto/
    if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL "11.2" AND ${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.18")
        message(STATUS "Enabling device link-time optimization of single-fft for cuda")
        target_link_options(single-fft PRIVATE -dlto)
    else()
        message(STATUS "No device link-time optimization for cuda with cuda version ${CMAKE_CUDA_COMPILER_VERSION} and cmake version ${CMAKE_VERSION}")
    endif()
endif()
