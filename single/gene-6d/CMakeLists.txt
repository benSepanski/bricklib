if(gtensor_FOUND)
    add_executable(new-single-gene6d main.cpp gene-6d.cpp gene-6d.cu)
    target_gtensor_sources(new-single-gene6d PRIVATE main.cpp gene-6d.cpp)
    target_link_libraries(new-single-gene6d brick brickhelper gtensor::gtensor nvToolsExt)

    set_target_properties(new-single-gene6d
            PROPERTIES OUTPUT_NAME new-gene6d)
    target_include_directories(new-single-gene6d PUBLIC
            "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
            )
endif ()