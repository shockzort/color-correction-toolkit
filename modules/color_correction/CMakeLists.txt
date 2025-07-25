# Color Correction Module CMakeLists.txt

# Interface library
add_library(color_correction_interface INTERFACE)
target_include_directories(color_correction_interface INTERFACE interface)
target_link_libraries(color_correction_interface INTERFACE shared_types)

# Internal implementation library
file(GLOB_RECURSE INTERNAL_SOURCES 
    "internal/*.cpp"
    "internal/*.hpp"
)

add_library(color_correction_internal STATIC ${INTERNAL_SOURCES})
target_include_directories(color_correction_internal 
    PUBLIC internal
    PRIVATE interface
)
target_link_libraries(color_correction_internal 
    PUBLIC ${OpenCV_LIBS}
    PRIVATE shared_types shared_utils color_correction_interface
)

# Optional GPU acceleration
if(OpenCL_FOUND)
    target_link_libraries(color_correction_internal PRIVATE ${OpenCL_LIBRARIES})
    target_compile_definitions(color_correction_internal PRIVATE OPENCL_ENABLED)
endif()

if(CUDA_FOUND)
    enable_language(CUDA)
    target_link_libraries(color_correction_internal PRIVATE ${CUDA_LIBRARIES})
    target_compile_definitions(color_correction_internal PRIVATE CUDA_ENABLED)
endif()

# Main color correction library
add_library(color_correction STATIC)
target_link_libraries(color_correction 
    PUBLIC color_correction_interface
    PRIVATE color_correction_internal
)

# Testing
if(TESTING_ENABLED)
    add_subdirectory(tests)
endif()

# Tools and applications
add_subdirectory(apps)