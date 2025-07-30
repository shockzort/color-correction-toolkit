cmake_minimum_required(VERSION 3.16)
project(ColorCorrectionSystem VERSION 1.0.0 LANGUAGES CXX)

# Set C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build configuration
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo")

# Global compiler flags
if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
    # Add linker flags to help with library compatibility
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--as-needed")
    # Try to ignore version conflicts
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--unresolved-symbols=ignore-in-shared-libs")
endif()

# Find required packages
find_package(OpenCV 4.5 REQUIRED
    COMPONENTS
        core
        imgproc
        imgcodecs
        highgui
        videoio
        photo  # Denoising functions
        mcc  # ColorChecker module
)

# Optional GPU acceleration
find_package(OpenCL QUIET)
find_package(CUDA QUIET)

# Google Test for unit testing
find_package(GTest QUIET)
if(GTest_FOUND)
    enable_testing()
    set(TESTING_ENABLED ON)
else()
    message(STATUS "Google Test not found - testing disabled")
    set(TESTING_ENABLED OFF)
endif()

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(modules)

# Add modules
add_subdirectory(modules/shared)
add_subdirectory(modules/color_correction)

# Optional tools
add_subdirectory(modules/tools)

# Print configuration summary
message(STATUS "=== Color Correction System Configuration ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: C++${CMAKE_CXX_STANDARD}")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCL support: ${OpenCL_FOUND}")
message(STATUS "CUDA support: ${CUDA_FOUND}")
message(STATUS "Testing enabled: ${TESTING_ENABLED}")
message(STATUS "==============================================")