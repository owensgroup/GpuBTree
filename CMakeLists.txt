cmake_minimum_required(VERSION 3.19 FATAL_ERROR)
if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
  cmake_policy(SET CMP0104 OLD)
endif()

project(GpuBTree LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

#default build type is Release
if ("${CMAKE_BUILD_TYPE}" STREQUAL "")
    set(CMAKE_BUILD_TYPE Release)
endif ()

# Auto-detect GPU architecture, sets ${CUDA_ARCHS}
include("cmake/AutoDetectCudaArch.cmake")
include("cmake/FindGoogleTest.cmake")

# Direct all output to /bin directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)

# Enable IDE folder support
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Hash tables lib
add_library(gpu_btree INTERFACE)
target_include_directories( gpu_btree
                            INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/src"
                            INTERFACE "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
target_compile_features(gpu_btree INTERFACE cxx_std_17 cuda_std_17)
set(CMAKE_CXX_EXTENSIONS OFF)

# CUDA and C++ compiler flags
set(CXX_FLAGS
 -m64
 -Wall
 -std=c++17
 $<$<STREQUAL:${MSVC},1>:-D_SCL_SECURE_NO_WARNINGS>)
set(CUDA_FLAGS
    -Xcompiler=-Wall
    ${CUDA_ARCHS}
	-lineinfo
	--expt-extended-lambda
    #-Xptxas -warn-spills -res-usage
	$<$<STREQUAL:${MSVC},0>:-O3>)
    #-Xcompiler)

add_library(developer_flags INTERFACE)
target_compile_options(developer_flags INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:${CXX_FLAGS}>
    $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_FLAGS}>)

target_link_libraries(gpu_btree
    INTERFACE $<BUILD_INTERFACE:developer_flags>
    INTERFACE -lstdc++fs)
## Fix includes in visual studio for intellisense
set(CMAKE_VS_SDK_INCLUDE_DIRECTORIES
    $(VC_IncludePath)
    $(WindowsSDK_IncludePath)
    "$(ProjectDir)..\\..\\src")

add_subdirectory(src)
add_subdirectory(test)
add_subdirectory(unittests)