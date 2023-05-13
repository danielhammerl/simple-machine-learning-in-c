cmake_minimum_required(VERSION 3.23)
project(SimpleMachineLearningInC VERSION 1.0)

include(FetchContent)
set(BUILD_SHARED_LIBS OFF)
include(CheckCCompilerFlag)
set(CMAKE_APPLE_SILICON_PROCESSOR arm64)
set(CMAKE_REQUIRED_LINK_OPTIONS "-arch;arm64")
check_c_compiler_flag("-arch arm64" arm64Supported)
set(CMAKE_CXX_STANDARD 20)

IF (${CMAKE_SYSTEM_NAME} MATCHES "Darwin" AND arm64Supported MATCHES 1)
    set(CMAKE_OSX_ARCHITECTURES arm64)
ENDIF ()

set(CMAKE_CXX_STANDARD_REQUIRED true)


add_executable(SimpleMachineLearningInC main.cpp )
