cmake_minimum_required(VERSION 2.8.9)
project(cardiac_imaging)

include(CMakeToolsHelpers OPTIONAL)

find_library(imebra_library NAMES imebra libimebra HINTS ${CMAKE_BINARY_DIR} /usr/local/lib)
set(IMEBRA_HEADERS /usr/local/includes/imebra)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    message(STATUS "MSVC detected, adding compile flags")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /Wp64")
    set(IMEBRA_LIBRARIES ${imebra_library})
else()
    message(STATUS "GCC|Clang detected, adding compile flags")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wextra -Wpedantic -Wconversion -Wfloat-equal -pthread")
    set(IMEBRA_LIBRARIES pthread ${imebra_library})
endif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")

# Specify include and source files
#---------------------------------
include_directories(${IMEBRA_HEADERS})
include_directories(include)

#However, the file(GLOB...) allows for wildcard additions:
file(GLOB SOURCES "src/*.cpp")
add_executable(testCardiac ${SOURCES})
target_link_libraries(testCardiac ${IMEBRA_LIBRARIES})
