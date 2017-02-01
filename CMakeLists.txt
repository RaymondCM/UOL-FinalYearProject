cmake_minimum_required(VERSION 2.8.9)
project(ProjectY3)

#Enable Target Debugging
include(CMakeToolsHelpers OPTIONAL)

#DCMTK package has a config file so load it (osconfig.h).
add_definitions(-DHAVE_CONFIG_H)

#Add root to be used in the source code
add_definitions(-DSOURCE_CODE_LOCATION="${CMAKE_SOURCE_DIR}")

#Find both packages
find_package(DCMTK REQUIRED)
find_package(OpenCV REQUIRED)

#Defne CPP flags
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    message(STATUS "MSVC detected, adding compile flags")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /Wp64")
else()
    message(STATUS "GCC|Clang detected, adding compile flags")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wextra -Wpedantic -Wconversion -Wfloat-equal -pthread")
endif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")

#Include DCMTK headers and project headers
include_directories(${DCMTK_INCLUDE_DIRS})
include_directories(include)

#Glob all source files
file(GLOB SOURCES "src/*.cpp")
add_executable(ProjectY3 ${SOURCES})

#Link library files
target_link_libraries(ProjectY3 ${DCMTK_LIBRARIES} )
target_link_libraries(ProjectY3 ${OpenCV_LIBS} )