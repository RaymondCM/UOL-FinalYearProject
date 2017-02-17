#Root CMake file. Declared shared dynamic libs here
cmake_minimum_required(VERSION 2.8.9)
project(ProjectY3)

#Enable Target Debugging
include(CMakeToolsHelpers OPTIONAL)

#DCMTK package has a config file so load it (osconfig.h).
add_definitions(-DHAVE_CONFIG_H)

#Add root to be used in the source code
add_definitions(-DSOURCE_CODE_LOCATION="${CMAKE_SOURCE_DIR}")

#Find both packages (If windows point find_package to install location)
find_package(DCMTK)
if(DCMTK_FOUND)
	include_directories(${DCMTK_INCLUDE_DIRS})
endif(DCMTK_FOUND)

if (WIN32)
	SET(CMAKE_PREFIX_PATH C:/OpenCV/build/install)
endif()

find_package(OpenCV REQUIRED)

MESSAGE( STATUS "OpenCV_DIR: " ${OpenCV_DIR})
MESSAGE( STATUS "OpenCV_LIBS: " ${OpenCV_LIBS})

#Add Project Subdirectories
add_subdirectory(Sequential)
add_subdirectory(Parallel)