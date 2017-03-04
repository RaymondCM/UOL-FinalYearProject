#Root CMake file. Declared shared dynamic libs here
cmake_minimum_required(VERSION 3.6.0)
project(ProjectY3)

#Use C++11
set (CMAKE_CXX_STANDARD 11)

#Enable Target Debugging 
include(CMakeToolsHelpers OPTIONAL)

#Add root to be used in the source code
add_definitions(-DPROJECT_ROOT="${CMAKE_SOURCE_DIR}")  

#DCMTK package has a config file so load it (osconfig.h).
add_definitions(-DHAVE_CONFIG_H)

#Find both packages (If windows point find_package to install location)
find_package(DCMTK REQUIRED)

if(DCMTK_FOUND)
	MESSAGE( STATUS "DCMTK_DIRS: " ${DCMTK_DIR})
	MESSAGE( STATUS "DCMTK_INCLUDE_DIRS: " ${DCMTK_INCLUDE_DIRS})
	MESSAGE( STATUS "DCMTK_LIBS: " ${DCMTK_LIBRARIES})
endif(DCMTK_FOUND)

if (WIN32)
	SET(OpenCV_DIR C:/OpenCV/build/install)
endif()

find_package(OpenCV REQUIRED)

#Remove these (Used for cross platform compilation debugging)
MESSAGE( STATUS "OpenCV_DIR: " ${OpenCV_DIR})
MESSAGE( STATUS "OpenCV_LIBS: " ${OpenCV_LIBS})

#Set Variables to shared libraries
set(SHARED_LIBS "${CMAKE_CURRENT_SOURCE_DIR}/shared/include")
set(SHARED_SRC "${CMAKE_CURRENT_SOURCE_DIR}/shared/src")

#Add Project Subdirectories
add_subdirectory(Sequential)
add_subdirectory(Parallel)