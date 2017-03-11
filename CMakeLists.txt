#Root CMake file. Declared shared dynamic libs here
cmake_minimum_required(VERSION 3.6.0)

#Use C++11
set (CMAKE_CXX_STANDARD 11)

#Enable Target Debugging 
include(CMakeToolsHelpers OPTIONAL)

#Add root to be used in the source code
add_definitions(-DPROJECT_ROOT="${CMAKE_SOURCE_DIR}")  

####################################################################
#--------------------Find Required Packages------------------------#
####################################################################

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

#Find OpenCL dependancy for Parallel
find_package(OpenCL REQUIRED)
MESSAGE( STATUS "OpenCL_INCLUDE_DIR: " ${OpenCL_INCLUDE_DIR})
MESSAGE( STATUS "OpenCL_LIBRARIES: " ${OpenCL_LIBRARIES})

####################################################################
#----------------------------Shared--------------------------------#
####################################################################
include_directories(headers)

####################################################################
#--------------------------Sequential------------------------------#
####################################################################

#Add all source files
add_executable(Sequential 
	"src/sequential.cpp")

#Include target specific include directories
target_include_directories(Sequential PUBLIC ${OpenCV_INCLUDE_DIR})
target_include_directories(Sequential PUBLIC ${DCMTK_INCLUDE_DIRS})

#Link library files
target_link_libraries(Sequential ${DCMTK_LIBRARIES})
target_link_libraries(Sequential ${OpenCV_LIBS})

####################################################################
#---------------------------Parallel-------------------------------#
####################################################################

#Add all source files
add_executable(Parallel 
	"src/parallel.cpp")

#Include target specific include directories
target_include_directories(Parallel PUBLIC ${OpenCV_INCLUDE_DIR})
target_include_directories(Parallel PUBLIC ${OpenCL_INCLUDE_DIR})
target_include_directories(Parallel PUBLIC ${DCMTK_INCLUDE_DIRS})
        
#Link library files
target_link_libraries(Parallel ${DCMTK_LIBRARIES})
target_link_libraries(Parallel ${OpenCV_LIBS})
target_link_libraries(Parallel ${OpenCL_LIBRARIES})