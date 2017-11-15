cmake_minimum_required(VERSION 3.6.0)
project(UOL-FinalYearProject)

set (CMAKE_CXX_STANDARD 11)

#########################################################################################
#-------------------------------Find Required Packages----------------------------------#
#########################################################################################

#Add root to base dir for relative data path access
add_definitions(-DROOT_DIR="${CMAKE_SOURCE_DIR}")  

#Load OSConfig.h from DCMTK packag
#TODO: Add dependancy
add_definitions(-DHAVE_CONFIG_H)
find_package(DCMTK)

if (WIN32)
	SET(OpenCV_DIR C:\\lib\\Install\\opencv\\x64\\vc15\\lib)
	
endif()

find_package(OpenCV REQUIRED)

#Set Variables to shared project files
set(SHARED_LIBS "${CMAKE_CURRENT_SOURCE_DIR}/src/Shared/include") 
set(SHARED_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/Shared/src")

#########################################################################################
#--------------------------Build Block Matching Projects--------------------------------#
#########################################################################################

#Add Project Subdirectories
add_subdirectory(src/Sequential)
add_subdirectory(src/Parallel)