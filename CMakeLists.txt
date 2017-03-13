cmake_minimum_required(VERSION 3.6.0)
project(UOL-FinalYearProject)

set (CMAKE_CXX_STANDARD 11)

#########################################################################################
#-------------------------------Find Required Packages----------------------------------#
#########################################################################################

#Add root to base dir for relative data path access
add_definitions(-DROOT_DIR="${CMAKE_SOURCE_DIR}")  

#Load OSConfig.h from DCMTK packag
add_definitions(-DHAVE_CONFIG_H)
find_package(DCMTK REQUIRED)

if (WIN32)
	SET(OpenCV_DIR C:/OpenCV)
endif()

find_package(OpenCV REQUIRED)

#Set Variables to shared project files
set(SHARED_LIBS "${CMAKE_CURRENT_SOURCE_DIR}/targets/Shared/include")
set(SHARED_SRC "${CMAKE_CURRENT_SOURCE_DIR}/targets/Shared/src")

#########################################################################################
#--------------------------Build Block Matching Projects--------------------------------#
#########################################################################################

#Set Variables to inter-shared libraries
set(BM_SHARED_LIBS "${CMAKE_CURRENT_SOURCE_DIR}/targets/BlockMatching/Shared/include")
set(BM_SHARED_SRC "${CMAKE_CURRENT_SOURCE_DIR}/targets/BlockMatching/Shared/src")

#Add Project Subdirectories
add_subdirectory(targets/BlockMatching/Sequential)
add_subdirectory(targets/BlockMatching/Parallel)

#########################################################################################
#----------------------------Build Segmentation Projects--------------------------------#
#########################################################################################

#Set Variables to inter-shared libraries
set(SEG_SHARED_LIBS "${CMAKE_CURRENT_SOURCE_DIR}/targets/Segmentation/Shared/include")
set(SEG_SHARED_SRC "${CMAKE_CURRENT_SOURCE_DIR}/targets/Segmentation/Shared/src")

#Add Project Subdirectories
add_subdirectory(targets/Segmentation/Sequential)
add_subdirectory(targets/Segmentation/Parallel)