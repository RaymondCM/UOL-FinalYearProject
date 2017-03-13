cmake_minimum_required(VERSION 3.6.0)

#Set enviroment variable for target files location
add_definitions(-DPROJECT_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")

#Find OpenCL dependancy for Parallel
find_package(OpenCL REQUIRED)

#Add all source files
add_executable(Par_Segmentation "src/main.cpp")

#Include target specific include directories
target_include_directories(Par_Segmentation PUBLIC include)
target_include_directories(Par_Segmentation PUBLIC ${SEG_SHARED_LIBS})
target_include_directories(Par_Segmentation PUBLIC ${SHARED_LIBS})
target_include_directories(Par_Segmentation PUBLIC ${OpenCL_INCLUDE_DIR})

#Link library files
target_link_libraries(Par_Segmentation ${OpenCV_LIBS} )
target_link_libraries(Par_Segmentation ${OpenCL_LIBRARIES} )