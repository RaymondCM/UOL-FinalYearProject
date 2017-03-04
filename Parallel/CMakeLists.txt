cmake_minimum_required(VERSION 3.6.0)

#Set enviroment variable for target files location
add_definitions(-DTARGET_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")

#Find OpenCL dependancy for Parallel
find_package(OpenCL REQUIRED)
MESSAGE( STATUS "OpenCL_INCLUDE_DIR: " ${OpenCL_INCLUDE_DIR})
MESSAGE( STATUS "OpenCL_LIBRARIES: " ${OpenCL_LIBRARIES})

#Add all source files
add_executable(Parallel "src/main.cpp")

#Include target specific include directories
target_include_directories(Parallel PUBLIC include)
target_include_directories(Parallel PUBLIC ${SHARED_LIBS})
target_include_directories(Parallel PUBLIC ${OpenCL_INCLUDE_DIR})

#Link library files
target_link_libraries(Parallel ${DCMTK_LIBRARIES} )
target_link_libraries(Parallel ${OpenCV_LIBS} )
target_link_libraries(Parallel ${OpenCL_LIBRARIES} )