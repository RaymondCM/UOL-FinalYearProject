cmake_minimum_required(VERSION 2.8.9)

#Set enviroment variable for target files location
add_definitions(-DTARGET_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")

#Find OpenCL dependancy for ParallelBlockMatching
find_package(OpenCL REQUIRED)
MESSAGE( STATUS "OpenCL_INCLUDE_DIR: " ${OpenCL_INCLUDE_DIR})
MESSAGE( STATUS "OpenCL_LIBRARIES: " ${OpenCL_LIBRARIES})

#Add all source files
add_executable(ParallelBlockMatching "src/main.cpp")

#Include target specific include directories
target_include_directories(ParallelBlockMatching PUBLIC include)
target_include_directories(ParallelBlockMatching PUBLIC ${SHARED_LIBS})
target_include_directories(ParallelBlockMatching PUBLIC ${OpenCL_INCLUDE_DIR})

#Link library files
target_link_libraries(ParallelBlockMatching ${DCMTK_LIBRARIES} )
target_link_libraries(ParallelBlockMatching ${OpenCV_LIBS} )
target_link_libraries(ParallelBlockMatching ${OpenCL_LIBRARIES} )