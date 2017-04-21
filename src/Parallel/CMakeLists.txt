cmake_minimum_required(VERSION 3.6.0)

#Set enviroment variable for target files location
add_definitions(-DPROJECT_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")

#Find OpenCL dependancy for Parallel
find_package(OpenCL REQUIRED)

#Add all source files
add_executable(Par_BlockMatching "src/main.cpp")

#Include target specific include directories
target_include_directories(Par_BlockMatching PUBLIC include)
target_include_directories(Par_BlockMatching PUBLIC ${SHARED_LIBS})
target_include_directories(Par_BlockMatching PUBLIC ${OpenCL_INCLUDE_DIR})
        
#Commented out temporarily to fix VS2017 IDE intelisense error
#TODO: Uncomment this line
#target_include_directories(Par_BlockMatching PUBLIC ${DCMTK_INCLUDE_DIRS})
#target_link_libraries(Par_BlockMatching ${DCMTK_LIBRARIES} )

#Link library files
target_link_libraries(Par_BlockMatching ${OpenCV_LIBS} )
target_link_libraries(Par_BlockMatching ${OpenCL_LIBRARIES} )