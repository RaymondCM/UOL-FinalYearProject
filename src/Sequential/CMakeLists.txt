cmake_minimum_required(VERSION 3.6.0)

#Set enviroment variable for target files location
add_definitions(-DPROJECT_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")

#Add all source files
add_executable(Seq_BlockMatching "src/main.cpp")

#Include target specific include directories
target_include_directories(Seq_BlockMatching PUBLIC include)
target_include_directories(Seq_BlockMatching PUBLIC ${SHARED_LIBS})

#Commented out temporarily to fix VS2017 IDE intelisense error
#TODO: Uncomment this line
#target_include_directories(Seq_BlockMatching PUBLIC ${DCMTK_INCLUDE_DIRS})
#target_link_libraries(Seq_BlockMatching ${DCMTK_LIBRARIES} )

#Link library files
target_link_libraries(Seq_BlockMatching ${OpenCV_LIBS} )