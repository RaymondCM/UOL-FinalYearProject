cmake_minimum_required(VERSION 2.8.9)

#Set enviroment variable for target files location
add_definitions(-DTARGET_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")

#Add all source files
add_executable(SequentialBlockMatching "src/main.cpp" "src/BlockMatching.cpp")

#Include target specific include directories
target_include_directories(SequentialBlockMatching PUBLIC include)
target_include_directories(SequentialBlockMatching PUBLIC ${SHARED_LIBS})

#Link library files
target_link_libraries(SequentialBlockMatching ${DCMTK_LIBRARIES} )
target_link_libraries(SequentialBlockMatching ${OpenCV_LIBS} )