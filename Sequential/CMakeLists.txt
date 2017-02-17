cmake_minimum_required(VERSION 2.8.9)

#Glob all source files
add_executable(SequentialBlockMatching "src/main.cpp")

#Include target specific include directories
target_include_directories(SequentialBlockMatching PUBLIC include)

#Link library files
target_link_libraries(SequentialBlockMatching ${DCMTK_LIBRARIES} )
target_link_libraries(SequentialBlockMatching ${OpenCV_LIBS} )