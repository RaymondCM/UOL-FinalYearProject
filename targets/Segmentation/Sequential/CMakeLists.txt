cmake_minimum_required(VERSION 3.6.0)

#Set enviroment variable for target files location
add_definitions(-DPROJECT_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")

#Add all source files
add_executable(Seq_Segmentation "src/main.cpp")

#Include target specific include directories
target_include_directories(Seq_Segmentation PUBLIC include)
target_include_directories(Seq_Segmentation PUBLIC ${SEG_SHARED_LIBS})
target_include_directories(Seq_Segmentation PUBLIC ${SHARED_LIBS})

#Link library files
target_link_libraries(Seq_Segmentation ${OpenCV_LIBS} )