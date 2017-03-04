cmake_minimum_required(VERSION 3.6.0)

#Set enviroment variable for target files location
add_definitions(-DTARGET_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")

#Add all source files
add_executable(Sequential "src/main.cpp")

#Include target specific include directories
target_include_directories(Sequential PUBLIC include)
target_include_directories(Sequential PUBLIC ${SHARED_LIBS})
target_include_directories(Sequential PUBLIC ${DCMTK_INCLUDE_DIRS})

#Link library files
target_link_libraries(Sequential ${DCMTK_LIBRARIES} )
target_link_libraries(Sequential ${OpenCV_LIBS} )