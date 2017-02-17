#Find OpenCL dependancy for ParallelBlockMatching
find_package(OpenCL REQUIRED)
MESSAGE( STATUS "OpenCL_INCLUDE_DIR: " ${OpenCL_INCLUDE_DIR})
MESSAGE( STATUS "OpenCL_LIBRARIES: " ${OpenCL_LIBRARIES})

#Glob all source files
add_executable(ParallelBlockMatching "src/main.cpp")

#Include target specific include directories
target_include_directories(ParallelBlockMatching PUBLIC include)
target_include_directories(ParallelBlockMatching PUBLIC ${OpenCL_INCLUDE_DIR})

#Link library files
target_link_libraries(ParallelBlockMatching ${DCMTK_LIBRARIES} )
target_link_libraries(ParallelBlockMatching ${OpenCV_LIBS} )
target_link_libraries(ParallelBlockMatching ${OpenCL_LIBRARIES} )