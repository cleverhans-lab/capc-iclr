cmake_minimum_required (VERSION 2.8.11)
project (emp-gc-test)
set(NAME "emp-gc-test")
set(ENABLE_FLOAT ON)

find_path(CMAKE_FOLDER NAMES cmake/emp-tool-config.cmake)
include(${CMAKE_FOLDER}/cmake/common.cmake)
include(${CMAKE_FOLDER}/cmake/enable_rdseed.cmake)
include(${CMAKE_FOLDER}/cmake/enable_float.cmake)

find_package(emp-sh2pc REQUIRED)
include_directories(${EMP-SH2PC_INCLUDE_DIRS})

add_executable(sum_histogram sum_histogram.cpp)
add_executable(argmax_1 argmax_1.cpp)
target_link_libraries(sum_histogram ${EMP-SH2PC_LIBRARIES})
target_link_libraries(argmax_1 ${EMP-SH2PC_LIBRARIES})

