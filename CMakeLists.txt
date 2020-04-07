cmake_minimum_required(VERSION 3.10)

project(stitch_ndk)

set(CMAKE_CXX_STANDARD 14)

add_executable(stitch_ndk main.cpp newstitchcheck.cpp)

find_package(OpenCV REQUIRED)

target_link_libraries(stitch_ndk ${OpenCV_LIBS})