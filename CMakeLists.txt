cmake_minimum_required(VERSION 3.16)
project(compsim_ss2022)

add_definitions("-std=c++11")
add_definitions("-O3")
add_definitions("-g")  # gcc adds debug information (confer `man gcc`)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

find_package(Qt6 REQUIRED COMPONENTS Widgets)
find_package(Qt6 REQUIRED COMPONENTS Charts)

add_subdirectory(u3/)
add_subdirectory(u4/)
add_subdirectory(u5/)
