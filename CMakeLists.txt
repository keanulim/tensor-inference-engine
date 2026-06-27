cmake_minimum_required(VERSION 4.2)
project(tensor_engine)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_COMPILER clang++)

add_executable(tensor_engine main.cpp)
