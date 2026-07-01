cmake_minimum_required(VERSION 3.20)
project(tensor_engine)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OSX_SYSROOT /Library/Developer/CommandLineTools/SDKs/MacOSX26.sdk)

add_executable(tensor_engine main.cpp kernels.cpp registry.cpp)
target_include_directories(tensor_engine PRIVATE ${CMAKE_OSX_SYSROOT}/usr/include/c++/v1)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)