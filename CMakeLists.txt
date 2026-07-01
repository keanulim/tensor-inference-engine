cmake_minimum_required(VERSION 3.20)
project(tensor_engine)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OSX_SYSROOT /Library/Developer/CommandLineTools/SDKs/MacOSX26.sdk)

add_compile_options(-isystem /Library/Developer/CommandLineTools/SDKs/MacOSX26.sdk/usr/include/c++/v1)

enable_testing()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)

FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(googletest)

find_package(Protobuf REQUIRED)
find_package(absl REQUIRED)

add_executable(tensor_engine main.cpp kernels.cpp registry.cpp)

add_executable(tensor_tests
        tests/test_main.cpp
        kernels.cpp
        tests/test_softmax.cpp
        tests/test_layernorm.cpp
        tests/test_gemm.cpp
        tests/test_relu.cpp
        tests/test_silu.cpp
        tests/test_embedding_lookup.cpp
)

target_link_libraries(tensor_tests
        GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(tensor_tests)

add_executable(bench_all benchmarks/bench_all.cpp kernels.cpp)
target_include_directories(bench_all PRIVATE ${CMAKE_OSX_SYSROOT}/usr/include/c++/v1)

add_executable(onnx_loader
        src/onnx_loader.cpp
        third_party/onnx/onnx.pb.cc
)

target_include_directories(onnx_loader PRIVATE
        ${CMAKE_OSX_SYSROOT}/usr/include/c++/v1
        ${Protobuf_INCLUDE_DIRS}
        third_party/onnx
)

target_link_libraries(onnx_loader
        ${Protobuf_LIBRARIES}
        absl::log
        absl::log_internal_message
        absl::log_internal_check_op
)