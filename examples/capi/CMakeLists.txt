cmake_minimum_required(VERSION 3.16)
project(rustnn_capi_example LANGUAGES C CXX)

find_package(rustnn CONFIG REQUIRED)

add_executable(rustnn_c_example main.c)
target_compile_features(rustnn_c_example PRIVATE c_std_11)
target_link_libraries(rustnn_c_example PRIVATE rustnn::rustnn)

add_executable(rustnn_cpp_example main.cpp)
target_compile_features(rustnn_cpp_example PRIVATE cxx_std_17)
target_link_libraries(rustnn_cpp_example PRIVATE rustnn::rustnn)
