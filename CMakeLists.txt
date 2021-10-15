project(bolt)

set(CMAKE_CXX_FLAGS "-march=native -mavx -ffast-math -std=c++14")
add_compile_definitions(BLAZE)

set(cpp_dir "bolt/cpp")
set(cpp_src_dir "${cpp_dir}/src")
set(quantize_src_dir "${cpp_src_dir}/quantize")
set(utils_src_dir "${cpp_src_dir}/utils")

include_directories("${cpp_dir}")

file(GLOB_RECURSE quantize_srcs "${quantize_src_dir}/*.hpp" "${quantize_src_dir}/*.cpp")
file(GLOB_RECURSE utils_srcs "${utils_src_dir}/*.hpp" "${utils_src_dir}/*.cpp")

add_library(bolt STATIC "${quantize_srcs}" "${utils_srcs}" "${cpp_src_dir}/sketch.hpp")