project(bolt)

set(CMAKE_CXX_FLAGS "-march=native -mavx -ffast-math -std=c++17")
add_compile_definitions(BLAZE)

set(cpp_dir "bolt/cpp")
set(cpp_src_dir "${cpp_dir}/src")
set(quantize_src_dir "${cpp_src_dir}/quantize")
set(utils_src_dir "${cpp_src_dir}/utils")

include_directories("${cpp_dir}")

file(GLOB_RECURSE quantize_srcs "${quantize_src_dir}/*.hpp" "${quantize_src_dir}/*.cpp")
file(GLOB_RECURSE utils_srcs "${utils_src_dir}/*.hpp" "${utils_src_dir}/*.cpp")

add_library(bolt STATIC "${quantize_srcs}" "${utils_srcs}" "${cpp_src_dir}/sketch.hpp")

set(test_dir "${cpp_dir}/test")
file(GLOB_RECURSE test_srcs "${test_dir}/*.hpp" "${test_dir}/*.cpp")
add_executable(bolt_test "${test_srcs}")
target_link_libraries(bolt_test PUBLIC bolt)

set(demo_dir "demo")
add_executable(demo "${demo_dir}/main.cc")
target_link_libraries(demo PUBLIC bolt)