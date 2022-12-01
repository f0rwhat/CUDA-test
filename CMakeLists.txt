cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(cuda_test LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

add_executable(${PROJECT_NAME} main.cu)

# Required for CUDA builds
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)