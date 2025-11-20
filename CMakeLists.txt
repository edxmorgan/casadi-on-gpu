cmake_minimum_required(VERSION 3.18)
cmake_policy(SET CMP0104 NEW)

project(cuda_ik_casadi LANGUAGES C CXX CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CUDA_STANDARD 14)

# Tell CMake which GPU architectures to compile for
# 86 is good for most recent NVIDIA laptop GPUs
set(CMAKE_CUDA_ARCHITECTURES 86)

include_directories(src)

add_executable(cuda_fk
    src/main.cu
    src/fk_alpha.cu
)

set_property(TARGET cuda_fk PROPERTY CUDA_SEPARABLE_COMPILATION ON)
