cmake_minimum_required(VERSION 3.18)
cmake_policy(SET CMP0104 NEW)

project(casadi_on_gpu LANGUAGES C CXX CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CUDA_STANDARD 14)

# 86 is good for most recent NVIDIA laptop GPUs
set(CMAKE_CUDA_ARCHITECTURES 86)

include_directories(
    src
    src/generated
    src/wrappers
    src/kernels
)

add_executable(run_kinematics_gpu
    src/kernels/kinematics_main.cu
    src/generated/fk_alpha.cu
)

set_property(TARGET run_kinematics_gpu PROPERTY CUDA_SEPARABLE_COMPILATION ON)

add_executable(run_dynamics_gpu
    src/kernels/dynamics_main.cu
    src/generated/dynamics_blue.cu
)

set_property(TARGET run_dynamics_gpu PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_compile_definitions(run_dynamics_gpu PRIVATE
    POSTERIOR_BIN_PATH="${CMAKE_SOURCE_DIR}/src/posterior.bin"
)
