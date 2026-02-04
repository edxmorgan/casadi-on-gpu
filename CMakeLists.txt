cmake_minimum_required(VERSION 3.18)
cmake_policy(SET CMP0104 NEW)

project(casadi_on_gpu LANGUAGES C CXX CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 14)
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

option(BUILD_PYTHON "Build Python bindings with pybind11" ON)
if (BUILD_PYTHON)
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    set(PYBIND11_LTO OFF CACHE BOOL "Disable pybind11 LTO" FORCE)
    find_package(pybind11 CONFIG REQUIRED)

    # Prevent pybind11 from adding its LTO target.
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)

    pybind11_add_module(casadi_on_gpu
        src/python/casadi_on_gpu_bindings.cpp
        src/python/casadi_on_gpu_kernels.cu
        src/generated/fk_alpha.cu
        src/generated/dynamics_blue.cu
    )

    target_include_directories(casadi_on_gpu PRIVATE
        src
        src/generated
        src/wrappers
        src/kernels
        src/python
    )

    set_property(TARGET casadi_on_gpu PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    set_property(TARGET casadi_on_gpu PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)
    target_compile_options(casadi_on_gpu PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-fno-lto>
    )
    target_link_options(casadi_on_gpu PRIVATE -fno-lto)

    enable_testing()
    add_test(NAME casadi_on_gpu_py_smoke
        COMMAND ${CMAKE_COMMAND} -E env
            PYTHONPATH=$<TARGET_FILE_DIR:casadi_on_gpu>
            ${Python3_EXECUTABLE}
            ${CMAKE_SOURCE_DIR}/python/test_smoke.py
    )

    # Install the Python extension into the prefix's site-packages.
    # Compute the interpreter's default platlib suffix: prefix/lib/pythonX.Y/(site|dist)-packages.
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c
                "import os, sysconfig; platlib = sysconfig.get_path('platlib'); prefix = os.path.dirname(os.path.dirname(os.path.dirname(platlib))); print(os.path.relpath(platlib, prefix))"
        OUTPUT_VARIABLE PYTHON_INSTALL_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    set(PYTHON_INSTALL_DIR "${PYTHON_INSTALL_DIR}" CACHE PATH "Python site-packages install dir (relative to prefix)")

    install(TARGETS casadi_on_gpu
        LIBRARY DESTINATION ${PYTHON_INSTALL_DIR}
    )
endif ()
