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
)

file(GLOB GENERATED_CUDA_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/src/generated/*.cu"
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
        src/python/casadi_on_gpu_kernel_registry.cu
        ${GENERATED_CUDA_SOURCES}
    )

    target_include_directories(casadi_on_gpu PRIVATE
        src
        src/generated
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
            ${CMAKE_SOURCE_DIR}/tests/test_smoke.py
    )

    # Install destination for Python artifacts:
    # - under scikit-build (pip), install directly into wheel platlib root
    # - under manual CMake install, resolve site-packages relative to CMAKE_INSTALL_PREFIX
    if (DEFINED SKBUILD OR DEFINED SKBUILD_PROJECT_NAME)
        set(PYTHON_INSTALL_DIR ".")
    else ()
        execute_process(
            COMMAND ${Python3_EXECUTABLE} -c
                    "import os, site, sysconfig; prefix=r'${CMAKE_INSTALL_PREFIX}'; user_site=site.getusersitepackages(); platlib=sysconfig.get_path('platlib', vars={'base': prefix, 'platbase': prefix}); print(os.path.relpath(user_site, prefix) if user_site.startswith(prefix + os.sep) else os.path.relpath(platlib, prefix))"
            OUTPUT_VARIABLE PYTHON_INSTALL_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
    endif ()
    set(PYTHON_INSTALL_DIR "${PYTHON_INSTALL_DIR}" CACHE PATH "Python site-packages install dir (relative to prefix)")

    install(TARGETS casadi_on_gpu
        LIBRARY DESTINATION ${PYTHON_INSTALL_DIR}
    )
    install(FILES src/generated/kernels_manifest.json
        DESTINATION ${PYTHON_INSTALL_DIR}
    )
endif ()
