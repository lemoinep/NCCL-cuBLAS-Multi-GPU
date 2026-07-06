cmake_minimum_required(VERSION 3.18)
project(NCCL_CuBLAS_MultiGPU LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

find_path(NCCL_INCLUDE_DIR nccl.h
    HINTS ${CUDAToolkit_INCLUDE_DIRS} /usr/include /usr/local/include /opt/nccl/include)

find_library(NCCL_LIBRARY nccl
    HINTS /usr/lib /usr/lib64 /usr/local/lib /usr/local/lib64 /opt/nccl/lib /opt/nccl/lib64)

if(NOT NCCL_INCLUDE_DIR OR NOT NCCL_LIBRARY)
    message(FATAL_ERROR "NCCL not found. Set NCCL_INCLUDE_DIR and NCCL_LIBRARY or load the NCCL module.")
endif()

add_executable(nccl-topopipe
    src/main.cpp
    src/topology.cpp
    src/cublas_ops.cpp
    src/nccl_pipeline.cpp
)

target_include_directories(nccl-topopipe PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
    ${NCCL_INCLUDE_DIR}
)

target_link_libraries(nccl-topopipe PRIVATE
    CUDA::cublas
    CUDA::cudart
    ${NCCL_LIBRARY}
)

set_target_properties(nccl-topopipe PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_compile_options(nccl-topopipe PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

option(BUILD_TESTS "Build tests" OFF)

if(BUILD_TESTS)
    enable_testing()
endif()
