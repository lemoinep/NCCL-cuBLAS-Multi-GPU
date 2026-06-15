cmake_minimum_required(VERSION 3.18)
project(nccl_cublas_matmul LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES native)

find_package(CUDAToolkit REQUIRED)
find_package(MPI REQUIRED)
find_library(NCCL_LIBRARY nccl)
find_library(CUBLAS_LIBRARY cublas)

if(NOT NCCL_LIBRARY)
  message(FATAL_ERROR "NCCL library not found")
endif()
if(NOT CUBLAS_LIBRARY)
  message(FATAL_ERROR "cuBLAS library not found")
endif()

add_library(core
  src/gpu_utils.cpp
  src/scheduler.cpp
  src/nccl_utils.cpp
  src/cublas_matmul.cu
)

target_include_directories(core PUBLIC include)
target_link_libraries(core PUBLIC CUDA::cudart ${NCCL_LIBRARY} ${CUBLAS_LIBRARY})

add_executable(nccl_cublas_a src/main_a.cu)
target_link_libraries(nccl_cublas_a PRIVATE core)

add_executable(nccl_cublas_c src/main_c_mpi.cu)
target_link_libraries(nccl_cublas_c PRIVATE core MPI::MPI_CXX)