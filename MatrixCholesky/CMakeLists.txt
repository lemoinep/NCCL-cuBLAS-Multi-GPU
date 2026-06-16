cmake_minimum_required(VERSION 3.18)
project(nccl_cusolver_cholesky LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES native)

find_package(CUDAToolkit REQUIRED)
find_package(MPI REQUIRED)
find_library(NCCL_LIBRARY nccl)
find_library(CUBLAS_LIBRARY cublas)
find_library(CUSOLVER_LIBRARY cusolver)

if(NOT NCCL_LIBRARY)
  message(FATAL_ERROR "NCCL library not found")
endif()
if(NOT CUBLAS_LIBRARY)
  message(FATAL_ERROR "cuBLAS library not found")
endif()
if(NOT CUSOLVER_LIBRARY)
  message(FATAL_ERROR "cuSOLVER library not found")
endif()

add_library(core
  src/gpu_utils.cpp
  src/nccl_utils.cpp
  src/linalg_utils.cu
  src/owner_utils.cpp
  src/task_scheduler.cpp
  src/validation.cpp
)
target_include_directories(core PUBLIC include)
target_link_libraries(core PUBLIC CUDA::cudart ${NCCL_LIBRARY} ${CUBLAS_LIBRARY} ${CUSOLVER_LIBRARY})

add_executable(nccl_cusolver_cholesky src/main_c_mpi.cu)
target_link_libraries(nccl_cusolver_cholesky PRIVATE core MPI::MPI_CXX)

