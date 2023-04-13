cmake_minimum_required(VERSION 3.25)
project(parallels)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS -pthread)
set(THREADS_PREFER_PTHREAD_FLAG ON)

find_package(MPI REQUIRED)

add_executable(parallels main.cpp Matrix.cpp Matrix.h matrix_mul_pthreads.cpp matrix_mul_pthreads.h matrix_mul_mpi.h matrix_mul_mpi.cpp utils.cpp utils.h test.cpp test.h)

include_directories(SYSTEM ${MPI_INCLUDE_PATH})
target_link_libraries(parallels ${MPI_C_LIBRARIES})

include_directories(E:\\cpp_libraries\\Eigen3)