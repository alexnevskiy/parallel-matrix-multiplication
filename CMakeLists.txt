cmake_minimum_required(VERSION 3.25)
project(parallels)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS -pthread)
set(THREADS_PREFER_PTHREAD_FLAG ON)
#set(ENV{OPENCV_DIR} "C:\\OpenCV_4.6.0\\opencv\\build")
#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

find_package(MPI REQUIRED)
#find_package(OpenCV REQUIRED)

add_executable(parallels main.cpp Matrix.cpp Matrix.h matrix_mul_pthreads.cpp matrix_mul_pthreads.h matrix_mul_mpi.h matrix_mul_mpi.cpp utils.cpp utils.h test.cpp test.h)

#find_package(Threads REQUIRED)
#target_link_libraries(parallels Threads::Threads)
#target_link_libraries(parallels pthread)

include_directories(SYSTEM ${MPI_INCLUDE_PATH})
target_link_libraries(parallels ${MPI_C_LIBRARIES})

include_directories(E:\\cpp_libraries\\Eigen3)
#find_package (Eigen3 3.4 REQUIRED NO_MODULE)
#target_link_libraries (example Eigen3::Eigen)

#include_directories(${OpenCV_INCLUDE_DIRS})
#set(OpenCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs)
#target_link_libraries(parallels ${OpenCV_LIBS})
