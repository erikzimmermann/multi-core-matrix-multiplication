cmake_minimum_required(VERSION 3.25)
project(matrix_multiplication_with_openmp)

set(CMAKE_CXX_STANDARD 17)

add_executable(matrix_multiplication_with_openmp main_openmp.cpp main_mpi.cpp)

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(matrix_multiplication_with_openmp PUBLIC OpenMP::OpenMP_CXX)
endif()

# Find MPI package
find_package(MPI REQUIRED)

# Add MPI include directory to include path
include_directories(${MPI_INCLUDE_PATH})

# Add MPI libraries to link list
set(LIBS ${LIBS} ${MPI_LIBRARIES})

target_link_libraries(matrix_multiplication_with_openmp ${LIBS})