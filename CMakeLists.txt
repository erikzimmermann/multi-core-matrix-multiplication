cmake_minimum_required(VERSION 3.25)
project(matrix_multiplication_with_openmp)

set(CMAKE_CXX_STANDARD 17)

add_executable(matrix_multiplication_with_openmp test-matrix.cpp mpi_mm.cpp mpi_mm.h mm.cpp mm.h)

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(matrix_multiplication_with_openmp PRIVATE OpenMP::OpenMP_CXX)
endif()

# Find MPI package
find_package(MPI REQUIRED)

# Add MPI include directory to include path
target_include_directories(matrix_multiplication_with_openmp PRIVATE ${MPI_INCLUDE_PATH})

# Add MPI libraries to link list
target_link_libraries(matrix_multiplication_with_openmp PRIVATE ${MPI_LIBRARIES})
