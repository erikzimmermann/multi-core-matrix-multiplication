cmake_minimum_required(VERSION 3.25)
project(multi_core_matrix_multiplication)

set(CMAKE_CXX_STANDARD 17)

add_executable(multi_core_matrix_multiplication matrix-multiplication.cpp mm_mpi.cpp mm_mpi.h mm_omp.cpp mm_omp.h mm_seq.cpp mm_seq.h)

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(multi_core_matrix_multiplication PRIVATE OpenMP::OpenMP_CXX)
endif()

# Find MPI package
find_package(MPI REQUIRED)

# Add MPI include directory to include path
target_include_directories(multi_core_matrix_multiplication PRIVATE ${MPI_INCLUDE_PATH})

# Add MPI libraries to link list
target_link_libraries(multi_core_matrix_multiplication PRIVATE ${MPI_LIBRARIES})
