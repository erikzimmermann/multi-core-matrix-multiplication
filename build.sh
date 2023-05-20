ml load palma/2022a
ml load GCC/11.3.0
ml load OpenMPI/4.1.4
mpic++ -fopenmp test-matrix.cpp mpi_mm.cpp omp_mm.cpp mm.cpp -o test-matrix
