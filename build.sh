module load palma/2022a
module load GCC/11.3.0
module load OpenMPI/4.1.4

mpic++ -fopenmp test-matrix.cpp mpi_mm.cpp mm.cpp -o test-matrix