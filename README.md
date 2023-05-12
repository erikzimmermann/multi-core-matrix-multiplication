# matrix-multiplication-with-openmp

Build c file: `g++ -std=c++11 -fopenmp test-matrix.cpp mpi_mm.cpp -o test-matrix`  
Build c file for MPI: `mpic++ test-matrix.cpp mpi_mm.cpp -o test-matrix`  
Run c file for MPI: `mpirun -np 4 ./test-matrix mpi`  

Allocate palma node: `salloc -N1 --ntasks-per-node 1 -p normal -t 00:01:00`  
Allocate palma node: `salloc -N1 --ntasks-per-node 1 -p express -t 00:01:00`  
Check queue: `squeue`