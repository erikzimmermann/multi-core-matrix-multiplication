# matrix-multiplication-with-openmp

Build c file: `mpic++ -fopenmp test-matrix.cpp mpi_mm.cpp mm.cpp -o test-matrix`  
Run c file for MPI: `mpirun -np 5 ./test-matrix mpi`  

Allocate palma node: `salloc -N1 --ntasks-per-node 1 -p normal -t 00:01:00`  
Allocate palma node: `salloc -N1 --ntasks-per-node 1 -p express -t 00:01:00`
Check queue: `squeue`