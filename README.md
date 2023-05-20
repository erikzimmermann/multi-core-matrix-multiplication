# multi-core-matrix-multiplication

Build c file: `mpic++ -fopenmp test-matrix.cpp mpi_mm.cpp omp_mm.cpp mm.cpp -o test-matrix`  
Run c file for MPI: `mpirun -np 17 ./test-matrix 2000 mpi`  

Allocate palma node: `salloc -N1 --ntasks-per-node=17 -p normal -t 01:00:00 --exclusive`  
Allocate palma node: `salloc -N4 --ntasks-per-node=17 -p normal -t 02:00:00 --exclusive`  
Check queue: `squeue`
