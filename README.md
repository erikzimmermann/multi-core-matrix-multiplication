# multi-core-matrix-multiplication

Build c file: `mpic++ -fopenmp test-matrix.cpp mpi_mm.cpp mm.cpp -o test-matrix`  
Run c file for MPI: `mpirun -np 17 ./test-matrix mpi`  

Allocate palma node: `salloc -N1 --ntasks-per-node=17 -p normal -t 01:00:00 --exclusive`  
Allocate palma node: `salloc -N1 --ntasks-per-node=17 -p express -t 01:00:00`  
Allocate palma node: `salloc -N1 --ntasks-per-node=17 -p bigsmp -t 01:00:00`  
Check queue: `squeue`
