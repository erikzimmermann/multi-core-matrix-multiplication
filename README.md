# multi-core-matrix-multiplication
This thesis is part of the seminar "Parallel Programming", which allows students to get a first insight into different technologies and techniques to enable the great advantage of multi-core computing. The aim of this paper is to provide an overview of different optimisations for simple matrix multiplication in a sequential and parallel way and to demonstrate the differences between single- and multi-core algorithms on a CPU using MPI and OpenMP.

### Execution
Every matrix multiplication implementation can be accessed by "matrix-multiplication" with additional parameters (see commands below).

### Commands

Allocate palma node: `salloc -N1 --ntasks-per-node=17 -p normal -t 01:00:00 --exclusive`  
Allocate palma node: `salloc -N4 --ntasks-per-node=17 -p normal -t 01:00:00 --exclusive`  

Required modules:
```
ml palma/2022a
ml GCC/11.3.0
ml OpenMPI/4.1.4
```

Compile with: `mpic++ -fopenmp test-matrix.cpp mpi_mm.cpp omp_mm.cpp mm.cpp -o test-matrix`

Run: `./matrix-multiplication <matrix-size> <naive, seq, omp[-<#threads>]>`   
Run with MPI: `mpirun -np 17 -npernode 17 -hostfile hosts.txt ./matrix-multiplication <matrix-size> mpi`  

