# multi-core-matrix-multiplication

Run c file for MPI: `mpirun -np 17 ./test-matrix 2000 mpi`  
Run c file for MPI: `mpirun -np 17 -npernode 17 -hostfile hosts.txt ./test-matrix 2000 mpi`  

Allocate palma node: `salloc -N1 --ntasks-per-node=17 -p normal -t 01:00:00 --exclusive`  
Allocate palma node: `salloc -N4 --ntasks-per-node=17 -p normal -t 02:00:00 --exclusive`  
Check queue: `squeue`

Other commands:
ml load palma/2022a
ml load GCC/11.3.0
ml load OpenMPI/4.1.4
mpic++ -fopenmp test-matrix.cpp mpi_mm.cpp omp_mm.cpp mm.cpp -o test-matrix

