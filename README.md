# matrix-multiplication-with-openmp

Build c file: `mpic++ -fopenmp test-matrix.cpp mpi_mm.cpp mm.cpp -o test-matrix`  
Run c file for MPI: `mpirun -np 17 ./test-matrix mpi`  

Allocate palma node: `salloc -N1 --ntasks-per-node=17 -p normal -t 01:00:00`
Allocate palma node: `salloc -N1 --ntasks-per-node=17 -p express -t 01:00:00`
Allocate palma node: `salloc -N1 --ntasks-per-node=17 -p bigsmp -t 01:00:00`
Check queue: `squeue`

sbatch --exclusive --partition=normal --nodes=1 --ntasks-per-node=65 --time=00:04:00 --output=/scratch/tmp/e_zimm08/run-01.out --error=/scratch/tmp/e_zimm08/run-01.error --mail-type=ALL --mail-user=erik.zimmermann@uni-muenster.de ./run.sh