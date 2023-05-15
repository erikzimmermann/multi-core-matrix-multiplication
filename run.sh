#!/bin/sh

for size in 64 256 1024 2048 4096; do
  # iterate over all types
  for type in "naive" "seq" "omp" "mpi"; do
    # run ./test-matrix <size> <type> for everything
    echo "Running $size $type"

    if [ "$type" = "mpi" ]; then
      for processes in 5 17; do
        mpirun -np $processes --oversubscribe ./test-matrix $size $type >> run.out
      done
    else
      ./test-matrix $size $type >> run.out
    fi

  done
done