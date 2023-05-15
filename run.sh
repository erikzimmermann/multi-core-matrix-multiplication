#!/bin/sh

for size in 64 128 256 512 1024 2048; do
  for type in "naive" "seq" "omp" "mpi-5" "mpi-17"; do
    echo "Running $size $type"

    min=
    for i in $(seq 1 10); do
      if [ "$type" = "mpi-5" ]; then
        run=$(mpirun -np 5 --oversubscribe ./test-matrix $size mpi)
      elif [ "$type" = "mpi-17" ]; then
        run=$(mpirun -np 17 --oversubscribe ./test-matrix $size mpi)
      else
        run=$(./test-matrix $size $type)
      fi

      run=$(echo "$run" | cut -d';' -f1)

#      if [ -z "$min" ] || [ "$run" -lt "$min" ]; then
#          min=$run
#      fi

      if [ -z "$min" ] || [ "$(echo "$run < $min" | bc -l)" -eq 1 ]; then
          min=$run
      fi
    done

    echo "${type}: ${size}x${size}, Time: ${min}s" >> run.out
  done
  echo "" >> run.out
done