#!/bin/bash

runs=7

for type in "mpi-64" "mpi-49" "mpi-36" "omp-36" "mpi-25" "omp-25" "mpi-16" "omp-16" "mpi-9" "omp-9" "mpi-4" "omp-4"; do
  for size in 500 1000 1500 2000 2500 3000 3500 4000 4500; do

    times=()
    for i in $(seq 1 $runs); do
      echo "Running $size $type $i/$runs"
      if [ "$type" = "mpi-4" ]; then
        run=$(mpirun -np 5 --oversubscribe ./test-matrix $size mpi)
      elif [ "$type" = "mpi-9" ]; then
        run=$(mpirun -np 10 --oversubscribe ./test-matrix $size mpi)
      elif [ "$type" = "mpi-16" ]; then
        run=$(mpirun -np 17 --oversubscribe ./test-matrix $size mpi)
      elif [ "$type" = "mpi-25" ]; then
        run=$(mpirun -np 26 --oversubscribe ./test-matrix $size mpi)
      elif [ "$type" = "mpi-36" ]; then
        run=$(mpirun -np 37 --oversubscribe ./test-matrix $size mpi)
      elif [ "$type" = "mpi-49" ]; then
        run=$(mpirun -np 50 --oversubscribe ./test-matrix $size mpi)
      elif [ "$type" = "mpi-64" ]; then
        run=$(mpirun -np 65 --oversubscribe ./test-matrix $size mpi)
      else
        run=$(./test-matrix $size $type)
      fi

      run=$(echo "$run" | cut -d';' -f1)
      times+=("$run")
    done

    sorted_times=($(printf '%s\n' "${times[@]}" | sort -n))

    sum=0
    for (( i=1; i<(runs-1); i++ )); do
      sum=$(echo "${sum} + ${sorted_times[i]}" | bc -l)
    done

    avg=$(echo "$sum / ($runs-2)" | bc -l)
    avg=$(printf "%.*f" 4 "$avg")

    echo "${type}: ${size}x${size}, Time: ${avg}s" >> run.out
  done
  echo "" >> run.out
done
