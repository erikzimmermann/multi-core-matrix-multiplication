#!/bin/bash

runs=7

for type in "naive" "seq" "omp" "mpi-5" "mpi-17"; do
  for size in 500 1000 1500 2000 2500 3000; do

    times=()
    for i in $(seq 1 $runs); do
      echo "Running $size $type $i/$runs"
      if [ "$type" = "mpi-5" ]; then
        run=$(mpirun -np 5 --oversubscribe ./test-matrix $size mpi)
      elif [ "$type" = "mpi-17" ]; then
        run=$(mpirun -np 17 --oversubscribe ./test-matrix $size mpi)
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
