#!/bin/bash

runs=7

for type in "mpi-4" "mpi-64" "mpi-49" "mpi-36" "omp-36" "mpi-25" "omp-25" "mpi-16" "omp-16" "mpi-9" "omp-9" "mpi-4" "omp-4"; do
  for size in 500 1000 1500 2000 2500 3000 3500 4000 4500; do

    times=()
    for i in $(seq 1 $runs); do
      echo "Running $size $type $i/$runs"
      if [[ "$type" == mpi-* ]]; then
        num_procs=${type#mpi-}
        num_procs=$((num_procs + 1))
        run=$(mpirun -np $num_procs --oversubscribe ./test-matrix $size mpi)
      elif [[ "$type" == mpiomp-* ]]; then
        num_procs=${type#mpiomp-}
        num_procs=$((num_procs + 1))
        run=$(mpirun -np $num_procs --oversubscribe ./test-matrix $size mpiomp)
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
