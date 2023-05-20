#!/bin/bash

runs=7

for type in "mpi-64" "mpiomp-64" "mpi-49" "mpiomp-49" "mpi-36" "mpiomp-36" "omp-36" "mpi-25" "mpiomp-25" "omp-25" "mpi-16" "mpiomp-16" "omp-16" "mpi-9" "mpiomp-9" "omp-9" "mpi-4" "mpiomp-4" "omp-4"; do
  if [[ "$type" == "mpi-64" || "$type" == "mpiomp-64" ]]; then
    arr=(496 1000 1496 2000 2496 3000 3496 4000 4496)
  elif [[ "$type" == "mpi-49" || "$type" == "mpiomp-49" ]]; then
    arr=(497 1001 1498 2002 2499 3003 3500 3997 4501)
  elif [[ "$type" == "mpi-36" || "$type" == "mpiomp-36" ]]; then
    arr=(498 1002 1500 1998 2502 3000 3498 4002 4500)
  elif [[ "$type" == "mpi-25" || "$type" == "mpiomp-25" ]]; then
    arr=(500 1000 1500 2000 2500 3000 3500 4000 4500)
  elif [[ "$type" == "mpi-16" || "$type" == "mpiomp-16" ]]; then
    arr=(500 1000 1500 2000 2500 3000 3500 4000 4500)
  elif [[ "$type" == "mpi-9" || "$type" == "mpiomp-9" ]]; then
    arr=(501 999 1500 2001 2499 3000 3501 3999 4500)
  elif [[ "$type" == "mpi-4" || "$type" == "mpiomp-4" ]]; then
    arr=(500 1000 1500 2000 2500 3000 3500 4000 4500)
  else
    arr=(500 1000 1500 2000 2500 3000 3500 4000 4500)  # Default arr for any other type
  fi

  for size in "${arr[@]}"; do
    times=()
    for i in $(seq 1 $runs); do
      echo "Running $size $type $i/$runs"
      if [[ "$type" == mpi-* ]]; then
        num_procs=${type#mpi-}
        num_procs=$((num_procs + 1))
        run=$(mpirun -np $num_procs -npernode 17 -hostfile hosts.txt --oversubscribe ./test-matrix $size mpi)
      elif [[ "$type" == mpiomp-* ]]; then
        num_procs=${type#mpiomp-}
        num_procs=$((num_procs + 1))
        run=$(mpirun -np $num_procs -npernode 17 -hostfile hosts.txt --oversubscribe ./test-matrix $size mpiomp)
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
