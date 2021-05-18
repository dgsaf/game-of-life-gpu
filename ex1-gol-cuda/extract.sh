#!/bin/bash --login

# parameters (matches parameters in jobs.sh)
nsteps="100"
n_set="1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384"

# performance file (with header)
performance_file="output/performance.nsteps-${nsteps}.txt"
printf "# n, cpu_time, gpu_time, speedup, kernel"
printf "# [], [ms], [ms], [], [ms]"

# submit jobs
echo "GOL (cpu|cuda) extract"
echo "> nsteps: ${nsteps}"
echo "> n_set: ${n_set}"

for n in ${n_set}
do
    echo "> extracting (cpu|cuda) timing for n=${n}"

    cpu=$(sed '3q;d' "output/timing-cpu.n-${n}.m-${n}.nsteps-${nsteps}.txt")
    gpu=$(sed '3q;d' "output/timing-gpu.n-${n}.m-${n}.nsteps-${nsteps}.txt")

    pattern="([0-9].+)"
    if [[ $cpu =~ $pattern ]] ; then
        cpu_time=${BASH_REMATCH[1]}
    else
        echo "failed to find cpu_time"
    fi

    if [[ $gpu =~ $pattern ]] ; then
        gpu_time=${BASH_REMATCH[1]}
        kernel_time=${BASH_REMATCH[2]}
    else
        echo "failed to find gpu_time, kernel_time"
    fi

    speedup=$(awk '{print $1/$2}' <<< "${cpu_time} ${gpu_time}")

    printf "${n} , ${cpu_time} , ${gpu_time} , ${speedup} , ${kernel_time}\n" \
           >> ${performance_file}

done
