#!/bin/bash --login

# parameters (matches parameters in jobs.sh)
nsteps="100"
n_set="1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384"

# performance file (with header)
performance_file="output/performance.nsteps-${nsteps}.txt"
printf "# n, cpu_time, gpu_time, speedup, kernel"
printf "# [], [ms], [ms], [], [ms]"

# extract
echo "GOL (cpu|cuda) extract"
echo "> nsteps: ${nsteps}"
echo "> n_set: ${n_set}"

for n in ${n_set}
do
    echo "> extracting (cpu|cuda) timing for n=${n}"

    # files to extract from
    cpu_file="output/timing-cpu.n-${n}.m-${n}.nsteps-${nsteps}.txt"
    gpu_file="output/timing-gpu-cuda.n-${n}.m-${n}.nsteps-${nsteps}.txt"

    # extract cpu_time
    if [ -f "${cpu_file}" ]
    then
        cpu_time=$(awk -F, 'FNR>2 { print $1 ; }' ${cpu_file})
    else
        cpu_time=""
        echo "> ${cpu_file} does not exists"
    fi

    # extract gpu_time, kernel_time
    if [ -f "${gpu_file}" ]
    then
        gpu_time=$(awk -F, 'FNR>2 { print $1 ; }' ${gpu_file})
        kernel_time=$(awk -F, 'FNR>2 { print $2 ; }' ${gpu_file})
    else
        gpu_time=""
        kernel_time=""
        echo "> ${gpu_file} does not exists"
    fi

    # if cpu_time, gpu_time both exist, calculate speedup
    if [ -z "${cpu_time}" ] || [ -z "${gpu_time}" ]
    then
        speedup=""
    else
        speedup=$(awk '{print $1/$2}' <<< "${cpu_time} ${gpu_time}")
    fi

    printf "${n} , ${cpu_time} , ${gpu_time} , ${speedup} , ${kernel_time}\n" \
           >> ${performance_file}

done
