#!/bin/bash --login
#SBATCH --account=courses0100
#SBATCH --reservation=courses0100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuq
#SBATCH --time=00:15:00
#SBATCH --export=NONE

# load modules
module load pgi
module load cuda
module load cascadelake

# run with PROFILER=(ON|OFF) flag
PROFILER="OFF"

# make binary
srun --export=all -u -n 1 make bin/01_gol_cpu_serial PROFILER=${PROFILER}

# check if `n`, `m`, `nsteps` have been supplied to this script before executing
if [ -z "${n}" ] || [ -z "${m}" ] || [ -z "${nsteps}" ]
then
    echo "cpu.slurm: missing arguments"
else
    case "${PROFILER}" in
        ON)
            srun --export=all -u -n 1 bin/01_gol_cpu_serial ${n} ${m} ${nsteps}
            ;;

        OFF)
            srun --export=all -u -n 1 bin/01_gol_cpu_serial ${n} ${m} ${nsteps}
            ;;
        *)
            echo "cpu.slurm: profiler not set to one of (ON|OFF)"
            ;;
    esac
fi
