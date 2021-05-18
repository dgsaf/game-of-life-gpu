#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuq
#SBATCH --time=00:05:00
#SBATCH --export=NONE

# load modules
module load gcc/8.3.0 cuda/10.2

# make binary
make bin/game_of_life_cuda

# check if `n`, `m`, `nsteps` have been supplied to this script before executing
if [ -z "${n}" ] || [ -z "${m}" ] || [ -z "${nsteps}" ]
then
    echo "gpu.slurm: missing arguments"
else
    srun bin/game_of_life_cuda ${n} ${m} ${nsteps}
fi
