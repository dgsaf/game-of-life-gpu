#!/bin/bash --login
#SBATCH --account=courses0100
#SBATCH --reservation=courses0100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuq
#SBATCH --time=00:10:00
#SBATCH --export=NONE

# load modules
module load gcc/8.3.0
module load cuda/10.2
module load cascadelake

# make binary
srun --export=all -u -n 1 make bin/game_of_life_cuda

# check if `n`, `m`, `nsteps` have been supplied to this script before executing
if [ -z "${n}" ] || [ -z "${m}" ] || [ -z "${nsteps}" ]
then
    echo "gpu.slurm: missing arguments"
else
    srun --export=all -u -n 1 bin/game_of_life_cuda ${n} ${m} ${nsteps}
fi
