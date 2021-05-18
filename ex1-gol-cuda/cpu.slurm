#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --parition=gpuq
#SBATCH --time=00:05:00
#SBATCH --export=NONE

# load modules
module load gcc/8.3.0

# make binary
make bin/game_of_life

# check if `n`, `m`, `nsteps` have been supplied to this script before executing
if [ -z "${n}" ] || [ -z "${m}" ] || [ -z "${nsteps}" ]
then
    echo "cpu.slurm: missing arguments"
else
    srun bin/game_of_life ${n} ${m} ${nsteps}
fi
