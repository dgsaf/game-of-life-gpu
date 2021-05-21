#!/bin/bash --login
#SBATCH --account=courses0100
#SBATCH --reservation=courses0100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuq
#SBATCH --time=00:10:00
#SBATCH --export=NONE
#SBATCH --mem=1G

# load modules
module load gcc/8.3.0
module load cuda
module load cascadelake

# make binary
# make bin/01_gol_cpu_serial
srun --export=all -u -n 1 make clean
srun --export=all -u -n 1 make bin/01_gol_cpu_serial

# check if `n`, `m`, `nsteps` have been supplied to this script before executing
if [ -z "${n}" ] || [ -z "${m}" ] || [ -z "${nsteps}" ]
then
    echo "cpu.slurm: missing arguments"
else
    srun bin/01_gol_cpu_serial ${n} ${m} ${nsteps}
fi
