#!/bin/bash --login

# load modules
module load gcc/9.2.0
module load cuda
module load cascadelake

# make clean (*.slurm files will compile)
make clean

# parameters
nsteps="100"
n_set="1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384"

# submit jobs
echo "GOL (cpu|openmp) jobs"
echo "> nsteps: ${nsteps}"
echo "> n_set: ${n_set}"

for n in ${n_set}
do
    echo "> submitting (cpu|openmp) jobs for n=${n}"

    # submit cpu job
    sbatch --export="n=${n},m=${n},nsteps=${nsteps}" cpu.slurm

    # submit gpu job
    sbatch --export="n=${n},m=${n},nsteps=${nsteps}" gpu.slurm
done
