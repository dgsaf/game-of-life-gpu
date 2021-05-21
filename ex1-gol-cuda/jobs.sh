#!/bin/bash --login

# load modules
module load gcc/9.2.0
module load cuda/10.2

# parameters
nsteps="100"
n_set="1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384"

# submit jobs
echo "GOL (cpu|cuda) jobs"
echo "> nsteps: ${nsteps}"
echo "> n_set: ${n_set}"

for n in ${n_set}
do
    echo "> submitting (cpu|cuda) jobs for n=${n}"

    # submit cpu job
    sbatch --export="n=${n},m=${n},nsteps=${nsteps}" cpu.slurm

    # submit gpu job
    sbatch --export="n=${n},m=${n},nsteps=${nsteps}" gpu.slurm
done
