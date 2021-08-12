#!/bin/bash

#SBATCH -q regular
#SBATCH -C gpu
#SBATCH --time 02:00:00

#SBATCH --nodes 1
#SBATCH --gpus 2
#SBATCH --ntasks 2
#SBATCH --cpus-per-task 20 
#SBATCH --gpus-per-task 1

#SBATCH --job-name weak-gene-6d-main/cori-single-node
#SBATCH --error weak-gene-6d-main_cori-single-node.err
#SBATCH --output weak-gene-6d-main_cori-single-node.out

#SBATCH --mail-type=ALL
#SBATCH -A m2956
#SBATCH --mail-user=ben.sepanski@gmail.com

# handle openmp thread affinity
export OMP_PROC_BIND=true  # allow affinity settings
export OMP_PLACES=threads  # bind openmp threads to physical hyperthreads
export OMP_NUM_THREADS=10  # 10 threads/process

# set up for problem & define any environment variables here
export gtensor_DIR=${HOME}/bricks2021/gtensor/bin
export BRICKS_DIR=${HOME}/bricks2021/bricklib

#run the application:
srun ${BRICKS_DIR}/dbuild/weak/gene6d

# perform any cleanup or short post-processing here
