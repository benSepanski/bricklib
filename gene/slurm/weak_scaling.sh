#!/bin/bash
email=$1

function build_weak_job {
  num_gpus=$1
  global_extent=$2
  python3 mpi_slurm_gen.py ${num_gpus} \
    -M perlmutter \
    -t 00:03:00 \
    --always-cuda-aware \
    -J mpi_${num_gpus}_weak_exascale_gene_domain \
    -d ${global_extent} \
    -e ${email}
}

build_weak_job 4 70,16,24,48,32,2
build_weak_job 8 280,32,24,48,32,2
build_weak_job 16 560,32,24,48,32,2
build_weak_job 32 560,32,24,96,32,2
build_weak_job 64 560,32,24,96,64,2
build_weak_job 128 1120,32,24,96,64,2
build_weak_job 256 1120,32,48,96,64,2
build_weak_job 512 1120,32,48,96,128,2
