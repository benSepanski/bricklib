#!/bin/bash

BRICKS_BUILD_DIR=${HOME}/jetbrainsRemotes/bricklib/build-perlmutter
MACHINE=perlmutter
ACCOUNT=m2956
EMAIL=ben.sepanski@gmail.com

for num_gpus in 4 8 12 16 ; do
    python3 mpi-gene.py ${num_gpus} \
        --bricks-build-dir ${BRICKS_BUILD_DIR} \
        -M ${MACHINE} \
        -A ${ACCOUNT} \
        -e ${EMAIL} \
        > mpi-gene-perlmutter-${num_gpus}gpus.sh
done
