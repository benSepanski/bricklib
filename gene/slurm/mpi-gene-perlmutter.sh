#!/bin/bash

BRICKS_BUILD_DIR=${HOME}/jetbrainsRemotes/bricklib/build-perlmutter
MACHINE=perlmutter
EMAIL= # FILL ME IN

for num_gpus in 4 8 12 16 ; do
    python3 mpi-gene.py ${num_gpus} \
        --bricks-build-dir ${BRICKS_BUILD_DIR} \
        -M ${MACHINE} \
        -e ${EMAIL} \
        > mpi-gene-perlmutter-${num_gpus}gpus.sh
done
