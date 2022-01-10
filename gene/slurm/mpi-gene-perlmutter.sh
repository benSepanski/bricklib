#!/bin/bash

MACHINE=perlmutter
EMAIL=ben.sepanski@gmail.com

for use_types in ON OFF ; do
    for cuda_aware in ON OFF ; do
        BRICKS_BUILD_DIR="perlmutter_use_types_${use_types}_cuda_aware_${cuda_aware}"
        for num_gz in 1 2 3 ; do
            for num_gpus in 4 8 12 16 ; do
                scripts_dir="generated-scripts"
                if [ ! -d ${scripts_dir} ] ; then
                    mkdir ${scripts_dir} ;
                fi
                job_name=mpi-gene-${num_gpus}gpus-${num_gz}gz-mpitypes${use_types}-cuda_aware${cuda_aware}
                python3 mpi-gene.py ${num_gpus} \
                    --bricks-build-dir ${BRICKS_BUILD_DIR} \
                    -M ${MACHINE} \
                    -e ${EMAIL} \
                    -J "${scripts_dir}/${job_name}" \
                    -t "02:00:00" \
                    --num-gz ${num_gz} \
                    > ${scripts_dir}/${job_name}.sh
            done
        done
    done
done
