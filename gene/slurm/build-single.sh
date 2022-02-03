#!/usr/bin/env bash

usage="Usage: ./build-single.sh [machine name]"

if [ $# -lt 1 ] ; then
  echo "Missing machine name"
  echo "${usage}"
  exit 1
fi
declare -A machine_to_cuda_arch=( ["perlmutter"]="80"
                                  ["cori"]=70
                )
machine="${1}"
if [[ "${machine_to_cuda_arch[*]}" =~ ${machine} ]] ; then
  echo "Unrecognized machine ${machine}"
  exit 1
fi
if [[ "${machine}" == "perlmutter" ]] ; then
  PERLMUTTER=ON
else
  PERLMUTTER=OFF
fi
cuda_arch=${machine_to_cuda_arch[${machine}]}

brick_dims=( 
  "2,32,2,2,1,1"
  "2,16,2,2,2,1"
  "2,16,2,4,1,1"
  "2,16,4,2,1,1"
  "4,16,2,2,1,1"
  "2,8,2,2,4,1"
  "4,8,2,2,2,1"
  "2,8,4,4,1,1"
  "2,16,2,2,1,1"
  "2,8,2,2,2,1"
  "2,8,2,4,1,1"
  "2,8,4,2,1,1"
  "4,8,2,2,1,1"
  "2,4,2,2,4,1"
  "4,4,2,2,2,1"
  "2,4,4,4,1,1"
)

cmake_src_dir=$(realpath ../../)
build_parent_dir="cmake-builds"
for brick_dim in ${brick_dims[*]} ; do
    echo "building with brick-dimension i,j,k,l,m,n = ${brick_dim}";
    dir_name="${build_parent_dir}/single-builds/${machine}/brick_size${brick_dim//,/x}";
    if [ ! -d "${dir_name}" ] ; then
        mkdir -p "${dir_name}"
    fi
    (cd "${dir_name}" && \
     cmake -S "${cmake_src_dir}" \
           -B . \
          -DCMAKE_CUDA_ARCHITECTURES="${cuda_arch}" \
          -DCMAKE_INSTALL_PREFIX=bin \
          -DGENE6D_USE_TYPES=OFF \
          -DGENE6D_CUDA_AWARE=OFF \
          -DGENE6D_BRICK_DIM=${brick_dim} \
          -DCMAKE_CUDA_FLAGS="-lineinfo -gencode arch=compute_${cuda_arch},code=[sm_${cuda_arch},lto_${cuda_arch}]" \
          -DCMAKE_BUILD_TYPE=Release \
          -DPERLMUTTER=${PERLMUTTER} && \
      make -j 20)
done;
