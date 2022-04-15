#!/bin/bash
export CUDA_VERSION=11.4

# Use gnu programming env
module load PrgEnv-gnu
module load cudatoolkit/${CUDA_VERSION}
module swap craype-network-ofi craype-network-ucx
module swap cray-mpich cray-mpich-ucx
module load cpe-cuda

# Cuda-aware MPI
# https://docs.nersc.gov/systems/perlmutter/#gpu-aware-mpi
export MPICH_GPU_SUPPORT_ENABLED=1
export CRAY_ACCEL_TARGET=nvidia80
module load craype-accel-nvidia80

# Make sure mpicc gets on path
export PATH="$PATH:${CRAY_MPICH_BASEDIR}/cray/10.0/bin"
# Handle issue from https://docs.nersc.gov/current/#new-issues (as of Jan 3, 2022)
export CUDA_MATH_LIBS="${CRAY_CUDATOOLKIT_DIR}/../../math_libs/${CUDA_VERSION}"
export CMAKE_PREFIX_PATH="${CUDA_MATH_LIBS}/include;${CUDA_MATH_LIBS}/targets/x86_64-linux/include;${CMAKE_PREFIX_PATH}"
# Make sure cufft can be found
export CMAKE_LIBRARY_PATH="${CUDA_MATH_LIBS}/lib64;${CMAKE_LIBRARY_PATH}"
export CUFFT_LINK_DIR="${CUDA_MATH_LIBS}/lib64"
export CUFFT_INCLUDE_DIR="${CUDA_MATH_LIBS}/include"

export UCX_TLS=rc,cuda_copy,cuda_ipc,gdr_copy
export UCX_IB_GPU_DIRECT_RDMA=yes

# Use wrappers
export CXX=CC
export CC=cc
module unload darshan