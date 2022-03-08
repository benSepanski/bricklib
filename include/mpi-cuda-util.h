//
// Created by Benjamin Sepanski on 12/1/21.
//

#ifndef BRICK_CUDA_MPI_UTIL_H
#define BRICK_CUDA_MPI_UTIL_H

#ifdef NDEBUG
#define mpiCheckCudaAware()

#else

#include <mpi.h>
#if defined(OPEN_MPI) && OPEN_MPI
#include "mpi-ext.h"
#endif
#include <cuda_runtime.h>
#define mpiCheckCudaAware() _mpiCheckCudaAware()

inline void _mpiCheckCudaAware() {
#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
#elif defined(MPIX_CUDA_AWARE_SUPPORT) && !MPIX_CUDA_AWARE_SUPPORT
#error "This MPI library does not have CUDA-aware support.\n"
#else
#warning "This MPI library cannot determine if there is CUDA-aware support.\n"
#endif /* MPIX_CUDA_AWARE_SUPPORT */
#if defined(MPIX_CUDA_AWARE_SUPPORT)
  if (1 != MPIX_Query_cuda_support()) {
    throw std::runtime_error("MPI does not have cuda support");
  }
#endif
}
#endif // NDEBUG

#endif // BRICK_CUDA_MPI_UTIL_H
