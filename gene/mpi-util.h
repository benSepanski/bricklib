//
// Created by Benjamin Sepanski on 12/12/21.
//

#ifndef BRICK_MPI_UTIL_H
#define BRICK_MPI_UTIL_H

#include "gene-6d-stencils.h"
#include "MPILayout.h"
#include <functional>

// useful types
typedef brick::MPILayout<FieldBrickDimsType, CommIn_kl> GeneMPILayout;

// global constants set by CLI
extern unsigned NUM_EXCHANGES; ///< how many mpi exchanges?
extern unsigned NUM_WARMUPS;   ///< how many warmup iters?

MPI_Comm buildCartesianComm(std::array<int, RANK> numProcsPerDim,
                            std::array<int, RANK> perProcessExtent);

void timeAndPrintMPIStats(std::function<void(void)> func, GeneMPILayout &mpiLayout,
                          double totElems);

#endif // BRICK_MPI_UTIL_H
