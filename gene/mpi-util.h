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

/**
 * @brief build a cartesian communicator
 *
 * Assumes MPI_Init_thread has already been called.
 *
 * Prints some useful information about the MPI setup.
 *
 * @param[in] numProcsPerDim the number of MPI processes to put in each dimension.
 *                                  Product must match the number of MPI processes.
 * @param[in] perProcessExtent extent in each dimension for each individual MPI processes.
 * @param[in] allowRankReordering if true, allow ranks in the cartesian communicator
 *                                to be different than the ranks in MPI_COMM_WORLD
 * @return MPI_comm a cartesian communicator built from MPI_COMM_WORLD
 */
MPI_Comm buildCartesianComm(std::array<int, RANK> numProcsPerDim,
                            std::array<int, RANK> perProcessExtent,
                            bool allowRankReordering);

/**
 * @brief times func and prints stats
 *
 * @param func the func to run
 * @param mpiLayout the MPI layout used
 * @param totElems the number of elements
 */
void timeAndPrintMPIStats(std::function<void(void)> func, GeneMPILayout &mpiLayout,
                          double totElems);

#endif // BRICK_MPI_UTIL_H
