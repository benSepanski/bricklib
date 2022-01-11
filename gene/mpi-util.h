//
// Created by Benjamin Sepanski on 12/12/21.
//

#ifndef BRICK_MPI_UTIL_H
#define BRICK_MPI_UTIL_H

#include "MPILayout.h"
#include "gene-6d-stencils.h"
#include "util.h"
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
 * @param func[in] the func to run
 * @param mpiLayout[in] the MPI layout used
 * @param totElems[in] the number of elements
 * @param numGhostZones[in] number of ghost zones
 * @param csvDataRecorder[in,out] csv data recorder to record data in (if rank is 0)
 */
void timeAndPrintMPIStats(std::function<void(void)> func, GeneMPILayout &mpiLayout,
                          double totElems, unsigned numGhostZones, CSVDataRecorder &csvDataRecorder);

/**
 * @brief parse args
 * @param[out] perProcessDomainSize the extent in each dimension of the domain
 * @param[out] numProcsPerDim the number of processes per dimension
 * @param[out] numGhostZones the number of ghost zones to use
 * @param[out] outputFileName the output file to write to
 * @param[out] appendToFile true if the file should be appended to
 * @param[in] in input stream to read from
 *
 * @return the number of iterations, with default 100
 */
trial_iter_count parse_mpi_args(std::array<int, RANK> *perProcessDomainSize,
                            std::array<int, RANK> *numProcsPerDim,
                            int *numGhostZones,
                            std::string *outputFileName,
                            bool *appendToFile,
                            std::istream &in);
#endif // BRICK_MPI_UTIL_H
