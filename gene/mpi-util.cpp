//
// Created by Benjamin Sepanski on 12/12/21.
//

#include "mpi-util.h"

// global constants set by CLI
unsigned NUM_EXCHANGES; ///< how many mpi exchanges?
unsigned NUM_WARMUPS;   ///< how many warmup iters?

MPI_Comm buildCartesianComm(std::array<int, RANK> numProcsPerDim,
                            std::array<int, RANK> perProcessExtent,
                            bool allowRankReordering) {
  // get number of MPI processes and my rank
  int size, rank;
  check_MPI(MPI_Comm_size(MPI_COMM_WORLD, &size));
  check_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

  // make sure num_procs_per_dim has product to number of processes
  int prodOfProcsPerDim =
      std::accumulate(numProcsPerDim.begin(), numProcsPerDim.end(), 1, std::multiplies<size_t>());
  if (prodOfProcsPerDim != size) {
    std::ostringstream error_stream;
    error_stream << "Product of number of processes per dimension is " << prodOfProcsPerDim
                 << " which does not match number of MPI processes (" << size << ")\n";
    throw std::runtime_error(error_stream.str());
  }

  // set up processes on a cartesian communication grid
  std::array<int, RANK> periodic{};
  for (int i = 0; i < RANK; ++i) {
    periodic[i] = true;
  }
  MPI_Comm cartesianComm;
  check_MPI(MPI_Cart_create(MPI_COMM_WORLD, RANK, numProcsPerDim.data(), periodic.data(),
                            allowRankReordering, &cartesianComm));
  if (cartesianComm == MPI_COMM_NULL) {
    std::cerr << "Rank " << rank << " received MPI_COMM_NULL communicator" << "\n";
    std::cerr << "MPI_COMM_WORLD Size: " << size << "\n";
    throw std::runtime_error("Failure in cartesian comm setup");
  }

  // return the communicator
  return cartesianComm;
}


void timeAndPrintMPIStats(std::function<void(void)> func, GeneMPILayout &mpiLayout,
                          double totElems) {
  // warmup function
  for (int i = 0; i < NUM_WARMUPS; ++i) {
    func();
  }

  // Reset mpi statistics and time the function
  packtime = calltime = waittime = movetime = calctime = 0;
  for (int i = 0; i < NUM_EXCHANGES; ++i) {
    func();
  }

  size_t totalExchangeSize = 0;
  for (const auto g : mpiLayout.getBrickDecompPtr()->ghost) {
    totalExchangeSize += g.len * FieldBrick_kl::BRICKSIZE * sizeof(bElem);
  }

  int totalNumIters = NUM_EXCHANGES * NUM_GHOST_ZONES;
  mpi_stats calcTimeStats = mpi_statistics(calctime / totalNumIters, MPI_COMM_WORLD);
  mpi_stats calcSpeedStats =
      mpi_statistics(totElems / (double)calctime / 1.0e9 * totalNumIters, MPI_COMM_WORLD);
  mpi_stats packTimeStats = mpi_statistics(packtime / totalNumIters, MPI_COMM_WORLD);
  mpi_stats packSpeedStats =
      mpi_statistics(totalExchangeSize / 1.0e9 / packtime * totalNumIters, MPI_COMM_WORLD);
  mpi_stats mpiCallTimeStats = mpi_statistics(calltime / totalNumIters, MPI_COMM_WORLD);
  mpi_stats mpiWaitTimeStats = mpi_statistics(waittime / totalNumIters, MPI_COMM_WORLD);
  mpi_stats mpiSpeedStats = mpi_statistics(
      totalExchangeSize / 1.0e9 / (calltime + waittime) * totalNumIters, MPI_COMM_WORLD);
  mpi_stats mpiExchangeSizeStats =
      mpi_statistics((double)totalExchangeSize * 1.0e-6, MPI_COMM_WORLD);
  double total =
      calcTimeStats.avg + mpiWaitTimeStats.avg + mpiCallTimeStats.avg + packTimeStats.avg;

  int rank;
  check_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

  if (rank == 0) {
    std::cout << "Average Per-Process Total Time: " << total << std::endl;

    std::cout << "calc " << calcTimeStats << std::endl;
    std::cout << "  | Calc speed (GStencil/s): " << calcSpeedStats << std::endl;
    std::cout << "pack " << packTimeStats << std::endl;
    std::cout << "  | Pack speed (GB/s): " << packSpeedStats << std::endl;
    std::cout << "call " << mpiCallTimeStats << std::endl;
    std::cout << "wait " << mpiWaitTimeStats << std::endl;
    std::cout << "  | MPI size (MB): " << mpiExchangeSizeStats << std::endl;
    std::cout << "  | MPI speed (GB/s): " << mpiSpeedStats << std::endl;

    double perf = (double)totElems * 1.0e-9;
    perf = perf / total;
    std::cout << "perf " << perf << " GStencil/s" << std::endl;
    std::cout << std::endl;
  }
}