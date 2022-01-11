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
                          double totElems, unsigned numGhostZones, CSVDataRecorder &dataRecorder) {
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

  int totalNumIters = NUM_EXCHANGES * numGhostZones;
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
    dataRecorder.newRow();
    std::cout << "Average Per-Process Total Time: " << total << std::endl;
    dataRecorder.record("AveragePerProcessTotalTime(s)", total);

    std::cout << "calc " << calcTimeStats << std::endl;
    dataRecorder.recordMPIStats("calculationTime(s)", calcTimeStats);
    std::cout << "  | Calc speed (GStencil/s): " << calcSpeedStats << std::endl;
    dataRecorder.recordMPIStats("calcSpeed(GStencil/s)", calcSpeedStats);
    std::cout << "pack " << packTimeStats << std::endl;
    dataRecorder.recordMPIStats("packTime(s)", packTimeStats);
    std::cout << "  | Pack speed (GB/s): " << packSpeedStats << std::endl;
    dataRecorder.recordMPIStats("PackSpeed(GStencil/s)", packSpeedStats);
    std::cout << "call " << mpiCallTimeStats << std::endl;
    dataRecorder.recordMPIStats("MPICallTime(s)", mpiCallTimeStats);
    std::cout << "wait " << mpiWaitTimeStats << std::endl;
    dataRecorder.recordMPIStats("MPIWaitTime(s)", mpiWaitTimeStats);
    std::cout << "  | MPI size (MB): " << mpiExchangeSizeStats << std::endl;
    dataRecorder.recordMPIStats("MPISize(MB)", mpiExchangeSizeStats);
    std::cout << "  | MPI speed (GB/s): " << mpiSpeedStats << std::endl;
    dataRecorder.recordMPIStats("MPISpeed(GB/s)", mpiSpeedStats);

    double perf = (double)totElems * 1.0e-9;
    perf = perf / total;
    std::cout << "perf " << perf << " GStencil/s" << std::endl;
    std::cout << std::endl;
    dataRecorder.record("TotalPerformance(GStencil/s)", perf);
  }
}
trial_iter_count parse_mpi_args(std::array<int, RANK> *perProcessDomainSize,
                            std::array<int, RANK> *numProcsPerDim,
                            int *numGhostZones,
                            std::string *outputFileName,
                            bool *appendToFile,
                            std::istream &in) {
  std::string optionString;
  trial_iter_count iterCount{};
  iterCount.num_iters = 100;
  iterCount.num_warmups = 5;
  std::vector<unsigned> tuple;
  bool readDomSize = false, readNumIters = false, readNumProcsPerDim = false,
       readNumWarmups = false;
  *outputFileName = "results.csv";
  *appendToFile = false;
  *numGhostZones = 1;
  std::string helpString = "Program options\n"
                            "  -h: show help (this message)\n"
                            "  Domain size,  in array order contiguous first\n"
                            "  -d: comma separated Int[6], per-process domain size\n"
                            "  Num Tasks per dimension, in array order contiguous first\n"
                            "  -p: comma separated Int[6], num process per dimension"
                            "  Benchmark control:\n"
                            "  -I: number of iterations, default 100 \n"
                            "  -W: number of warmup iterations, default 5\n"
                            "  -G: number of ghost zones, default 1\n"
                            "  -o: csv file to write to (default results.csv)\n"
                            "  -a: If passed, will append data to output file (if it already exists)\n"
                            "Example usage:\n"
                            "  weak/gene6d -d 70,16,24,48,32,2 -p 1,1,3,1,2,1\n";
  std::ostringstream errorStream;
  while (in >> optionString) {
    if (optionString[0] != '-' || optionString.size() != 2) {
      errorStream << "Unrecognized option " << optionString << std::endl;
    }
    if (!errorStream.str().empty()) {
      errorStream << helpString;
      throw std::runtime_error(errorStream.str());
    }
    switch (optionString[1]) {
    case 'a':
      *appendToFile = true;
      break;
    case 'd':
      tuple = read_uint_tuple(in, ',');
      if (readDomSize) {
        errorStream << "-d option should only be passed once" << std::endl;
      } else if (tuple.size() != RANK) {
        errorStream << "Expected extent of length " << RANK << ", not " << tuple.size();
      } else {
        std::copy(tuple.begin(), tuple.end(), perProcessDomainSize->begin());
      }
      readDomSize = true;
      break;
    case 'o':
      in >> *outputFileName;
      break;
    case 'p':
      tuple = read_uint_tuple(in, ',');
      if (readNumProcsPerDim) {
        errorStream << "-p option should only be passed once" << std::endl;
      } else if (tuple.size() != RANK) {
        errorStream << "Expected num procs per dim of length " << RANK << ", not " << tuple.size();
      } else {
        std::copy(tuple.begin(), tuple.end(), numProcsPerDim->begin());
      }
      readNumProcsPerDim = true;
      break;
    case 'G':
      in >> *numGhostZones;
      break;
    case 'I':
      if (readNumIters) {
        errorStream << "-I option should only be passed once" << std::endl;
      } else {
        in >> iterCount.num_iters;
      }
      readNumIters = true;
      break;
    case 'W':
      if (readNumWarmups) {
        errorStream << "-W option should only be passed once" << std::endl;
      } else {
        in >> iterCount.num_warmups;
      }
      readNumWarmups = true;
      break;
    default:
      errorStream << "Unrecognized option " << optionString << std::endl;
    }
  }
  if (!readNumProcsPerDim) {
    errorStream << "Missing -p option" << std::endl << helpString;
    throw std::runtime_error(errorStream.str());
  }
  if (!readDomSize) {
    errorStream << "Missing -d option" << std::endl << helpString;
    throw std::runtime_error(errorStream.str());
  }
  return iterCount;
}