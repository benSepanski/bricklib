/**
 * @file gene-6d-main.cpp
 * @author Ben Sepanski (ben_sepanski@utexas.edu)
 * @brief
 * @version 0.1
 * @date 2021-08-03
 *
 * @copyright Copyright (c) 2021
 *
 * Copied and modified from main.cu written by Tuowen Zhao
 *
 */

#include "mpi-cuda-util.h"
#include <mpi.h>
#include <numeric>

#include "MPILayout.h"
#include "gene-6d-stencils.h"
#include "gene6d-gtensor-stencils.h"

// useful types
typedef brick::MPILayout<FieldBrickDimsType, CommIn_kl> GeneMPILayout;

// global constants set by CLI
unsigned NUM_EXCHANGES; ///< how many mpi exchanges?
unsigned NUM_WARMUPS; ///< how many warmup iters?

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
 * @return MPI_comm a cartesian communicator built from MPI_COMM_WORLD
 */
MPI_Comm buildCartesianComm(std::array<int, RANK> numProcsPerDim,
                            std::array<int, RANK> perProcessExtent) {
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
  bool allowRankReordering = true;
  MPI_Comm cartesianComm;
  check_MPI(MPI_Cart_create(MPI_COMM_WORLD, RANK, numProcsPerDim.data(), periodic.data(),
                            allowRankReordering, &cartesianComm));
  if (cartesianComm == MPI_COMM_NULL) {
    throw std::runtime_error("Failure in cartesian comm setup");
  }

  // return the communicator
  return cartesianComm;
}

/**
 * @brief times func and prints stats
 *
 * @param func the func to run
 * @param mpiLayout the MPI layout used
 * @param totElems the number of elements
 */
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
  mpi_stats calcSpeedStats = mpi_statistics(totElems / (double) calctime / 1.0e9 * totalNumIters, MPI_COMM_WORLD);
  mpi_stats packTimeStats = mpi_statistics(packtime / totalNumIters, MPI_COMM_WORLD);
  mpi_stats packSpeedStats = mpi_statistics(totalExchangeSize / 1.0e9 / packtime * totalNumIters, MPI_COMM_WORLD);
  mpi_stats mpiCallTimeStats = mpi_statistics(calltime / totalNumIters, MPI_COMM_WORLD);
  mpi_stats mpiWaitTimeStats = mpi_statistics(waittime / totalNumIters, MPI_COMM_WORLD);
  mpi_stats mpiSpeedStats = mpi_statistics(totalExchangeSize / 1.0e9 / (calltime + waittime) * totalNumIters, MPI_COMM_WORLD);
  mpi_stats mpiExchangeSizeStats = mpi_statistics((double)totalExchangeSize * 1.0e-6, MPI_COMM_WORLD);
  double total = calcTimeStats.avg + mpiWaitTimeStats.avg + mpiCallTimeStats.avg + packTimeStats.avg;

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

void validateLaunchConfig(dim3 grid, dim3 block) {
  std::ostringstream gridErrorStream;
  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications__technical-specifications-per-compute-capability
  if (grid.z > 65535) {
    gridErrorStream << "grid.z = " << grid.z << " should be at most 65535" << std::endl;
  }
  if (grid.x > (1U << 31) - 1) {
    gridErrorStream << "grid.x = " << grid.x << " should be at most " << (1U << 31) - 1
                    << std::endl;
  }
  if (block.x > 1024 || block.y > 1024 || block.z > 64) {
    gridErrorStream << "block (.x = " << block.x << ", .y = " << block.y << ", .z = " << block.z
                    << ")"
                    << " is too large in one or more dimensions." << std::endl;
  }
  if (!gridErrorStream.str().empty()) {
    throw std::runtime_error(gridErrorStream.str());
  }
}

/**
 * @brief perform semi-arakawa k-l derivative kernel weak-scaling benchmark
 *
 * Uses array layout
 *
 * @param[out] out output data (has ghost-zones)
 * @param[in] in input data (has ghost-zones)
 * @param[in] coeffs input coefficients (has ghost-zones)
 * @param[in] mpiLayout the mpi layout
 */
void semiArakawaDistributedGTensor(complexArray6D out, const complexArray6D in, realArray6D coeffs,
                                   GeneMPILayout &mpiLayout) {
  brick::Array<bComplexElem, 6> inWithPadding(
      {in.extent[0] + 2 * in.PADDING(0), in.extent[1] + 2 * in.PADDING(1),
       in.extent[2] + 2 * in.PADDING(2), in.extent[3] + 2 * in.PADDING(3),
       in.extent[4] + 2 * in.PADDING(4), in.extent[5] + 2 * in.PADDING(4)},
      in.getData());
  brick::Array<bElem, 6> coeffsWithPadding(
      {coeffs.extent[1] + 2 * coeffs.PADDING(1), coeffs.extent[0] + 2 * coeffs.PADDING(0),
       coeffs.extent[2] + 2 * coeffs.PADDING(2), coeffs.extent[3] + 2 * coeffs.PADDING(3),
       coeffs.extent[4] + 2 * coeffs.PADDING(4), coeffs.extent[5] + 2 * coeffs.PADDING(5)},
      coeffs.getData());
  auto shape6D =
      gt::shape(inWithPadding.extent[0], inWithPadding.extent[1], inWithPadding.extent[2],
                inWithPadding.extent[3], inWithPadding.extent[4], inWithPadding.extent[5]);
  auto coeffShape = gt::shape(coeffsWithPadding.extent[1], coeffsWithPadding.extent[0],
                              coeffsWithPadding.extent[2], coeffsWithPadding.extent[3],
                              coeffsWithPadding.extent[4], coeffsWithPadding.extent[5]);

  // copy in-arrays to gtensor (stripping off the padding)
  auto gt_in = gt::empty<gt::complex<bElem>>(shape6D);
  auto gt_coeff = gt::empty<bElem>(coeffShape);
#pragma omp parallel for collapse(5)
  for (unsigned n = 0; n < inWithPadding.extent[5]; ++n) {
    for (unsigned m = 0; m < inWithPadding.extent[4]; ++m) {
      for (unsigned l = 0; l < inWithPadding.extent[3]; ++l) {
        for (unsigned k = 0; k < inWithPadding.extent[2]; ++k) {
          for (unsigned j = 0; j < inWithPadding.extent[1]; ++j) {
#pragma omp simd
            for (unsigned i = 0; i < inWithPadding.extent[0]; ++i) {
              gt_in(i, j, k, l, m, n) = inWithPadding(i, j, k, l, m, n);
            }
          }
        }
      }
    }
  }

#pragma omp parallel for collapse(5)
  for (unsigned n = 0; n < coeffsWithPadding.extent[5]; ++n) {
    for (unsigned m = 0; m < coeffsWithPadding.extent[4]; ++m) {
      for (unsigned l = 0; l < coeffsWithPadding.extent[3]; ++l) {
        for (unsigned k = 0; k < coeffsWithPadding.extent[2]; ++k) {
          for (unsigned j = 0; j < coeffsWithPadding.extent[1]; ++j) {
#pragma omp simd
            for (unsigned i = 0; i < coeffsWithPadding.extent[0]; ++i) {
              gt_coeff(j, i, k, l, m, n) = coeffsWithPadding(i, j, k, l, m, n);
            }
          }
        }
      }
    }
  }

  // copy the in-arrays to device
  auto in_dev = in.allocateOnDevice();
  gt::gtensor<gt::complex<bElem>, 6, gt::space::device>
      gt_in_dev = gt::adapt_device((gt::complex<bElem>*) in_dev.getData().get(), shape6D);
  gt::gtensor<bElem, 6, gt::space::device> gt_coeff_dev = gt::empty_device<bElem>(coeffShape);
  gt::copy(gt_in, gt_in_dev);
  gt::copy(gt_coeff, gt_coeff_dev);
  // declare our out-array
  auto out_dev = out.allocateOnDevice();
  gt::gtensor<gt::complex<bElem>, 6, gt::space::device>
      gt_out_dev = gt::adapt_device((gt::complex<bElem>*) out_dev.getData().get(), shape6D);
  // set up MPI types for transfer
  auto complexFieldMPIArrayTypesHandle = mpiLayout.buildArrayTypesHandle(in);

  // get the gtensor kernel
  auto gtensorKernel = buildArakawaGTensorKernel<gt::space::device>(gt_in_dev, gt_out_dev, gt_coeff_dev);

  // build a function which computes our stencil
  auto gtensorFunc = [&]() -> void {
    float elapsed;
    cudaEvent_t c_0, c_1;
    gpuCheck(cudaEventCreate(&c_0));
    gpuCheck(cudaEventCreate(&c_1));
#if !defined(CUDA_AWARE) || !defined(USE_TYPES)
    // Copy everything back from device
    double st = omp_get_wtime();
    gt::copy(gt_in_dev, gt_in);
    movetime += omp_get_wtime() - st;
    mpiLayout.exchangeArray(in);
    st = omp_get_wtime();
    in.copyToDevice(in_dev);
    movetime += omp_get_wtime() - st;
#else
    mpiCheckCudaAware();
    mpiLayout.exchangeArray(in_dev, complexFieldMPIArrayTypesHandle);
#endif
    gpuCheck(cudaEventRecord(c_0));
    for (int i = 0; i < NUM_GHOST_ZONES; ++i) {
      gtensorKernel();
#ifndef NDEBUG
      gpuCheck(cudaPeekAtLastError());
#endif
      if (i + 1 < NUM_GHOST_ZONES) {
        std::cout << "Swapping!" << std::endl;
        // TODO:
//        std::swap(outWithPaddingPtr_dev, inWithPaddingPtr_dev);
//        std::swap(inPtr_dev, outPtr_dev);
//        std::swap(inPtr, outPtr);
      }
    }
    gpuCheck(cudaEventRecord(c_1));
    gpuCheck(cudaEventSynchronize(c_1));
    gpuCheck(cudaEventElapsedTime(&elapsed, c_0, c_1));
    calctime += elapsed / 1000.0;
  };

  int rank;
  check_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  if (rank == 0)
    std::cout << "gtensor MPI decomp" << std::endl;
  double numNonGhostElements = 1.0;
  for(unsigned d = 0; d < RANK; ++d) {
    numNonGhostElements *= in.extent[d] - 2 * GHOST_ZONE[d];
  }
  timeAndPrintMPIStats(gtensorFunc, mpiLayout, numNonGhostElements);

  // copy output data back to host
  auto gt_out = gt::empty<gt::complex<bElem>>(shape6D);
  gt::copy(gt_out_dev, gt_out);
  // copy data from gtensor back to padded array
#pragma omp parallel for collapse(5)
  for (long n = 0; n < out.extent[5]; ++n) {
    for (long m = 0; m < out.extent[4]; ++m) {
      for (long l = 0; l < out.extent[3]; ++l) {
        for (long k = 0; k < out.extent[2]; ++k) {
          for (long j = 0; j < out.extent[1]; ++j) {
#pragma omp simd
            for (long i = 0; i < out.extent[0]; ++i) {
              out(i, j, k, l, m, n) = reinterpret_cast<bComplexElem &>(
                  gt_out(i + out.PADDING(0), j + out.PADDING(1), k + out.PADDING(2),
                         l + out.PADDING(3), m + out.PADDING(4), n + out.PADDING(5)));
            }
          }
        }
      }
    }
  }
}

/**
 * @brief perform semi-arakawa k-l derivative kernel weak-scaling benchmark
 *
 * Uses bricks layout
 *
 * @param[out] out_ptr output data (has ghost-zones and padding)
 * @param[in] in_ptr input data (has ghost-zones and padding)
 * @param[in] coeffs input coefficients (has ghost-zones but no padding)
 * @param[in] mpiLayout the mpi-layout
 */
void semiArakawaDistributedBrick(complexArray6D out, const complexArray6D in, realArray6D coeffs,
                                 GeneMPILayout &mpiLayout) {
  // set up brick-info and storage on host
  brick::BrickLayout<RANK> fieldLayout = mpiLayout.getBrickLayout();
#ifdef DECOMP_PAGEUNALIGN
  BrickedFieldArray bInArray(fieldLayout);
  BrickedFieldArray bOutArray(fieldLayout);
#else
  // load with mmap
  BrickedFieldArray bInArray(fieldLayout, nullptr);
  BrickedFieldArray bOutArray(fieldLayout, nullptr);
#endif
  // load in input
  bInArray.loadFrom(in);
  std::array<unsigned, RANK> coeffBrickGridExtent{};
  for (unsigned d = 0; d < RANK; ++d) {
    assert(coeffs.extent[d] % ARAKAWA_COEFF_BRICK_DIM[d] == 0);
    coeffBrickGridExtent[d] = coeffs.extent[d] / ARAKAWA_COEFF_BRICK_DIM[d];
  }
  brick::BrickLayout<RANK> coeffLayout(coeffBrickGridExtent);
  BrickedArakawaCoeffArray bCoeffArray(coeffLayout);
  bCoeffArray.loadFrom(coeffs);

  // set up on device
  bInArray.copyToDevice();
  bCoeffArray.copyToDevice();
  FieldBrick_kl bIn_dev = bInArray.viewBricksOnDevice<CommIn_kl>();
  FieldBrick_kl bOut_dev = bOutArray.viewBricksOnDevice<CommIn_kl>();
  ArakawaCoeffBrick bCoeff_dev = bCoeffArray.viewBricksOnDevice<NoComm>();
  auto fieldIndexInStorage_dev = fieldLayout.indexInStorage.allocateOnDevice();
  fieldLayout.indexInStorage.copyToDevice(fieldIndexInStorage_dev);
  auto coeffIndexInStorage_dev = coeffLayout.indexInStorage.allocateOnDevice();
  coeffLayout.indexInStorage.copyToDevice(coeffIndexInStorage_dev);

  // set up grid
  const unsigned *const brickExtentWithGZ = fieldLayout.indexInStorage.extent;
  dim3 cuda_grid_size(brickExtentWithGZ[2], brickExtentWithGZ[3],
                      brickExtentWithGZ[0] * brickExtentWithGZ[1] * brickExtentWithGZ[4] *
                          brickExtentWithGZ[5]),
      cuda_block_size(BRICK_DIM[0], BRICK_DIM[1],
                      FieldBrick_kl::BRICKLEN / BRICK_DIM[0] / BRICK_DIM[1]);
  validateLaunchConfig(cuda_grid_size, cuda_block_size);

#ifndef DECOMP_PAGEUNALIGN
  ExchangeView ev = mpiLayout.buildBrickedArrayMMAPExchangeView(bInArray);
#endif
  // setup brick function to compute stencil
  auto brickFunc = [&]() -> void {
    float elapsed;
    cudaEvent_t c_0, c_1;
    cudaEventCreate(&c_0);
    cudaEventCreate(&c_1);
#ifndef CUDA_AWARE
    {
      double t_a = omp_get_wtime();
      mpiLayout.copyBoundaryFromCuda(bInArray);
      double t_b = omp_get_wtime();
      movetime += t_b - t_a;
#ifdef DECOMP_PAGEUNALIGN
      mpiLayout.exchangeBrickedArray(bInArray);
#else
      ev.exchange();
#endif
      t_a = omp_get_wtime();
      mpiLayout.copyGhostToCuda(bInArray);
      t_b = omp_get_wtime();
      movetime += t_b - t_a;
    }
#else
    mpiCheckCudaAware();
    mpiLayout.exchangeCudaBrickedArray(bInArray);
#endif
    gpuCheck(cudaEventRecord(c_0));
    for (int i = 0; i < NUM_GHOST_ZONES; ++i) {
      semiArakawaBrickKernel<<<cuda_grid_size, cuda_block_size>>>(
          fieldIndexInStorage_dev, coeffIndexInStorage_dev, bIn_dev, bOut_dev, bCoeff_dev);
#ifndef NDEBUG
      gpuCheck(cudaPeekAtLastError());
#endif
      if (i + 1 < NUM_GHOST_ZONES) {
        std::cout << "Swapping!" << std::endl;
        std::swap(bOut_dev, bIn_dev);
      }
    }
    gpuCheck(cudaEventRecord(c_1));
    gpuCheck(cudaEventSynchronize(c_1));
    gpuCheck(cudaEventElapsedTime(&elapsed, c_0, c_1));
    calctime += elapsed / 1000.0;
  };

  // time function
  int rank;
  check_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  if (rank == 0)
    std::cout << "brick MPI decomp" << std::endl;
  timeAndPrintMPIStats(brickFunc, mpiLayout, (double)in.numElements);

  // Copy back
  bOutArray.copyFromDevice();
  bOutArray.storeTo(out);
}

/**
 * @brief Reads a tuple of unsigneds delimited by delim
 *
 * @param in the input stream to read from
 * @param delim the delimiter between unsigneds
 * @return std::vector<unsigned> of the values read in
 */
std::vector<unsigned> read_uint_tuple(std::istream &in, char delim = ',') {
  std::vector<unsigned> tuple;
  unsigned value;
  do {
    if (in.peek() == delim)
      in.get();
    in >> value;
    tuple.push_back(value);
  } while (in.peek() == delim);
  return tuple;
}

struct trial_iter_count {
  int num_warmups, num_iters;
};

/**
 * @brief parse args
 * @param[out] per_process_domain_size the extent in each dimension of the domain
 * @param[out] num_procs_per_dim the number of processes per dimension
 * @param[in] in input stream to read from
 *
 * @return the number of iterations, with default 100
 */
trial_iter_count parse_args(std::array<int, RANK> *per_process_domain_size,
                            std::array<int, RANK> *num_procs_per_dim, std::istream &in) {
  std::string option_string;
  trial_iter_count iter_count;
  iter_count.num_iters = 100;
  iter_count.num_warmups = 5;
  std::vector<unsigned> tuple;
  bool read_dom_size = false,
       read_num_iters = false,
       read_num_procs_per_dim = false,
       read_num_warmups = false;
  std::string help_string = "Program options\n"
                            "  -h: show help (this message)\n"
                            "  Domain size,  in array order contiguous first\n"
                            "  -d: comma separated Int[6], per-process domain size\n"
                            "  Num Tasks per dimension, in array order contiguous first\n"
                            "  -p: comma separated Int[6], num process per dimension"
                            "  Benchmark control:\n"
                            "  -I: number of iterations, default 100 \n"
                            "  -W: number of warmup iterations, default 5\n"
                            "Example usage:\n"
                            "  weak/gene6d -d 70,16,24,48,32,2 -p 1,1,3,1,2,1\n";
  std::ostringstream error_stream;
  while (in >> option_string) {
    if (option_string[0] != '-' || option_string.size() != 2) {
      error_stream << "Unrecognized option " << option_string << std::endl;
    }
    if (error_stream.str().size() != 0) {
      error_stream << help_string;
      throw std::runtime_error(error_stream.str());
    }
    switch (option_string[1]) {
    case 'd':
      tuple = read_uint_tuple(in, ',');
      if (read_dom_size) {
        error_stream << "-d option should only be passed once" << std::endl;
      } else if (tuple.size() != RANK) {
        error_stream << "Expected extent of length " << RANK << ", not " << tuple.size();
      } else {
        std::copy(tuple.begin(), tuple.end(), per_process_domain_size->begin());
      }
      read_dom_size = true;
      break;
    case 'p':
      tuple = read_uint_tuple(in, ',');
      if (read_num_procs_per_dim) {
        error_stream << "-p option should only be passed once" << std::endl;
      } else if (tuple.size() != RANK) {
        error_stream << "Expected num procs per dim of length " << RANK << ", not " << tuple.size();
      } else {
        std::copy(tuple.begin(), tuple.end(), num_procs_per_dim->begin());
      }
      read_num_procs_per_dim = true;
      break;
    case 'I':
      if (read_num_iters) {
        error_stream << "-I option should only be passed once" << std::endl;
      } else {
        in >> iter_count.num_iters;
      }
      read_num_iters = true;
      break;
    case 'W':
      if (read_num_warmups) {
        error_stream << "-W option should only be passed once" << std::endl;
      } else {
        in >> iter_count.num_warmups;
      }
      read_num_warmups = true;
      break;
    default:
      error_stream << "Unrecognized option " << option_string << std::endl;
    }
  }
  if (!read_num_procs_per_dim) {
    error_stream << "Missing -p option" << std::endl << help_string;
    throw std::runtime_error(error_stream.str());
  }
  if (!read_dom_size) {
    error_stream << "Missing -d option" << std::endl << help_string;
    throw std::runtime_error(error_stream.str());
  }
  return iter_count;
}

/**
 * @brief Run weak-scaling gene6d benchmark
 */
int main(int argc, char **argv) {
  int provided;
  // setup MPI environment
  check_MPI(MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided));
  if (provided != MPI_THREAD_SERIALIZED) {
    check_MPI(MPI_Finalize());
    return 1;
  }
  // TODO: get per-process extent and number of processors per-dimension from
  //       cmdline
  std::array<int, RANK> numProcsPerDim{}, globalExtent{}, perProcessExtent{};
  std::stringstream input_stream;
  for (int i = 1; i < argc; ++i) {
    input_stream << argv[i] << " ";
  }
  trial_iter_count iter_count = parse_args(&perProcessExtent, &numProcsPerDim, input_stream);
  NUM_EXCHANGES = iter_count.num_iters;
  NUM_WARMUPS = iter_count.num_warmups;
  for (int i = 0; i < RANK; ++i) {
    globalExtent[i] = perProcessExtent[i] * numProcsPerDim[i];
  }

  // Print information about setup (copied from Tuowen Zhao's args.cpp)
  // https://github.com/CtopCsUtahEdu/bricklib/blob/ef28a307962fe319cd723a589df4ff6fb4a75d18/weak/args.cpp#L133-L144
  int rank, size;
  check_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  check_MPI(MPI_Comm_size(MPI_COMM_WORLD, &size));
  if (rank == 0) {
    int numthreads;
#pragma omp parallel shared(numthreads) default(none)
    numthreads = omp_get_num_threads();
    long page_size = sysconf(_SC_PAGESIZE);
    size_t totElems =
        std::accumulate(globalExtent.begin(), globalExtent.end(), 1, std::multiplies<size_t>());
    int io_col_width = 30;
    std::cout << std::setw(io_col_width) << "Pagesize :" << page_size << "\n"
              << std::setw(io_col_width) << "MPI Processes :" << size << "\n"
              << std::setw(io_col_width) << "OpenMP threads :" << numthreads << "\n"
              << std::setw(io_col_width) << "Domain size (per-process) of :";
    for (int i = 0; i < RANK; ++i) {
      std::cout << std::setw(2) << perProcessExtent[i];
      if (i < RANK - 1)
        std::cout << " x ";
    }
    std::cout << " for a total of " << totElems << " elements " << std::endl
              << std::setw(io_col_width) << "Ghost Zone :";
    size_t totElemsWithGhosts = 1;
    for (int i = 0; i < RANK; ++i) {
      totElemsWithGhosts *= perProcessExtent[i] + 2 * GHOST_ZONE[i];
      std::cout << std::setw(2) << GHOST_ZONE[i];
      if (i < RANK - 1)
        std::cout << " x ";
    }
    std::cout << " for a total of " << totElemsWithGhosts << " elements " << std::endl
              << std::setw(io_col_width) << "Array Padding :";
    for (int i = 0; i < RANK; ++i) {
      std::cout << std::setw(2) << PADDING[i];
      if (i < RANK - 1)
        std::cout << " x ";
    }
    std::cout << std::endl << std::setw(io_col_width) << "MPI Processes :";
    for (int i = 0; i < RANK; ++i) {
      std::cout << std::setw(2) << numProcsPerDim[i];
      if (i < RANK - 1)
        std::cout << " x ";
    }
    std::cout << " for a total of " << size << " processes" << std::endl
              << std::setw(io_col_width) << "Brick Size :";
    for (int i = 0; i < RANK; ++i) {
      std::cout << std::setw(2) << BRICK_DIM[i];
      if (i < RANK - 1)
        std::cout << " x ";
    }
    std::cout << "\n"
              << "Iters Between exchanges : " << NUM_GHOST_ZONES << "\n"
              << "Num Warmup Exchanges: " << NUM_WARMUPS << "\n"
              << "Num Exchanges : " << NUM_EXCHANGES << std::endl;
  }

  // build cartesian communicator and setup MEMFD
  MPI_Comm cartesianComm = buildCartesianComm(numProcsPerDim, perProcessExtent);
  MEMFD::setup_prefix("weak-gene-6d-main", rank);
  // get array/brick extents set up for my MPI process (all include ghost-zones)
  std::array<int, RANK> perProcessExtentWithGZ{}, per_process_extent_with_padding{};
  for (int i = 0; i < RANK; ++i) {
    perProcessExtentWithGZ[i] = perProcessExtent[i] + 2 * (int)GHOST_ZONE[i];
    per_process_extent_with_padding[i] = perProcessExtentWithGZ[i] + 2 * (int)PADDING[i];
    if (perProcessExtent[i] % BRICK_DIM[i] != 0) {
      std::ostringstream error_stream;
      error_stream << "Brick-dimension " << i << " (" << BRICK_DIM[i] << ")"
                   << " does not divide per-process extent " << i << " (" << perProcessExtent[i]
                   << ")";
      throw std::runtime_error(error_stream.str());
    }
    if (GHOST_ZONE[i] % BRICK_DIM[i] != 0) {
      std::ostringstream error_stream;
      error_stream << "Brick-dimension " << i << " (" << BRICK_DIM[i] << ")"
                   << " does not divide ghost-zone " << i << " (" << GHOST_ZONE[i] << ")";
      throw std::runtime_error(error_stream.str());
    }
  }
  if (rank == 0) {
    std::cout << "Building input arrays..." << std::endl;
  }

  // set up coeffs
  std::array<unsigned, RANK> coeffExtent{}, coeffGhostDepth{};
  coeffExtent[0] = ARAKAWA_STENCIL_SIZE;
  coeffGhostDepth[0] = 0;
  coeffExtent[1] = perProcessExtent[0];
  coeffGhostDepth[0] = GHOST_ZONE[0];
  for (unsigned i = 2; i < RANK; ++i) {
    coeffExtent[i] = perProcessExtent[i];
    coeffGhostDepth[i] = GHOST_ZONE[i];
  }
  std::array<unsigned, RANK> coeffExtentWithGZ{};
  for (unsigned i = 0; i < RANK; ++i) {
    coeffExtentWithGZ[i] = coeffExtent[i] + 2 * coeffGhostDepth[i];
  }

  complexArray6D in{complexArray6D::random(perProcessExtentWithGZ)},
      array_out{perProcessExtentWithGZ, 0.0}, brick_out{perProcessExtentWithGZ, 0.0};

  if (rank == 0) {
    std::cout << "Input arrays built. Setting up brick decomposition..." << std::endl;
  }
  // build 2d skin from 3d skin by removing all faces with a 3,
  // and replacing 1 -> 3, 2 -> 4
  std::vector<BitSet> skin2d = skin3d_good;
  auto set_contains_three = [](BitSet set) -> bool { return set.get(3) || set.get(-3); };
  skin2d.erase(std::remove_if(skin2d.begin(), skin2d.end(), set_contains_three), skin2d.end());
  for (BitSet &bitset : skin2d) {
    if (bitset.get(1)) {
      bitset.flip(1);
      bitset.flip(3);
    }
    if (bitset.get(-1)) {
      bitset.flip(-1);
      bitset.flip(-3);
    }
    if (bitset.get(2)) {
      bitset.flip(2);
      bitset.flip(4);
    }
    if (bitset.get(-2)) {
      bitset.flip(-2);
      bitset.flip(-4);
    }
  }
  // build brick decomp
  brick::MPILayout<FieldBrickDimsType, CommIn_kl> mpiLayout(cartesianComm, perProcessExtent,
                                                            GHOST_ZONE, skin2d);

  if (rank == 0) {
    std::cout << "Brick decomposition setup complete. Beginning coefficient setup..." << std::endl;
  }
  // initialize my coefficients to random data, and receive coefficients for ghost-zones
  realArray6D coeffs{realArray6D::random(coeffExtentWithGZ)};
  if (rank == 0) {
    std::cout << "Beginning coefficient exchange" << std::endl;
  }
  brick::MPILayout<ArakawaCoeffBrickDimsType, CommIn_kl> coeffMpiLayout(cartesianComm, coeffExtent,
                                                                        coeffGhostDepth, skin2d);
#if defined(USE_TYPES)
  auto coeffMPIArrayTypesHandle = coeffMpiLayout.buildArrayTypesHandle(coeffs);
  coeffMpiLayout.exchangeArray(coeffs, coeffMPIArrayTypesHandle);
#else
  coeffMpiLayout.exchangeArray(coeffs);
#endif

  if (rank == 0) {
    std::cout << "Coefficient exchange complete. Beginning array computation" << std::endl;
  }
  // run array computation
  semiArakawaDistributedGTensor(array_out, in, coeffs, mpiLayout);
  if (rank == 0) {
    std::cout << "Array computation complete. Beginning bricks computation" << std::endl;
  }
  semiArakawaDistributedBrick(brick_out, in, coeffs, mpiLayout);

// check for correctness
#pragma omp parallel for collapse(5)
  for (unsigned n = GHOST_ZONE[5]; n < GHOST_ZONE[5] + perProcessExtent[5]; ++n) {
    for (unsigned m = GHOST_ZONE[4]; m < GHOST_ZONE[4] + perProcessExtent[4]; ++m) {
      for (unsigned l = GHOST_ZONE[3]; l < GHOST_ZONE[3] + perProcessExtent[3]; ++l) {
        for (unsigned k = GHOST_ZONE[2]; k < GHOST_ZONE[2] + perProcessExtent[2]; ++k) {
          for (unsigned j = GHOST_ZONE[1]; j < GHOST_ZONE[1] + perProcessExtent[1]; ++j) {
            // #pragma omp simd
            for (unsigned i = GHOST_ZONE[0]; i < GHOST_ZONE[0] + perProcessExtent[0]; ++i) {
              std::complex<bElem> arr = array_out(i, j, k, l, m, n);
              std::complex<bElem> bri = brick_out(i, j, k, l, m, n);
              if (std::abs(arr - bri) >= 1e-7) {
                std::ostringstream error_stream;
                error_stream << "Mismatch at [" << n << "," << m << "," << l << "," << k << "," << j
                             << "," << i << "]: "
                             << "(array) " << arr << " != " << bri << "(brick)" << std::endl;
                throw std::runtime_error(error_stream.str());
              }
            }
          }
        }
      }
    }
  }

  check_MPI(MPI_Finalize());
  return 0;
}
