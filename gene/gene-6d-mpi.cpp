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
#include <numeric>

#include "gene-6d-stencils.h"
#include "gene6d-gtensor-stencils.h"
#include "mpi-util.h"


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
void semiArakawaDistributedGTensor(complexArray6D out, const complexArray6D in,
                                   const realArray6D coeffs, GeneMPILayout &mpiLayout) {
  complexArray6D inCopy(
      {in.extent[0], in.extent[1], in.extent[2], in.extent[3], in.extent[4], in.extent[5]});
  inCopy.loadFrom(in);
  auto shape6D = gt::shape(in.extent[0] + 2 * complexArray6D::PADDING(0), in.extent[1] + 2 * complexArray6D::PADDING(1),
                           in.extent[2] + 2 * complexArray6D::PADDING(2), in.extent[3] + 2 * complexArray6D::PADDING(3),
                           in.extent[4] + 2 * complexArray6D::PADDING(4), in.extent[5] + 2 * complexArray6D::PADDING(5));
  auto coeffShapeWithStencilAxisFirst =
      gt::shape(coeffs.extent[0] + 2 * realArray6D::PADDING(0), coeffs.extent[1] + 2 * realArray6D::PADDING(1),
                coeffs.extent[2] + 2 * realArray6D::PADDING(2), coeffs.extent[3] + 2 * realArray6D::PADDING(3),
                coeffs.extent[4] + 2 * realArray6D::PADDING(4), coeffs.extent[5] + 2 * realArray6D::PADDING(5));
  auto gt_in = gt::adapt(inCopy.getData().get(), shape6D);
  auto gt_coeffWithStencilAxisFirst =
      gt::adapt(coeffs.getData().get(), coeffShapeWithStencilAxisFirst);
  // reorder axes of coefficients
  auto coeffShape = gt::shape(coeffShapeWithStencilAxisFirst[1], coeffShapeWithStencilAxisFirst[0],
                              coeffShapeWithStencilAxisFirst[2], coeffShapeWithStencilAxisFirst[3],
                              coeffShapeWithStencilAxisFirst[4], coeffShapeWithStencilAxisFirst[5]);
  gt::gtensor<bElem, 6, gt::space::host> gt_coeff(coeffShape);
  {
    using namespace gt::placeholders;
    gt_coeff.view(_all, _all, _all, _all, _all, _all) =
        gt::transpose(gt_coeffWithStencilAxisFirst, gt::shape(1, 0, 2, 3, 4, 5));
    gt::synchronize();
  }

  // copy the in-arrays to device
  auto in_dev = inCopy.allocateOnDevice();
  inCopy.copyToDevice(in_dev);
  auto gt_in_dev = gt::adapt_device((gt::complex<bElem> *)in_dev.getData().get(), shape6D);
  auto gt_coeff_dev = gt::empty_device<bElem>(coeffShape);
  gt::copy(gt_coeff, gt_coeff_dev);
  // declare our out-array
  auto out_dev = out.allocateOnDevice();
  auto gt_out_dev = gt::adapt_device((gt::complex<bElem> *)out_dev.getData().get(), shape6D);
  // set up MPI types for transfer
  auto complexFieldMPIArrayTypesHandle = mpiLayout.buildArrayTypesHandle(inCopy);

  // get the gtensor kernel
  auto gtensorKernel =
      buildArakawaGTensorKernel<gt::space::device>(gt_in_dev, gt_out_dev, gt_coeff_dev);

  // build a function which computes our stencil
  auto gtensorFunc = [&]() -> void {
    float elapsed;
    cudaEvent_t c_0, c_1;
    gpuCheck(cudaEventCreate(&c_0));
    gpuCheck(cudaEventCreate(&c_1));
#if !defined(CUDA_AWARE) || !defined(USE_TYPES)
    double st = omp_get_wtime();
    inCopy.copyFromDevice(in_dev); ///< Copy device -> host
    movetime += omp_get_wtime() - st;
    mpiLayout.exchangeArray(inCopy); ///< Exchange on host
    st = omp_get_wtime();
    inCopy.copyToDevice(in_dev); ///< Copy host -> device
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
  for (unsigned d = 0; d < RANK; ++d) {
    numNonGhostElements *= in.extent[d] - 2 * GHOST_ZONE[d];
  }
  timeAndPrintMPIStats(gtensorFunc, mpiLayout, numNonGhostElements);

  // copy output data back to host
  auto gt_out = gt::adapt(out.getData().get(), shape6D);
  out.copyFromDevice(out_dev);
  // copy data from gtensor back to padded array
  //#pragma omp parallel for collapse(5)
  for (long n = 0; n < out.extent[5]; ++n) {
    for (long m = 0; m < out.extent[4]; ++m) {
      for (long l = 0; l < out.extent[3]; ++l) {
        for (long k = 0; k < out.extent[2]; ++k) {
          for (long j = 0; j < out.extent[1]; ++j) {
            //#pragma omp simd
            for (long i = 0; i < out.extent[0]; ++i) {
              auto o = out(i, j, k, l, m, n);
              auto g = gt_out(i + out.PADDING(0), j + out.PADDING(1), k + out.PADDING(2),
                              l + out.PADDING(3), m + out.PADDING(4), n + out.PADDING(5));
              assert(o == reinterpret_cast<bComplexElem &>(g));
              assert(out(i, j, k, l, m, n) ==
                     reinterpret_cast<bComplexElem &>(
                         gt_out(i + out.PADDING(0), j + out.PADDING(1), k + out.PADDING(2),
                                l + out.PADDING(3), m + out.PADDING(4), n + out.PADDING(5))));
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

  auto arakawaBrickKernel = buildBricksArakawaKernel(fieldLayout, bCoeffArray);

  // set up on device
  bInArray.copyToDevice();
  bCoeffArray.copyToDevice();
  FieldBrick_kl bIn_dev = bInArray.viewBricksOnDevice<CommIn_kl>();
  FieldBrick_kl bOut_dev = bOutArray.viewBricksOnDevice<CommIn_kl>();

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
      arakawaBrickKernel(bIn_dev, bOut_dev);
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
  bool read_dom_size = false, read_num_iters = false, read_num_procs_per_dim = false,
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
  mpiLayout.exchangeArray(in);
  semiArakawaDistributedGTensor(array_out, in, coeffs, mpiLayout);
  if (rank == 0) {
    std::cout << "Array computation complete. Beginning bricks computation" << std::endl;
  }
  semiArakawaDistributedBrick(brick_out, in, coeffs, mpiLayout);

// check for correctness
#ifdef NDEBUG
#pragma omp parallel for collapse(5)
#endif
  for (unsigned n = GHOST_ZONE[5]; n < GHOST_ZONE[5] + perProcessExtent[5]; ++n) {
    for (unsigned m = GHOST_ZONE[4]; m < GHOST_ZONE[4] + perProcessExtent[4]; ++m) {
      for (unsigned l = GHOST_ZONE[3]; l < GHOST_ZONE[3] + perProcessExtent[3]; ++l) {
        for (unsigned k = GHOST_ZONE[2]; k < GHOST_ZONE[2] + perProcessExtent[2]; ++k) {
          for (unsigned j = GHOST_ZONE[1]; j < GHOST_ZONE[1] + perProcessExtent[1]; ++j) {
#ifdef NDEBUG
#pragma omp simd
#endif
            for (unsigned i = GHOST_ZONE[0]; i < GHOST_ZONE[0] + perProcessExtent[0]; ++i) {
              std::complex<bElem> arr = array_out(i, j, k, l, m, n);
              std::complex<bElem> bri = brick_out(i, j, k, l, m, n);
              if (std::abs(arr - bri) >= 1e-7) {
                std::ostringstream error_stream;
                error_stream << "Mismatch at [n,m,l,k,j,i] = [" << n << "," << m << "," << l << ","
                             << k << "," << j << "," << i << "]: "
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
