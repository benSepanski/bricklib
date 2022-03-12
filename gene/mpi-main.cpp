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

#include "brick-stencils.h"
#include "gtensor-stencils.h"
#include "mpi-util.h"
#include "util.h"

/**
 * @brief perform semi-arakawa k-l derivative kernel weak-scaling benchmark
 *
 * Uses array layout
 *
 * @param[out] out output data (has ghost-zones)
 * @param[in] in input data (has ghost-zones)
 * @param[in] coeffs input coefficients (has ghost-zones)
 * @param[in] mpiHandle the mpi handle
 * @param[in] numGhostZones the number of ghost zones to use
 * @param[in] totalExchangeSize the total exchange side, just used for logging output
 * @param[in, out] csvDataRecorder records data for CSV output if rank is zero
 */
void semiArakawaDistributedGTensor(complexArray6D out, const complexArray6D& in,
                                   const realArray6D& coeffs, GeneMPIHandle &mpiHandle,
                                   int numGhostZones, size_t totalExchangeSize,
                                   CSVDataRecorder &csvDataRecorder) {
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
  std::array<unsigned, RANK> ghostDepth{0, 0, 2 * (unsigned) numGhostZones, 2 * (unsigned) numGhostZones, 0, 0};
  auto complexFieldMPIArrayTypesHandle = mpiHandle.buildArrayTypesHandle(inCopy, ghostDepth);

  // get the gtensor kernel
  unsigned numGhostZonesToSkip = 0;
  auto gtensorKernelInToOut =
      buildArakawaGTensorKernel<gt::space::device>(gt_in_dev, gt_out_dev, gt_coeff_dev, numGhostZonesToSkip);
  auto gtensorKernelOutToIn =
      buildArakawaGTensorKernel<gt::space::device>(gt_out_dev, gt_in_dev, gt_coeff_dev, numGhostZonesToSkip);

  // build a function to perform an exchange
  auto exchange = [&](complexArray6D &toExchange, complexArray6D &toExchange_dev) -> void {
#if !defined(CUDA_AWARE) || !defined(USE_TYPES)
    double st = omp_get_wtime();
    toExchange.copyFromDevice(toExchange_dev); ///< Copy device -> host
    movetime += omp_get_wtime() - st;
    mpiHandle.exchangeArray(toExchange, ghostDepth); ///< Exchange on host
    st = omp_get_wtime();
    toExchange.copyToDevice(toExchange_dev); ///< Copy host -> device
    movetime += omp_get_wtime() - st;
#else
    mpiCheckCudaAware();
    mpiHandle.exchangeArray(toExchange_dev, complexFieldMPIArrayTypesHandle);
#endif
  };

  // build a function which computes our stencil
  bool inToOut = true;
  auto gtensorFunc = [&]() -> void {
    // set up timing variables
    float elapsed;
    cudaEvent_t c_0, c_1;
    gpuCheck(cudaEventCreate(&c_0));
    gpuCheck(cudaEventCreate(&c_1));

    // perform exchange
    if(inToOut) {
      exchange(inCopy, in_dev);
    } else {
      exchange(out, out_dev);
    }

    // perform computation
    gpuCheck(cudaEventRecord(c_0));
    for (int i = 0; i < numGhostZones; ++i) {
      if(inToOut) {
        gtensorKernelInToOut();
      } else {
        gtensorKernelOutToIn();
      }
#ifndef NDEBUG
      gpuCheck(cudaPeekAtLastError());
#endif
      inToOut = !inToOut;
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
  timeAndPrintMPIStats(gtensorFunc, totalExchangeSize, (double) in.numElements, numGhostZones, csvDataRecorder);

  // copy output data back to host
  if(inToOut) {
    out.copyFromDevice(in_dev);
  } else {
    out.copyFromDevice(out_dev);
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
 * @param[in] kernelType the kernel type to use
 * @param[in] numGhostZones the number of ghost zones
 * @param[in] totalExchangeSize the total exchange side, just used for logging output
 * @param[out] csvDataRecorder records data for CSV output if rank is zero
 */
void semiArakawaDistributedBrick(complexArray6D out, const complexArray6D &in,
                                 const realArray6D &coeffs, GeneMPILayout &mpiLayout,
                                 BricksArakawaKernelType kernelType, int numGhostZones,
                                 size_t totalExchangeSize, CSVDataRecorder &csvDataRecorder) {
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

  auto arakawaBrickKernel = buildBricksArakawaKernel(fieldLayout, bCoeffArray, kernelType);

  // set up on device
  bInArray.copyToDevice();
  bCoeffArray.copyToDevice();
  FieldBrick_kl bIn_dev = bInArray.viewBricksOnDevice<CommIn_kl>();
  FieldBrick_kl bOut_dev = bOutArray.viewBricksOnDevice<CommIn_kl>();

  // setup function to exchange
#ifndef DECOMP_PAGEUNALIGN
  ExchangeView ev = mpiLayout.buildBrickedArrayMMAPExchangeView(bInArray);
#endif
  auto exchange = [&](BrickedFieldArray &toExchange) -> void {
#ifndef CUDA_AWARE
    {
      double t_a = omp_get_wtime();
      mpiLayout.copyBoundaryFromCuda(toExchange);
      double t_b = omp_get_wtime();
      movetime += t_b - t_a;
#ifdef DECOMP_PAGEUNALIGN
      mpiLayout.exchangeBrickedArray(toExchange);
#else
      ev.exchange();
#endif
      t_a = omp_get_wtime();
      mpiLayout.copyGhostToCuda(toExchange);
      t_b = omp_get_wtime();
      movetime += t_b - t_a;
    }
#else
    mpiCheckCudaAware();
    mpiLayout.exchangeCudaBrickedArray(toExchange);
#endif
  };

  // setup brick function to compute stencil
  bool inToOut = true;
  auto brickFunc = [&]() -> void {
    // set up some cuda events
    float elapsed;
    cudaEvent_t c_0, c_1;
    cudaEventCreate(&c_0);
    cudaEventCreate(&c_1);

    // perform the exchange
    if(inToOut) {
      exchange(bInArray);
    } else {
      exchange(bOutArray);
    }

    // perform the computation
    gpuCheck(cudaEventRecord(c_0));
    for (int i = 0; i < numGhostZones; ++i) {
      if(inToOut) {
        arakawaBrickKernel(bIn_dev, bOut_dev);
      } else {
        arakawaBrickKernel(bOut_dev, bIn_dev);
      }
      inToOut = !inToOut;
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
  timeAndPrintMPIStats(brickFunc, totalExchangeSize, (double)in.numElements, numGhostZones, csvDataRecorder);

  // Copy back
  auto finalOutputBrickedArray = inToOut ? bInArray : bOutArray;
  finalOutputBrickedArray.copyFromDevice();
  finalOutputBrickedArray.storeTo(out);
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
  std::array<int, RANK> numProcsPerDim{}, globalExtent{}, perProcessExtent{};
  std::stringstream input_stream;
  for (int i = 1; i < argc; ++i) {
    input_stream << argv[i] << " ";
  }
  std::string outputFileName;
  bool appendToFile;
  int numGhostZones;
  trial_iter_count iter_count = parse_mpi_args(&perProcessExtent, &numProcsPerDim, &numGhostZones,
                                               &outputFileName, &appendToFile, input_stream);

  NUM_EXCHANGES = iter_count.num_iters;
  NUM_WARMUPS = iter_count.num_warmups;
  for (int i = 0; i < RANK; ++i) {
    globalExtent[i] = perProcessExtent[i] * numProcsPerDim[i];
  }
  std::array<unsigned, RANK> ghostZoneDepth{}; ///< Initializes to all zero
  ghostZoneDepth[2] = 2 * numGhostZones;
  ghostZoneDepth[3] = 2 * numGhostZones;

  // Record setup data in CSV
  CSVDataRecorder dataRecorder;

  // Print information about setup (copied and modified from Tuowen Zhao's args.cpp)
  // https://github.com/CtopCsUtahEdu/bricklib/blob/ef28a307962fe319cd723a589df4ff6fb4a75d18/weak/args.cpp#L133-L144
  int rank, size;
  check_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  check_MPI(MPI_Comm_size(MPI_COMM_WORLD, &size));
  const char dimNames[RANK] = {'i', 'j', 'k', 'l', 'm', 'n'};
  if (rank == 0) {
    int numThreads;
#pragma omp parallel shared(numThreads) default(none)
    numThreads = omp_get_num_threads();
    long page_size = sysconf(_SC_PAGESIZE);
    size_t totElems =
        std::accumulate(globalExtent.begin(), globalExtent.end(), 1, std::multiplies<>());
    int io_col_width = 30;
    dataRecorder.setDefaultValue("PageSize", page_size);
    dataRecorder.setDefaultValue("MPISize", size);
    dataRecorder.setDefaultValue("OpenMPThreads", numThreads);
    std::cout << std::setw(io_col_width) << "Pagesize :" << page_size << "\n"
              << std::setw(io_col_width) << "MPI Processes :" << size << "\n"
              << std::setw(io_col_width) << "OpenMP threads :" << numThreads << "\n"
              << std::setw(io_col_width) << "Domain size (per-process) of :";
    for (int i = 0; i < RANK; ++i) {
      std::cout << std::setw(2) << perProcessExtent[i];
      if (i < RANK - 1) {
        std::cout << " x ";
      }
      dataRecorder.setDefaultValue((std::string) "PerProcessExtent_" + dimNames[i], perProcessExtent[i]);
    }
    dataRecorder.setDefaultValue("TotalNonGhostElements", totElems);
    std::cout << " for a total of " << totElems << " elements " << std::endl
              << std::setw(io_col_width) << "Ghost Zone :";
    size_t totElemsWithGhosts = 1;
    for (int i = 0; i < RANK; ++i) {
      totElemsWithGhosts *= perProcessExtent[i] + 2 * ghostZoneDepth[i];
      std::cout << std::setw(2) << ghostZoneDepth[i];
      if (i < RANK - 1) {
        std::cout << " x ";
      }
      dataRecorder.setDefaultValue((std::string) "GhostSize_" + dimNames[i], ghostZoneDepth[i]);
    }
    dataRecorder.setDefaultValue("TotalElementsWithGhosts", totElemsWithGhosts);
    std::cout << " for a total of " << totElemsWithGhosts << " elements " << std::endl
              << std::setw(io_col_width) << "Array Padding :";
    for (int i = 0; i < RANK; ++i) {
      std::cout << std::setw(2) << PADDING[i];
      if (i < RANK - 1) {
        std::cout << " x ";
      }
    }
    std::cout << std::endl << std::setw(io_col_width) << "MPI Processes :";
    for (int i = 0; i < RANK; ++i) {
      std::cout << std::setw(2) << numProcsPerDim[i];
      if (i < RANK - 1) {
        std::cout << " x ";
      }
      dataRecorder.setDefaultValue((std::string) "MPIGrid_" + dimNames[i], numProcsPerDim[i]);
    }
    std::cout << " for a total of " << size << " processes" << std::endl
              << std::setw(io_col_width) << "Brick Size :";
    for (int i = 0; i < RANK; ++i) {
      std::cout << std::setw(2) << BRICK_DIM[i];
      if (i < RANK - 1)
        std::cout << " x ";
    }
    dataRecorder.setDefaultValue("ItersBetweenExchanges", numGhostZones);
    dataRecorder.setDefaultValue("NumWarmupExchanges", NUM_WARMUPS);
    dataRecorder.setDefaultValue("NumExchanges", NUM_EXCHANGES);
    std::cout << "\n"
              << "Iters Between exchanges : " << numGhostZones << "\n"
              << "Num Warmup Exchanges: " << NUM_WARMUPS << "\n"
              << "Num Exchanges : " << NUM_EXCHANGES << std::endl;
  }
  // Set some values so we can record things about CUDA-AWARE/MPI_TYPES
  bool cudaAware = false;
#ifdef CUDA_AWARE
  cudaAware = true;
#endif
  bool mpiTypes = false;
#ifdef USE_TYPES
  mpiTypes = true;
#endif
  bool gtensorMPITypes = mpiTypes;
  bool gtensorCudaAware = cudaAware & gtensorMPITypes;
  bool bricksCudaAware = cudaAware;

  // build cartesian communicator and setup MEMFD
  bool allowRankReordering = false;
#if defined(OPEN_MPI) && OPEN_MPI
  // FIXME: Either b/c of MPICH or my Perlmutter environment, allowing
  //        rank reordering provokes an error on Perlmutter
  allowRankReordering = true;
#endif
  MPI_Comm cartesianComm = buildCartesianComm(numProcsPerDim, perProcessExtent, allowRankReordering);
  MEMFD::setup_prefix("weak-gene-6d-main", rank);
  // get array/brick extents set up for my MPI process (all include ghost-zones)
  std::array<int, RANK> perProcessExtentWithGZ{}, per_process_extent_with_padding{};
  for (int i = 0; i < RANK; ++i) {
    perProcessExtentWithGZ[i] = perProcessExtent[i] + 2 * (int)ghostZoneDepth[i];
    per_process_extent_with_padding[i] = perProcessExtentWithGZ[i] + 2 * (int)PADDING[i];
    if (perProcessExtent[i] % BRICK_DIM[i] != 0) {
      std::ostringstream error_stream;
      error_stream << "Brick-dimension " << i << " (" << BRICK_DIM[i] << ")"
                   << " does not divide per-process extent " << i << " (" << perProcessExtent[i]
                   << ")";
      throw std::runtime_error(error_stream.str());
    }
    if (ghostZoneDepth[i] % BRICK_DIM[i] != 0) {
      std::ostringstream error_stream;
      error_stream << "Brick-dimension " << i << " (" << BRICK_DIM[i] << ")"
                   << " does not divide ghost-zone " << i << " (" << ghostZoneDepth[i] << ")";
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
  coeffGhostDepth[0] = ghostZoneDepth[0];
  for (unsigned i = 2; i < RANK; ++i) {
    coeffExtent[i] = perProcessExtent[i];
    coeffGhostDepth[i] = ghostZoneDepth[i];
  }
  std::array<unsigned, RANK> coeffExtentWithGZ{};
  for (unsigned i = 0; i < RANK; ++i) {
    coeffExtentWithGZ[i] = coeffExtent[i] + 2 * coeffGhostDepth[i];
  }

  complexArray6D in{complexArray6D::random(perProcessExtentWithGZ)},
      arrayOut{perProcessExtentWithGZ, 0.0}, brickOut{perProcessExtentWithGZ, 0.0};
  unsigned idx = 0;
  for(auto &val : in) {
    val = idx++;
  }

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

  // NOTE: Typically there are Dirichlet BCs in the L-dimension, so communication
  //       does not need to occur on the "left-most" and "right-most" (w.r.t L)
  //       MPI ranks. Actually implementing this would require us to allow
  //       bricks to have separately defined ghost and skin lists, since bricks
  //       currently assumes that the ghosts we receive are the mirror of our skin.
  //
  //       Instead, we approximate this behavior by communicating at every rank in L iff
  //       there are at least 2 ranks with different coordinates along the L axis
  //
  //       The K axis usually has periodic BCs, so a copy has to occur even when there
  //       is only one MPI process along the K axis. Thus, we always communicate over K, and rely
  //       on MPI to perform the copy without doing communication
  std::array<unsigned, RANK> fieldMPIExtent{};
  std::array<unsigned, RANK> fieldMPIGhostZoneDepth{};
  std::array<unsigned, RANK> coeffMPIExtent{};
  std::array<unsigned, RANK> coeffMPIGhostZoneDepth{};
  for(unsigned d = 0; d < RANK; ++d) {
    fieldMPIExtent[d] = perProcessExtent[d];
    fieldMPIGhostZoneDepth[d] = ghostZoneDepth[d];
    coeffMPIExtent[d] = coeffExtent[d];
    coeffMPIGhostZoneDepth[d] = coeffGhostDepth[d];
  }
  auto coeffSkin2D = skin2d;
  auto fieldSkin2D = skin2d;
  if(numProcsPerDim[3] <= 1) {
    auto set_contains_four = [](BitSet set) -> bool {return set.get(4) || set.get(-4);};
    fieldSkin2D.erase(std::remove_if(fieldSkin2D.begin(), fieldSkin2D.end(), set_contains_four), fieldSkin2D.end());
    fieldMPIExtent[3] += 2 * fieldMPIGhostZoneDepth[3];
    fieldMPIGhostZoneDepth[3] = 0;
    coeffMPIExtent[3] += 2 * coeffMPIGhostZoneDepth[3];
    coeffMPIGhostZoneDepth[3] = 0;
  }

  // build brick decomp
  GeneMPIHandle mpiHandle(cartesianComm);
  brick::MPILayout<FieldBrickDimsType, CommIn_kl> mpiLayout(mpiHandle, fieldMPIExtent,
                                                            fieldMPIGhostZoneDepth, fieldSkin2D);

  if (rank == 0) {
    std::cout << "Brick decomposition setup complete. Beginning coefficient setup..." << std::endl;
  }
  // initialize my coefficients to random data, and receive coefficients for ghost-zones
  realArray6D coeffs{realArray6D::random(coeffExtentWithGZ)};
#pragma omp parallel
  for(auto &val : coeffs) {
    // Expected value of each coefficient is 0.5.
    // Each iteration will (on average) grow each input point by 5 * 0.5 = 2.5
    // Divide out by this value so that each iteration will hopefully keep the
    // output array of similar norm to the input array
    val /= 2.5;
  }

  if (rank == 0) {
    std::cout << "Beginning coefficient exchange" << std::endl;
  }
  brick::MPIHandle<RANK, CommIn_kl> coeffMPIHandle(cartesianComm);
#if defined(USE_TYPES)
  auto coeffMPIArrayTypesHandle = coeffMPIHandle.buildArrayTypesHandle(coeffs, coeffGhostDepth);
  coeffMPIHandle.exchangeArray(coeffs, coeffMPIArrayTypesHandle);
#else
  coeffMPIHandle.exchangeArray(coeffs, coeffGhostDepth);
#endif

  // get total exchange size
  size_t totalExchangeSize = 0;
  for (const auto g : mpiLayout.getBrickDecompPtr()->ghost) {
    totalExchangeSize += g.len * FieldBrick_kl::BRICKSIZE * sizeof(bElem);
  }

  if (rank == 0) {
    std::cout << "Coefficient exchange complete. Beginning array computation" << std::endl;
    dataRecorder.setDefaultValue("CUDA_AWARE", gtensorCudaAware);
    dataRecorder.setDefaultValue("MPI_TYPES", gtensorMPITypes);
    dataRecorder.setDefaultValue("Layout", "array");

    for(unsigned d = 0; d < RANK; ++d) {
      dataRecorder.setDefaultValue((std::string) "ArrayPadding_" + dimNames[d], PADDING[d]);
    }
  }
  // run array computation
  semiArakawaDistributedGTensor(arrayOut, in, coeffs, mpiHandle, numGhostZones, totalExchangeSize, dataRecorder);
  if (rank == 0) {
    std::cout << "Array computation complete. Beginning bricks computation" << std::endl;
    dataRecorder.setDefaultValue("CUDA_AWARE", bricksCudaAware);
    dataRecorder.setDefaultValue("MPI_TYPES", false);
    dataRecorder.setDefaultValue("Layout", "bricks");
    for(unsigned d = 0; d < RANK; ++d) {
      dataRecorder.unsetDefaultValue((std::string) "ArrayPadding_" + dimNames[d]);
      dataRecorder.setDefaultValue((std::string) "BrickDim_" + dimNames[d], BRICK_DIM[d]);
      dataRecorder.setDefaultValue((std::string) "BrickVecDim_" + dimNames[d], BRICK_VECTOR_DIM[d]);
    }
  }
  std::array<BricksArakawaKernelType, 1> kernelTypes = {
      //SIMPLE_KLIJMN,
      OPT_IJKLMN,
//      OPT_IKJLMN,
//      OPT_IKLJMN,
//      OPT_KIJLMN,
//      OPT_KILJMN,
//      OPT_KLIJMN
  };
  for(auto kernelType : kernelTypes) {
    if(rank == 0) {
      dataRecorder.setDefaultValue("OptimizedBrickKernel",kernelType != SIMPLE_KLIJMN);
      dataRecorder.setDefaultValue("BrickIterationOrder", toString(kernelType));
      std::cout << "Trying with iteration order " << toString(kernelType) << std::endl;
    }
    semiArakawaDistributedBrick(brickOut, in, coeffs, mpiLayout, kernelType, numGhostZones, totalExchangeSize, dataRecorder);
    checkClose(brickOut, arrayOut, ghostZoneDepth);
    // clear out brick to be sure correct values don't propagate through loop
    brickOut.set(0.0);
  }

  // write data
  if(rank == 0) {
    dataRecorder.writeToFile(outputFileName, appendToFile);
  }

  check_MPI(MPI_Finalize());
  return 0;
}
