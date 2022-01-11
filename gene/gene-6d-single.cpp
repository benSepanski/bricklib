/**
 * @file gene-6d-single.cpp
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

#include <numeric>

#include "gene-6d-gtensor-stencils.h"
#include "gene-6d-stencils.h"
#include "single-util.h"
#include "util.h"

/**
 * @brief perform semi-arakawa k-l derivative kernel
 *
 * Uses array layout
 *
 * @param[out] out output data (has ghost-zones)
 * @param[in] in input data (has ghost-zones)
 * @param[in] coeffs input coefficients (has ghost-zones)
 * @param[out] csvDataRecorder records data for CSV output if rank is zero
 */
void semiArakawaGTensor(complexArray6D out, const complexArray6D &in, const realArray6D &coeffs,
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

  // get the gtensor kernel
  unsigned numGhostZonesToSkip = 1;
  auto gtensorKernel =
      buildArakawaGTensorKernel<gt::space::device>(gt_in_dev, gt_out_dev, gt_coeff_dev, numGhostZonesToSkip);

  // build a function which computes our stencil
  auto gtensorFunc = [&]() -> void {
    gtensorKernel();
#ifndef NDEBUG
    gpuCheck(cudaPeekAtLastError());
#endif
  };
  size_t numNonGhostElements = in.numElements / (in.extent[2] * in.extent[3])
                               * (in.extent[2] - 4) * (in.extent[3] - 4);
  timeAndPrintStats(gtensorFunc, numNonGhostElements, csvDataRecorder);

  // copy output data back to host
  out.copyFromDevice(out_dev);
}

/**
 * Compute i-j derivative kernel using bricks layout
 *
 * @param out output array
 * @param in the input array
 * @param p1 coefficient to scale i-derivative
 * @param p2 coefficient to scale j-derivative
 * @param ikj 2*pi*fourier mode
 * @param i_deriv_coeff the coefficients
 * @param dataRecorder the recorder to hold data in
 */
void ijderivGTensor(complexArray6D out, const complexArray6D &in, const complexArray5D &p1,
                    const complexArray5D &p2, const complexArray1D_J &ikj, bElem i_deriv_coeff[5],
                    CSVDataRecorder &dataRecorder) {
  auto out_dev = out.allocateOnDevice();
  out.copyToDevice(out_dev);
  auto in_dev = in.allocateOnDevice();
  in.copyToDevice(in_dev);
  auto p1_dev = p1.allocateOnDevice();
  p1.copyToDevice(p1_dev);
  auto p2_dev = p2.allocateOnDevice();
  p2.copyToDevice(p2_dev);
  auto ikj_dev = ikj.allocateOnDevice();
  ikj.copyToDevice(ikj_dev);

  auto shape6D = gt::shape(out.extent[0], out.extent[1], out.extent[2], out.extent[3], out.extent[4], out.extent[5]);
  auto gt_out = gt::adapt_device(out_dev.getData().get(), shape6D);
  auto gt_in = gt::adapt_device(in_dev.getData().get(), shape6D);
  auto shape5D = gt::shape(p1.extent[0], p1.extent[1], p1.extent[2], p1.extent[3], p1.extent[4]);
  auto gt_p1 = gt::adapt_device(p1_dev.getData().get(), shape5D);
  auto gt_p2 = gt::adapt_device(p2_dev.getData().get(), shape5D);
  auto gt_ikj = gt::adapt_device(ikj_dev.getData().get(), gt::shape(ikj.extent[0]));

  auto gtensorKernel = [&]() -> void {
    computeIJDerivGTensor<gt::space::device>(gt_out, gt_in, gt_p1, gt_p2, gt_ikj, i_deriv_coeff);
  };

  size_t numNonGhostElements = in.numElements / in.extent[0] * (in.extent[0] - 4);
  timeAndPrintStats(gtensorKernel, numNonGhostElements, dataRecorder);

  // copy result to out
  out.copyFromDevice(out_dev);
}

/**
 * Compute i-j derivative kernel using bricks layout
 *
 * @param out output array
 * @param in the input array
 * @param p1 coefficient to scale i-derivative
 * @param p2 coefficient to scale j-derivative
 * @param ikj 2*pi*fourier mode
 * @param i_deriv_coeff the coefficients
 * @param dataRecorder the recorder to hold data in
 */
void ijderivBrick(complexArray6D out, const complexArray6D &in, const complexArray5D &p1,
                  const complexArray5D &p2, const complexArray1D_J &ikj, bElem ij_deriv_coeffs[5],
                  CSVDataRecorder &dataRecorder) {
  // set up brick-info and storage on host
  std::array<unsigned, RANK> fieldBrickGridExtent{};
  for(unsigned d = 0; d < RANK; ++d) {
    fieldBrickGridExtent[d] = in.extent[d] / BRICK_DIM[d];
  }
  std::array<unsigned, RANK-1> coeffBrickGridExtent{};
  for(unsigned d = 0; d < RANK - 1; ++d) {
    coeffBrickGridExtent[d] = p1.extent[d] / PCOEFF_BRICK_DIM[d];
  }
  // build brick layouts
  brick::BrickLayout<RANK> fieldLayout(fieldBrickGridExtent);
  brick::BrickLayout<RANK-1> coeffLayout(coeffBrickGridExtent);
  // build bricked arrays
  BrickedFieldArray bInArray(fieldLayout);
  BrickedFieldArray bOutArray(fieldLayout);
  BrickedPCoeffArray bP1Array(coeffLayout);
  BrickedPCoeffArray bP2Array(coeffLayout);

  // Move to device
  bInArray.copyToDevice();
  bP1Array.copyToDevice();
  bP2Array.copyToDevice();

  // view bricks on the device
  FieldBrick_i bIn_dev = bInArray.viewBricksOnDevice<CommIn_i>();
  FieldBrick_i bOut_dev = bOutArray.viewBricksOnDevice<CommIn_i>();
  PreCoeffBrick bP1_dev = bP1Array.viewBricksOnDevice<NoComm>();
  PreCoeffBrick bP2_dev  = bP2Array.viewBricksOnDevice<NoComm>();

  // build kernel
  auto brickKernel = [&]() -> void {
    // TODO
  };

  // time kernel
  timeAndPrintStats(brickKernel, in.numElements, dataRecorder);

  // copy back from device
  bOutArray.copyFromDevice();
  bOutArray.storeTo(bOut);
}

/**
 * @brief perform semi-arakawa k-l derivative kernel
 *
 * Uses bricks layout
 *
 * @param[out] out_ptr output data (has ghost-zones and padding)
 * @param[in] in_ptr input data (has ghost-zones and padding)
 * @param[in] coeffs input coefficients (has ghost-zones but no padding)
 * @param[in] kernelType the kernel type to use
 * @param[out] csvDataRecorder records data for CSV output if rank is zero
 */
void semiArakawaBrick(complexArray6D out, const complexArray6D& in, const realArray6D& coeffs,
                      BricksArakawaKernelType kernelType, CSVDataRecorder &csvDataRecorder) {
  // set up brick-info and storage on host
  std::array<unsigned, RANK> gridExtent{};
  for(unsigned d = 0; d < RANK; ++d) {
    gridExtent[d] = in.extent[d] / BRICK_DIM[d];
  }
  // build brick decomp
  brick::BrickLayout<RANK> brickLayout(gridExtent);
#ifdef DECOMP_PAGEUNALIGN
  BrickedFieldArray bInArray(fieldLayout);
  BrickedFieldArray bOutArray(fieldLayout);
#else
  // load with mmap
  BrickedFieldArray bInArray(brickLayout, nullptr);
  BrickedFieldArray bOutArray(brickLayout, nullptr);
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

  auto arakawaBrickKernel = buildBricksArakawaKernel(brickLayout, bCoeffArray, kernelType);

  // set up on device
  bInArray.copyToDevice();
  bCoeffArray.copyToDevice();
  FieldBrick_kl bIn_dev = bInArray.viewBricksOnDevice<CommIn_kl>();
  FieldBrick_kl bOut_dev = bOutArray.viewBricksOnDevice<CommIn_kl>();

  // setup brick function to compute stencil
  auto brickFunc = [&]() -> void {
    arakawaBrickKernel(bIn_dev, bOut_dev);
  };

  // time function
  timeAndPrintStats(brickFunc, in.numElements, csvDataRecorder);

  // Copy back
  bOutArray.copyFromDevice();
  bOutArray.storeTo(out);
}

/**
 * Run bricks and gtensor arakawa benchmarks
 * @param extent the extent of the arrays to use
 * @param dataRecorder where to record the data
 */
void runArakawa(std::array<unsigned, RANK> extent, CSVDataRecorder &dataRecorder) {
  std::cout << "Building input arrays..." << std::endl;

  // set up coeffs
  std::array<unsigned, RANK> coeffExtent{};
  coeffExtent[0] = ARAKAWA_STENCIL_SIZE;
  coeffExtent[1] = extent[0];
  for (unsigned i = 2; i < RANK; ++i) {
    coeffExtent[i] = extent[i];
  }

  complexArray6D in{complexArray6D::random(extent)},
      arrayOut{extent, 0.0}, brickOut{extent, 0.0};

  std::cout << "Brick decomposition setup complete. Beginning coefficient setup..." << std::endl;
  // initialize my coefficients to random data, and receive coefficients for ghost-zones
  realArray6D coeffs{realArray6D::random(coeffExtent)};

  std::cout << "Coefficient setup complete. Beginning Arakawa array computation" << std::endl;
  dataRecorder.setDefaultValue("Layout", "array");
  dataRecorder.setDefaultValue("Kernel", "arakawa");

  const char dimNames[RANK] = {'i', 'j', 'k', 'l', 'm', 'n'};
  for(unsigned d = 0; d < RANK; ++d) {
    dataRecorder.setDefaultValue((std::string) "ArrayPadding_" + dimNames[d], PADDING[d]);
  }
  // run array computation
  semiArakawaGTensor(arrayOut, in, coeffs, dataRecorder);
  std::cout << "Array computation complete. Beginning Arakawa bricks computation" << std::endl;
  dataRecorder.setDefaultValue("Layout", "bricks");
  for(unsigned d = 0; d < RANK; ++d) {
    dataRecorder.unsetDefaultValue((std::string) "ArrayPadding_" + dimNames[d]);
    dataRecorder.setDefaultValue((std::string) "BrickDim_" + dimNames[d], BRICK_DIM[d]);
    dataRecorder.setDefaultValue((std::string) "BrickVecDim_" + dimNames[d], BRICK_VECTOR_DIM[d]);
  }

  std::array<BricksArakawaKernelType, 7> kernelTypes = {
      SIMPLE_KLIJMN,
      OPT_IJKLMN,
      OPT_IKJLMN,
      OPT_IKLJMN,
      OPT_KIJLMN,
      OPT_KILJMN,
      OPT_KLIJMN
  };
  for(auto kernelType : kernelTypes) {
    dataRecorder.setDefaultValue("OptimizedBrickKernel",kernelType != SIMPLE_KLIJMN);
    dataRecorder.setDefaultValue("BrickIterationOrder", toString(kernelType));
    std::cout << "Bricks " << toString(kernelType) << " ";

    semiArakawaBrick(brickOut, in, coeffs, kernelType, dataRecorder);
    checkClose(brickOut, arrayOut, {0, 0, 2, 2, 0, 0});
    // clear out brick to be sure correct values don't propagate through loop
    brickOut.set(0.0);
  }
}

/**
 * Run bricks and gtensor ij-derivative benchmarks
 * @param extent the extent of the arrays to use
 * @param dataRecorder where to record the data
 */
void runIJDeriv(std::array<unsigned, RANK> extent, CSVDataRecorder &dataRecorder) {
  std::cout << "Building input arrays..." << std::endl;

  complexArray6D in{complexArray6D::random(extent)},
      arrayOut{extent, 0.0}, brickOut{extent, 0.0};

  // set up coeffs
  std::array<unsigned, RANK - 1> coeffExtent{};
  coeffExtent[0] = extent[0];
  for (unsigned i = 2; i < RANK; ++i) {
    coeffExtent[i-1] = extent[i];
  }
  complexArray5D p1{complexArray5D::random(coeffExtent)},
      p2{complexArray5D::random(coeffExtent)};
  complexArray1D_J ikj({extent[1]});
  constexpr double pi = 3.14159265358979323846;
  for(unsigned j = 0; j < ikj.extent[0]; ++j) {
    ikj(j) = 2 * pi * j * bComplexElem(0.0, 1.0);
  }
  bElem i_deriv_coeff[5] = {1. / 12., -2. / 3., 0., 2. / 3., -1 / 12.};

  std::cout << "Array setup complete. Beginning I-J derivative array computation" << std::endl;
  dataRecorder.setDefaultValue("Layout", "array");
  dataRecorder.setDefaultValue("Kernel", "ijderiv");

  const char dimNames[RANK] = {'i', 'j', 'k', 'l', 'm', 'n'};
  for(unsigned d = 0; d < RANK; ++d) {
    dataRecorder.setDefaultValue((std::string) "ArrayPadding_" + dimNames[d], PADDING[d]);
  }
  // run array computation
  ijderivGTensor(arrayOut, in, p1, p2, ikj, i_deriv_coeff, dataRecorder);

  std::cout << "Array computation complete. Beginning I-J derivative bricks computation" << std::endl;
  dataRecorder.setDefaultValue("Layout", "bricks");
  for(unsigned d = 0; d < RANK; ++d) {
    dataRecorder.unsetDefaultValue((std::string) "ArrayPadding_" + dimNames[d]);
    dataRecorder.setDefaultValue((std::string) "BrickDim_" + dimNames[d], BRICK_DIM[d]);
    dataRecorder.setDefaultValue((std::string) "BrickVecDim_" + dimNames[d], BRICK_VECTOR_DIM[d]);
  }

  std::cout << "Bricks " << " ";

  ijderivBrick(brickOut, in, p1, p2, ikj, dataRecorder);
  checkClose(brickOut, arrayOut, {0, 0, 2, 2, 0, 0});
  // clear out brick to be sure correct values don't propagate through loop
  brickOut.set(0.0);
}


/**
 * @brief Run weak-scaling gene6d benchmark
 */
int main(int argc, char **argv) {
  std::array<unsigned, RANK> extent{};
  std::stringstream input_stream;
  for (int i = 1; i < argc; ++i) {
    input_stream << argv[i] << " ";
  }
  std::string outputFileName;
  bool appendToFile;
  trial_iter_count iter_count =
      parse_single_args(&extent, &outputFileName, &appendToFile, input_stream);
  NUM_WARMUPS = iter_count.num_warmups;
  NUM_ITERATIONS = iter_count.num_iters;

  // Record setup data in CSV
  CSVDataRecorder dataRecorder;

  // Print information about setup (copied and modified from Tuowen Zhao's args.cpp)
  // https://github.com/CtopCsUtahEdu/bricklib/blob/ef28a307962fe319cd723a589df4ff6fb4a75d18/weak/args.cpp#L133-L144
  const char dimNames[RANK] = {'i', 'j', 'k', 'l', 'm', 'n'};
  int numThreads;
#pragma omp parallel shared(numThreads) default(none)
  numThreads = omp_get_num_threads();
  long page_size = sysconf(_SC_PAGESIZE);
  size_t totElems = std::accumulate(extent.begin(), extent.end(), 1, std::multiplies<>());
  int io_col_width = 30;
  dataRecorder.setDefaultValue("PageSize", page_size);
  dataRecorder.setDefaultValue("OpenMPThreads", numThreads);
  std::cout << std::setw(io_col_width) << "Pagesize :" << page_size << "\n"
            << std::setw(io_col_width) << "OpenMP threads :" << numThreads << "\n"
            << std::setw(io_col_width) << "Domain size of :";
  for (int i = 0; i < RANK; ++i) {
    std::cout << std::setw(2) << extent[i];
    if (i < RANK - 1) {
      std::cout << " x ";
    }
    dataRecorder.setDefaultValue((std::string) "Extent_" + dimNames[i], extent[i]);
  }
  std::cout << "\n" << std::setw(io_col_width) << "Brick Size :";
  for (int i = 0; i < RANK; ++i) {
    std::cout << std::setw(2) << BRICK_DIM[i];
    if (i < RANK - 1)
      std::cout << " x ";
  }
  dataRecorder.setDefaultValue("NumWarmups", NUM_WARMUPS);
  dataRecorder.setDefaultValue("NumIterations", NUM_ITERATIONS);
  std::cout << "\n"
            << "Num Warmups: " << NUM_WARMUPS << "\n"
            << "Num Iterations : " << NUM_ITERATIONS << std::endl;
  // check array/brick extents
  for (int i = 0; i < RANK; ++i) {
    if (extent[i] % BRICK_DIM[i] != 0) {
      std::ostringstream error_stream;
      error_stream << "Brick-dimension " << i << " (" << BRICK_DIM[i] << ")"
                   << " does not divide per-process extent " << i << " (" << extent[i]
                   << ")";
      throw std::runtime_error(error_stream.str());
    }
    if((i == 0 || i == 2 || i == 3) && extent[i] < 3 * BRICK_DIM[i]) {
      std::ostringstream error_stream;
      error_stream << "Extent" << i << " (" << extent[i] << ")"
                   << " must be at least 3 * BRICK_DIM[" << i << "] = 3 * " << BRICK_DIM[i];
      throw std::runtime_error(error_stream.str());
    }
  }
  runArakawa(extent, dataRecorder);
  runIJDeriv(extent, dataRecorder);

  // write data
  dataRecorder.writeToFile(outputFileName, appendToFile);

  return 0;
}
