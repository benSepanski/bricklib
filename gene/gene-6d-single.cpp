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
#include "util.h"
#include "single-util.h"

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

  std::cout << "Coefficient setup complete. Beginning array computation" << std::endl;
  dataRecorder.setDefaultValue("Layout", "array");

  for(unsigned d = 0; d < RANK; ++d) {
    dataRecorder.setDefaultValue((std::string) "ArrayPadding_" + dimNames[d], PADDING[d]);
  }
  // run array computation
  semiArakawaGTensor(arrayOut, in, coeffs, dataRecorder);
  std::cout << "Array computation complete. Beginning bricks computation" << std::endl;
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

  // write data
  dataRecorder.writeToFile(outputFileName, appendToFile);

  return 0;
}
