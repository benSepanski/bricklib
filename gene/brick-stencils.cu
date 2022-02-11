#include "brick-stencils.h"

__constant__ bElem const_i_deriv_coeff_dev[5];

__host__
void copy_i_deriv_coeff(const bElem i_deriv_coeff_host[5]) {
  gpuCheck(cudaMemcpyToSymbol(const_i_deriv_coeff_dev, i_deriv_coeff_host, 5 * sizeof(bElem)));
}

constexpr unsigned max_blocks_per_sm(unsigned max_block_size) {
  unsigned max_num_warps_per_block = max_block_size / WARP_SIZE;
  unsigned max_blocks_per_sm = MAX_WARPS_PER_SM / max_num_warps_per_block;
  if (max_blocks_per_sm > MAX_BLOCKS_PER_SM) {
    max_blocks_per_sm = MAX_BLOCKS_PER_SM;
  }
  return max_blocks_per_sm;
}

/**
 * @brief Compute on the non-ghost bricks
 *
 * Assumes that grid-size is I x J x KLMN.
 * Assumes i-deriv coeff has been copied to constant memory
 * @see copy_i_deriv_coeff
 */
__launch_bounds__(NUM_ELEMENTS_PER_FIELD_BRICK,
                  max_blocks_per_sm(NUM_ELEMENTS_PER_FIELD_BRICK))
__global__
void ijDerivBrickKernel(brick::Array<unsigned, RANK, brick::Padding<>, unsigned> grid,
                           brick::Array<unsigned, RANK-1, brick::Padding<>, unsigned> coeffGrid,
                           FieldBrick_i bIn, FieldBrick_i bOut, PreCoeffBrick bP1,
                           PreCoeffBrick bP2, bComplexElem *ikj,
                           int numGhostZonesToSkip) {
  // compute brick index
  unsigned b_i = numGhostZonesToSkip + blockIdx.x;
  unsigned b_j = blockIdx.y;
  unsigned b_klmn = blockIdx.z;
  unsigned b_k = b_klmn % grid.extent[2];
  unsigned b_lmn = b_klmn / grid.extent[2];
  unsigned b_l = b_lmn % grid.extent[3];
  unsigned b_mn = b_lmn / grid.extent[3];
  unsigned b_m = b_mn % grid.extent[4];
  unsigned b_n = b_mn / grid.extent[4];

  // get field and coeff brick indexes
  unsigned bFieldIndex = grid(b_i, b_j, b_k, b_l, b_m, b_n);
  unsigned bCoeffIndex = coeffGrid(b_i, b_k, b_l, b_m, b_n);

  unsigned n = threadIdx.z / (BRICK_DIM[2] * BRICK_DIM[3] * BRICK_DIM[4]);
  unsigned m = threadIdx.z / (BRICK_DIM[2] * BRICK_DIM[3]) % BRICK_DIM[4];
  unsigned l = (threadIdx.z / BRICK_DIM[2]) % BRICK_DIM[3];
  unsigned k = threadIdx.z % BRICK_DIM[2];
  unsigned j = threadIdx.y;
  unsigned i = threadIdx.x;

  // bounds check
  assert(i < BRICK_DIM[0]);
  assert(j < BRICK_DIM[1]);
  assert(k < BRICK_DIM[2]);
  assert(l < BRICK_DIM[3]);
  assert(m < BRICK_DIM[4]);
  assert(n < BRICK_DIM[5]);

  // load data together so that memory accesses get coalesced
  bComplexElem in[5];
  in[0] = bIn[bFieldIndex][n][m][l][k][j][i - 2];
  in[1] = bIn[bFieldIndex][n][m][l][k][j][i - 1];
  in[2] = bIn[bFieldIndex][n][m][l][k][j][i];
  in[3] = bIn[bFieldIndex][n][m][l][k][j][i + 1];
  in[4] = bIn[bFieldIndex][n][m][l][k][j][i + 2];

  bComplexElem p1 = bP1[bCoeffIndex][n][m][l][k][i], p2 = bP2[bCoeffIndex][n][m][l][k][i],
               my_ikj = ikj[PADDING[1] + b_j * BRICK_DIM[1] + j];

  // perform computation
  bComplexElem out = p1 * (const_i_deriv_coeff_dev[0] * in[0] + const_i_deriv_coeff_dev[1] * in[1] +
                           const_i_deriv_coeff_dev[2] * in[2] + const_i_deriv_coeff_dev[3] * in[3] +
                           const_i_deriv_coeff_dev[4] * in[4]) +
                     p2 * my_ikj * in[2];

  // store computation
  bOut[bFieldIndex][n][m][l][k][j][i] = out;
}

/**
 * Compute arakawa on all bricks
 * @param grid the field brick-grid
 * @param coeffGrid the coefficient brick-grid
 * @param bIn the input bricks
 * @param bOut the output bricks
 * @param coeff the coefficient bricks
 * @param numGhostZonesToSkip the number of ghost zones to skip
 */
__launch_bounds__(NUM_ELEMENTS_PER_FIELD_BRICK, max_blocks_per_sm(NUM_ELEMENTS_PER_FIELD_BRICK))
__global__
void semiArakawaBrickKernelSimple(brick::Array<unsigned, RANK, brick::Padding<>, unsigned> grid,
                                  brick::Array<unsigned, RANK, brick::Padding<>, unsigned> coeffGrid,
                                  FieldBrick_kl bIn, FieldBrick_kl bOut, ArakawaCoeffBrick coeff,
                                  int numGhostZonesToSkip) {
  // get brick index for field
  unsigned b_k = blockIdx.x + numGhostZonesToSkip;
  unsigned b_l = blockIdx.y + numGhostZonesToSkip;
  unsigned b_ijmn = blockIdx.z;
  unsigned b_i = b_ijmn % grid.extent[0];
  unsigned b_jmn = b_ijmn / grid.extent[0];
  unsigned b_j = b_jmn % grid.extent[1];
  unsigned b_mn = b_jmn / grid.extent[1];
  unsigned b_m = b_mn % grid.extent[4];
  unsigned b_n = b_mn / grid.extent[4];

  // get field and coeff brick indexes
  unsigned fieldBrickIdx = grid(b_i, b_j, b_k, b_l, b_m, b_n);

  // intra-brick indexing
  int i = threadIdx.x;
  int j = threadIdx.y;
  int klmn = threadIdx.z;
  int k = klmn % BRICK_DIM[2];
  int lmn = klmn / BRICK_DIM[2];
  int l = lmn % BRICK_DIM[3];
  int mn = lmn / BRICK_DIM[3];
  int m = mn % BRICK_DIM[4];
  int n = mn / BRICK_DIM[4];

  // check for intra-brick OOB
  assert(i < BRICK_DIM[0] && i < ARAKAWA_COEFF_BRICK_DIM[1]);
  assert(j < BRICK_DIM[1]);
  assert(k < BRICK_DIM[2] && k < ARAKAWA_COEFF_BRICK_DIM[2]);
  assert(l < BRICK_DIM[3] && l < ARAKAWA_COEFF_BRICK_DIM[3]);
  assert(m < BRICK_DIM[4] && m < ARAKAWA_COEFF_BRICK_DIM[4]);
  assert(n < BRICK_DIM[5] && n < ARAKAWA_COEFF_BRICK_DIM[5]);

  // compute stencil
  bComplexElem result = 0.0;
  auto in = bIn[fieldBrickIdx];
  auto input = [&](int deltaK, int deltaL) -> bComplexElem {
    return in[n][m][l + deltaL][k + deltaK][j][i];
  };
  auto c = [&](unsigned stencilIdx) -> bElem {
    unsigned coeffBrickIndex = coeffGrid(stencilIdx, b_i, b_k, b_l, b_m, b_n);
    return coeff[coeffBrickIndex][n][m][l][k][i][0];
  };
  bOut[fieldBrickIdx][n][m][l][k][j][i] =
         c(0) * input(0, -2)
       + c(1) * input(-1, -1)
       + c(2) * input(0, -1)
       + c(3) * input(1, -1)
       + c(4) * input(-2, 0)
       + c(5) * input(-1, 0)
       + c(6) * input(0, 0)
       + c(7) * input(1, 0)
       + c(8) * input(2, 0)
       + c(9) * input(-1, 1)
       + c(10) * input(0, 1)
       + c(11) * input(1, 1)
       + c(12) * input(0, 2);
}


__constant__ unsigned optimizedSemiArakawaGridIterationOrder[RANK];
__constant__ unsigned optimizedSemiArakawaFieldBrickGridExtent[RANK];
__constant__ unsigned optimizedSemiArakawaFieldBrickGridStride[RANK];


/**
 * Compute arakawa on all bricks.
 *
 * The user must set brick grid extents and brick grid strides appropriately.
 * optimizedSemiArakawaGridIterationOrder
 * must be set appropriately (e.g. [2, 3, 0, 1, 4, 5]) to set the iteration order
 *
 * @param grid the field brick-grid
 * @param coeffGrid the coefficient brick-grid
 * @param bIn the input bricks
 * @param bOut the output bricks
 * @param coeff the coefficient bricks
* @param numGhostZonesToSkip the number of ghost zones to skip
 */
__launch_bounds__(NUM_ELEMENTS_PER_FIELD_BRICK, max_blocks_per_sm(NUM_ELEMENTS_PER_FIELD_BRICK))
__global__ void
semiArakawaBrickKernelOptimized(brick::Array<unsigned, RANK, brick::Padding<>, unsigned> grid,
                                brick::Array<unsigned, RANK, brick::Padding<>, unsigned> coeffGrid,
                                FieldBrick_kl bIn,
                                FieldBrick_kl bOut,
                                ArakawaCoeffBrick coeff,
                                int numGhostZonesToSkip)
{
  static_assert(ARAKAWA_COEFF_BRICK_DIM[0] == 1, "Assumes arakawa brick-dim is 1 in first index");
  // compute indices
  unsigned bFieldIndex, ///< Index into storage of current field brick
      bCoeffGridIndex; ///< Index into coeffGrid of first coeff brick
  {
    // alias the const memory
    const unsigned *const gridIterationOrder = optimizedSemiArakawaGridIterationOrder;
    const unsigned * const extent = optimizedSemiArakawaFieldBrickGridExtent;
    const unsigned * const stride = optimizedSemiArakawaFieldBrickGridStride;

    unsigned bFieldGridIndex = numGhostZonesToSkip * (stride[2] + stride[3]);

    const unsigned d0 = gridIterationOrder[0];
    assert(blockIdx.x < extent[d0]);
    bFieldGridIndex += blockIdx.x * stride[d0];
    const unsigned d1 = gridIterationOrder[1];
    assert(blockIdx.y < extent[d1]);
    bFieldGridIndex += blockIdx.y * stride[d1];

    unsigned blockIdxZ = blockIdx.z;
    for (unsigned axis = 2; axis < RANK; ++axis) {
      const unsigned dAxis = gridIterationOrder[axis];
      bFieldGridIndex += (blockIdxZ % extent[dAxis]) * stride[dAxis];
      blockIdxZ /= extent[dAxis];
    }
    assert(blockIdxZ == 0);
    bFieldIndex = grid.atFlatIndex(bFieldGridIndex);
    // IJKLMN -> IKLMN -> CoefficientAxisIKLMN
    bCoeffGridIndex =
        ((bFieldGridIndex % extent[0]) + extent[0] * (bFieldGridIndex / (extent[0] * extent[1]))) *
        ARAKAWA_STENCIL_SIZE;
  }

  brick("semi_arakawa_stencil.py", "CUDA", (GENE6D_BRICK_DIM), (GENE6D_VEC_DIM), bFieldIndex);
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

ijDerivBrickKernelType buildBricksIJDerivBrickKernel(brick::BrickLayout<RANK> fieldLayout,
                                                     BrickedPCoeffArray &p1,
                                                     BrickedPCoeffArray &p2,
                                                     const complexArray1D_J &ikj,
                                                     bElem i_deriv_coeffs[5],
                                                     int numGhostZonesToSkip) {
  const unsigned *const brickExtentWithGZ = fieldLayout.indexInStorage.extent;
  if(numGhostZonesToSkip < 0) {
    throw std::runtime_error("numGhostZonesToSkip must be non-negative");
  }
  if(numGhostZonesToSkip * 3 > brickExtentWithGZ[0]) {
    std::ostringstream errStream;
    errStream << "numGhostZonesToSkip = " << numGhostZonesToSkip
              << " is greater than 3 * brick-grid-extent[0] = 3 * " << brickExtentWithGZ[0];
    throw std::runtime_error(errStream.str());
  }

  dim3 cuda_grid_size(brickExtentWithGZ[0] - 2 * numGhostZonesToSkip, brickExtentWithGZ[1],
                      brickExtentWithGZ[2] * brickExtentWithGZ[3] * brickExtentWithGZ[4] *
                          brickExtentWithGZ[5]),
      cuda_block_size(BRICK_DIM[0], BRICK_DIM[1],
                      FieldBrick_kl::BRICKLEN / BRICK_DIM[0] / BRICK_DIM[1]);
  validateLaunchConfig(cuda_grid_size, cuda_block_size);

  // Get known parameters to kernel
  auto fieldIndexInStorage_dev = fieldLayout.indexInStorage.allocateOnDevice();
  fieldLayout.indexInStorage.copyToDevice(fieldIndexInStorage_dev);
  if(p1.getLayout().indexInStorage.getData().get() != p2.getLayout().indexInStorage.getData().get()) {
    throw std::runtime_error("p1 and p2 must use the same layout");
  }
  auto pCoeffIndexInStorage_dev = p1.getLayout().indexInStorage.allocateOnDevice();
  p1.getLayout().indexInStorage.copyToDevice(pCoeffIndexInStorage_dev);
  p1.copyToDevice();
  PreCoeffBrick p1_dev = p1.viewBricksOnDevice<NoComm>();
  p2.copyToDevice();
  PreCoeffBrick p2_dev = p2.viewBricksOnDevice<NoComm>();

  // copy ikj to device
  auto ikj_dev = ikj.allocateOnDevice();
  ikj.copyToDevice(ikj_dev);

  // copy coefficients to constant memory
  copy_i_deriv_coeff(i_deriv_coeffs);

  // define the computation
  std::function<void(FieldBrick_kl, FieldBrick_kl)> arakawaComputation;
  auto ijDerivComputation = [=](const FieldBrick_i& bIn_dev, const FieldBrick_i& bOut_dev) -> void {
    ijDerivBrickKernel<<<cuda_grid_size, cuda_block_size>>>(
        fieldIndexInStorage_dev, pCoeffIndexInStorage_dev, bIn_dev, bOut_dev, p1_dev, p2_dev, ikj_dev.getData().get(),
        numGhostZonesToSkip);
#ifndef NDEBUG
    gpuCheck(cudaPeekAtLastError());
#endif
  };
  return ijDerivComputation;
}

ArakawaBrickKernelType buildBricksArakawaKernel(brick::BrickLayout<RANK> fieldLayout,
                                                BrickedArakawaCoeffArray bCoeff,
                                                BricksArakawaKernelType kernelType,
                                                int numGhostZonesToSkip) {
  if(numGhostZonesToSkip < 0) {
    throw std::runtime_error("numGhostZonesToSkip must be non-negative");
  }
  for(unsigned d = 2; d <= 3; ++d) {
    if (numGhostZonesToSkip * 3 > fieldLayout.indexInStorage.extent[d]) {
      std::ostringstream errStream;
      errStream << "numGhostZonesToSkip = " << numGhostZonesToSkip
                << " is greater than 3 * brick-grid-extent[" << d << "]"
                << " = 3 * " << fieldLayout.indexInStorage.extent[d];
      throw std::runtime_error(errStream.str());
    }
  }

  std::string iterationOrderString = toString(kernelType);
  std::array<unsigned, RANK> iterationOrder{};
  for(unsigned d = 0; d < RANK; ++d) {
    iterationOrder[d] = iterationOrderString[d] - 'i';
  }

  // check input
  if(bCoeff.getLayout().indexInStorage.extent[0] != ARAKAWA_STENCIL_SIZE) {
    throw std::runtime_error("bCoeff extent [0] must be ARAKAWA_STENCIL_SIZE");
  }
  if(bCoeff.getLayout().indexInStorage.extent[1] != fieldLayout.indexInStorage.extent[0]) {
    throw std::runtime_error("bCoeff and fieldLayout extents are incompatible");
  }
  for(unsigned d = 2; d < RANK; ++d) {
    if(bCoeff.getLayout().indexInStorage.extent[d] != fieldLayout.indexInStorage.extent[d]) {
      throw std::runtime_error("bCoeff and fieldLayout extents are incompatible");
    }
  }

  std::array<unsigned, RANK> brickExtent{};
  for(unsigned d = 0; d < RANK; ++d) {
    brickExtent[d] = fieldLayout.indexInStorage.extent[d];
    if(d == 2 || d == 3) {
      brickExtent[d] -= 2 * numGhostZonesToSkip;
    }
  }
  dim3 cuda_grid_size(brickExtent[iterationOrder[0]], brickExtent[iterationOrder[1]],
                      brickExtent[iterationOrder[2]] * brickExtent[iterationOrder[3]] *
                      brickExtent[iterationOrder[4]] * brickExtent[iterationOrder[5]]);

  // Get known parameters to kernel
  auto fieldIndexInStorage_dev = fieldLayout.indexInStorage.allocateOnDevice();
  fieldLayout.indexInStorage.copyToDevice(fieldIndexInStorage_dev);
  auto coeffIndexInStorage_dev = bCoeff.getLayout().indexInStorage.allocateOnDevice();
  bCoeff.getLayout().indexInStorage.copyToDevice(coeffIndexInStorage_dev);
  bCoeff.copyToDevice();
  ArakawaCoeffBrick bCoeff_dev = bCoeff.viewBricksOnDevice<NoComm>();

  // define the computation
  std::function<void(FieldBrick_kl, FieldBrick_kl)> arakawaComputation;
  if(kernelType == SIMPLE_KLIJMN) {
    dim3 cuda_block_size(BRICK_DIM[0], BRICK_DIM[1],
                    FieldBrick_kl::BRICKLEN / BRICK_DIM[0] / BRICK_DIM[1]);
    validateLaunchConfig(cuda_grid_size, cuda_block_size);

    arakawaComputation = [=](FieldBrick_kl bIn_dev, FieldBrick_kl bOut_dev) -> void {
      semiArakawaBrickKernelSimple<< <cuda_grid_size, cuda_block_size>> >(
          fieldIndexInStorage_dev, coeffIndexInStorage_dev, bIn_dev, bOut_dev, bCoeff_dev,
          numGhostZonesToSkip);
#ifndef NDEBUG
      gpuCheck(cudaPeekAtLastError());
#endif
    };
  }
  else {
    unsigned fieldBrickGridStride[RANK];
    fieldBrickGridStride[0] = 1;
    for(size_t d = 1; d < RANK; ++d) {
      fieldBrickGridStride[d] = fieldBrickGridStride[d-1] * fieldLayout.indexInStorage.extent[d-1];
    }

    gpuCheck(cudaMemcpyToSymbol(optimizedSemiArakawaGridIterationOrder, iterationOrder.data(), RANK * sizeof(unsigned)));
    gpuCheck(cudaMemcpyToSymbol(optimizedSemiArakawaFieldBrickGridExtent, brickExtent.data(), RANK * sizeof(unsigned)));
    gpuCheck(cudaMemcpyToSymbol(optimizedSemiArakawaFieldBrickGridStride, fieldBrickGridStride, RANK * sizeof(unsigned)));

    dim3 cuda_block_size{32};
    validateLaunchConfig(cuda_grid_size, cuda_block_size);

    arakawaComputation = [=](FieldBrick_kl bIn_dev, FieldBrick_kl bOut_dev) -> void {
      semiArakawaBrickKernelOptimized<< <cuda_grid_size, cuda_block_size>> >(
          fieldIndexInStorage_dev, coeffIndexInStorage_dev, bIn_dev, bOut_dev, bCoeff_dev,
          numGhostZonesToSkip);
    #ifndef NDEBUG
        gpuCheck(cudaPeekAtLastError());
    #endif
    };
  }

  return arakawaComputation;
}
