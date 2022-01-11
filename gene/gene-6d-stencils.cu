#include "assert.h"
#include "gene-6d-stencils.h"

__constant__ bElem const_i_deriv_coeff_dev[5];

__host__
void copy_i_deriv_coeff(const bElem i_deriv_coeff_host[5]) {
  gpuCheck(cudaMemcpyToSymbol(const_i_deriv_coeff_dev, i_deriv_coeff_host, 5 * sizeof(bElem)));
}

__launch_bounds__(NUM_ELEMENTS_PER_FIELD_BRICK,
                  max_blocks_per_sm(NUM_ELEMENTS_PER_FIELD_BRICK))
__global__
void ijDerivBrickKernel(brick::Array<unsigned, RANK, brick::Padding<>, unsigned> grid,
                           brick::Array<unsigned, RANK-1, brick::Padding<>, unsigned> coeffGrid,
                           FieldBrick_i bIn, FieldBrick_i bOut, PreCoeffBrick bP1,
                           PreCoeffBrick bP2, bComplexElem *ikj) {
  // compute brick index
  unsigned b_i = blockIdx.x;
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
 */
__launch_bounds__(NUM_ELEMENTS_PER_FIELD_BRICK, max_blocks_per_sm(NUM_ELEMENTS_PER_FIELD_BRICK))
__global__
void semiArakawaBrickKernelSimple(brick::Array<unsigned, RANK, brick::Padding<>, unsigned> grid,
                                  brick::Array<unsigned, RANK, brick::Padding<>, unsigned> coeffGrid,
                                  FieldBrick_kl bIn, FieldBrick_kl bOut, ArakawaCoeffBrick coeff) {
  // get brick index for field
  unsigned b_k = blockIdx.x;
  unsigned b_l = blockIdx.y;
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
 */
__launch_bounds__(NUM_ELEMENTS_PER_FIELD_BRICK, max_blocks_per_sm(NUM_ELEMENTS_PER_FIELD_BRICK))
__global__ void
semiArakawaBrickKernelOptimized(brick::Array<unsigned, RANK, brick::Padding<>, unsigned> grid,
                                brick::Array<unsigned, RANK, brick::Padding<>, unsigned> coeffGrid,
                                FieldBrick_kl bIn,
                                FieldBrick_kl bOut,
                                ArakawaCoeffBrick coeff)
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

    unsigned bFieldGridIndex = 0;

    const unsigned d0 = gridIterationOrder[0];
    assert(blockIdx.x < extent[d0]);
    bFieldGridIndex = blockIdx.x * stride[d0];
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

  unsigned *neighbors = bIn.bInfo->adj[bFieldIndex];

  int n = threadIdx.z / (BRICK_DIM[2] *  BRICK_DIM[3]  * BRICK_DIM[4]);
  int m = threadIdx.z / (BRICK_DIM[2] *  BRICK_DIM[3]) % BRICK_DIM[4];
  int l = threadIdx.z /  BRICK_DIM[2] % BRICK_DIM[3];
  int k = threadIdx.z %  BRICK_DIM[2];
  int j = threadIdx.y;
  int i = threadIdx.x;

  // bounds check
  assert(i < BRICK_DIM[0]);
  assert(j < BRICK_DIM[1]);
  assert(k < BRICK_DIM[2]);
  assert(l < BRICK_DIM[3]);
  assert(m < BRICK_DIM[4]);
  assert(n < BRICK_DIM[5]);

  // load in data
  bComplexElem in[13], result = 0.0;
  auto myIndex = BrickIndex<FieldBrick_kl>(n, m, l, k, j, i);
  unsigned flatIndexNoJ = i + BRICK_DIM[0] * (k + BRICK_DIM[2] * (l + BRICK_DIM[3] * (m + BRICK_DIM[4] * n)));

  myIndex.shiftInDims<3>(-2);
  in[0] = bIn.dat[bIn.step * neighbors[myIndex.indexInNbrList] + myIndex.getIndexInBrick()];
  bElem my_coeff = coeff.dat[coeff.step * coeffGrid.atFlatIndex(bCoeffGridIndex) + flatIndexNoJ];
  result += in[0] * my_coeff;

  myIndex.shiftInDims<3, 2>(+1, -1);
  in[1] = bIn.dat[bIn.step * neighbors[myIndex.indexInNbrList] + myIndex.getIndexInBrick()];
  my_coeff = coeff.dat[coeff.step * coeffGrid.atFlatIndex(bCoeffGridIndex + 1) + flatIndexNoJ];
  result += in[1] * my_coeff;

  myIndex.shiftInDims<2>(+1);
  in[2] = bIn.dat[bIn.step * neighbors[myIndex.indexInNbrList] + myIndex.getIndexInBrick()];
  my_coeff = coeff.dat[coeff.step * coeffGrid.atFlatIndex(bCoeffGridIndex + 2) + flatIndexNoJ];
  result += in[2] * my_coeff;

  myIndex.shiftInDims<2>(+1);
  in[3] = bIn.dat[bIn.step * neighbors[myIndex.indexInNbrList] + myIndex.getIndexInBrick()];
  my_coeff = coeff.dat[coeff.step * coeffGrid.atFlatIndex(bCoeffGridIndex + 3) + flatIndexNoJ];
  result += in[3] * my_coeff;

  myIndex.shiftInDims<3, 2>(+1, -3);
  in[4] = bIn.dat[bIn.step * neighbors[myIndex.indexInNbrList] + myIndex.getIndexInBrick()];
  my_coeff = coeff.dat[coeff.step * coeffGrid.atFlatIndex(bCoeffGridIndex + 4) + flatIndexNoJ];
  result += in[4] * my_coeff;

  myIndex.shiftInDims<2>(+1);
  in[5] = bIn.dat[bIn.step * neighbors[myIndex.indexInNbrList] + myIndex.getIndexInBrick()];
  my_coeff = coeff.dat[coeff.step * coeffGrid.atFlatIndex(bCoeffGridIndex + 5) + flatIndexNoJ];
  result += in[5] * my_coeff;

  myIndex.shiftInDims<2>(+1);
  in[6] = bIn.dat[bIn.step * neighbors[myIndex.indexInNbrList] + myIndex.getIndexInBrick()];
  my_coeff = coeff.dat[coeff.step * coeffGrid.atFlatIndex(bCoeffGridIndex + 6) + flatIndexNoJ];
  result += in[6] * my_coeff;

  myIndex.shiftInDims<2>(+1);
  in[7] = bIn.dat[bIn.step * neighbors[myIndex.indexInNbrList] + myIndex.getIndexInBrick()];
  my_coeff = coeff.dat[coeff.step * coeffGrid.atFlatIndex(bCoeffGridIndex + 7) + flatIndexNoJ];
  result += in[7] * my_coeff;

  myIndex.shiftInDims<2>(+1);
  in[8] = bIn.dat[bIn.step * neighbors[myIndex.indexInNbrList] + myIndex.getIndexInBrick()];
  my_coeff = coeff.dat[coeff.step * coeffGrid.atFlatIndex(bCoeffGridIndex + 8) + flatIndexNoJ];
  result += in[8] * my_coeff;

  myIndex.shiftInDims<3, 2>(+1, -3);
  in[9] = bIn.dat[bIn.step * neighbors[myIndex.indexInNbrList] + myIndex.getIndexInBrick()];
  my_coeff = coeff.dat[coeff.step * coeffGrid.atFlatIndex(bCoeffGridIndex + 9) + flatIndexNoJ];
  result += in[9] * my_coeff;

  myIndex.shiftInDims<2>(+1);
  in[10] = bIn.dat[bIn.step * neighbors[myIndex.indexInNbrList] + myIndex.getIndexInBrick()];
  my_coeff = coeff.dat[coeff.step * coeffGrid.atFlatIndex(bCoeffGridIndex + 10) + flatIndexNoJ];
  result += in[10] * my_coeff;

  myIndex.shiftInDims<2>(+1);
  in[11] = bIn.dat[bIn.step * neighbors[myIndex.indexInNbrList] + myIndex.getIndexInBrick()];
  my_coeff = coeff.dat[coeff.step * coeffGrid.atFlatIndex(bCoeffGridIndex + 11) + flatIndexNoJ];
  result += in[11] * my_coeff;

  myIndex.shiftInDims<3, 2>(+1, -1);
  in[12] = bIn.dat[bIn.step * neighbors[myIndex.indexInNbrList] + myIndex.getIndexInBrick()];
  my_coeff = coeff.dat[coeff.step * coeffGrid.atFlatIndex(bCoeffGridIndex + 12) + flatIndexNoJ];
  result += in[12] * my_coeff;

  // store result
  unsigned flatIndex = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
  bOut.dat[bFieldIndex * bOut.step + flatIndex] = result;
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

ArakawaBrickKernel buildBricksArakawaKernel(brick::BrickLayout<RANK> fieldLayout,
                                            BrickedArakawaCoeffArray bCoeff,
                                            BricksArakawaKernelType kernelType) {

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

  const unsigned *const brickExtentWithGZ = fieldLayout.indexInStorage.extent;
  dim3 cuda_grid_size(brickExtentWithGZ[iterationOrder[0]], brickExtentWithGZ[iterationOrder[1]],
                      brickExtentWithGZ[iterationOrder[2]] * brickExtentWithGZ[iterationOrder[3]] *
                          brickExtentWithGZ[iterationOrder[4]] * brickExtentWithGZ[iterationOrder[5]]),
      cuda_block_size(BRICK_DIM[0], BRICK_DIM[1],
                      FieldBrick_kl::BRICKLEN / BRICK_DIM[0] / BRICK_DIM[1]);
  validateLaunchConfig(cuda_grid_size, cuda_block_size);

  // Get known parameters to kernel
  auto fieldIndexInStorage_dev = fieldLayout.indexInStorage.allocateOnDevice();
  fieldLayout.indexInStorage.copyToDevice(fieldIndexInStorage_dev);
  auto coeffIndexInStorage_dev = bCoeff.getLayout().indexInStorage.allocateOnDevice();
  bCoeff.getLayout().indexInStorage.copyToDevice(coeffIndexInStorage_dev);
  ArakawaCoeffBrick bCoeff_dev = bCoeff.viewBricksOnDevice<NoComm>();

  // define the computation
  std::function<void(FieldBrick_kl, FieldBrick_kl)> arakawaComputation;
  if(kernelType == SIMPLE_KLIJMN) {
    arakawaComputation = [=](FieldBrick_kl bIn_dev, FieldBrick_kl bOut_dev) -> void {
      semiArakawaBrickKernelSimple<< <cuda_grid_size, cuda_block_size>> >(
          fieldIndexInStorage_dev, coeffIndexInStorage_dev, bIn_dev, bOut_dev, bCoeff_dev);
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
    gpuCheck(cudaMemcpyToSymbol(optimizedSemiArakawaFieldBrickGridExtent, fieldLayout.indexInStorage.extent, RANK * sizeof(unsigned)));
    gpuCheck(cudaMemcpyToSymbol(optimizedSemiArakawaFieldBrickGridStride, fieldBrickGridStride, RANK * sizeof(unsigned)));

    arakawaComputation = [=](FieldBrick_kl bIn_dev, FieldBrick_kl bOut_dev) -> void {
      semiArakawaBrickKernelOptimized<< <cuda_grid_size, cuda_block_size>> >(
          fieldIndexInStorage_dev, coeffIndexInStorage_dev, bIn_dev, bOut_dev, bCoeff_dev);
    #ifndef NDEBUG
        gpuCheck(cudaPeekAtLastError());
    #endif
    };
  }

  return arakawaComputation;
}
