#include "assert.h"
#include "gene-6d-stencils.h"

constexpr unsigned GHOST_ZONE_BRICKS[RANK] = {
    GHOST_ZONE[0] / BRICK_DIM[0], GHOST_ZONE[1] / BRICK_DIM[1], GHOST_ZONE[2] / BRICK_DIM[2],
    GHOST_ZONE[3] / BRICK_DIM[3], GHOST_ZONE[4] / BRICK_DIM[4], GHOST_ZONE[5] / BRICK_DIM[5]};

__constant__ bElem const_i_deriv_coeff_dev[5];
__constant__ unsigned brick_grid_extents[RANK] = {BRICK_EXTENT};
__constant__ unsigned brick_grid_strides[RANK] = {
    1,
    GZ_BRICK_EXTENT_i,
    GZ_BRICK_EXTENT_i *GZ_BRICK_EXTENT_j,
    GZ_BRICK_EXTENT_i *GZ_BRICK_EXTENT_j *GZ_BRICK_EXTENT_k,
    GZ_BRICK_EXTENT_i *GZ_BRICK_EXTENT_j *GZ_BRICK_EXTENT_k *GZ_BRICK_EXTENT_l,
    GZ_BRICK_EXTENT_i *GZ_BRICK_EXTENT_j *GZ_BRICK_EXTENT_k *GZ_BRICK_EXTENT_l *GZ_BRICK_EXTENT_m};
__constant__ unsigned brick_grid_ghost_zones[RANK] = {GHOST_ZONE_BRICKS[0], GHOST_ZONE_BRICKS[1],
                                                      GHOST_ZONE_BRICKS[2], GHOST_ZONE_BRICKS[3],
                                                      GHOST_ZONE_BRICKS[4], GHOST_ZONE_BRICKS[5]};
__constant__ char grid_iteration_order[RANK + 1];

__host__ void copy_i_deriv_coeff(const bElem i_deriv_coeff_host[5]) {
  gpuCheck(cudaMemcpyToSymbol(const_i_deriv_coeff_dev, i_deriv_coeff_host, 5 * sizeof(bElem)));
}

__host__ void copy_grid_iteration_order(const char *grid_iteration_order_host) {
  gpuCheck(cudaMemcpyToSymbol(grid_iteration_order, grid_iteration_order_host,
                              (RANK + 1) * sizeof(char)));
}

__launch_bounds__(NUM_ELEMENTS_PER_FIELD_BRICK,
                  max_blocks_per_sm(NUM_ELEMENTS_PER_FIELD_BRICK)) __global__
    void ij_deriv_brick_kernel(brick::Array<unsigned, RANK, brick::Padding<>, unsigned> grid,
                               brick::Array<unsigned, RANK, brick::Padding<>, unsigned> coeffGrid,
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
  unsigned fieldBrickIdx = grid(b_i, b_j, b_k, b_l, b_m, b_n);
  unsigned coeffBrickIdx = coeffGrid(b_i, b_k, b_l, b_m, b_n);

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
               my_ikj = ikj[PADDING_j + tj * BDIM_j + j];

  // perform computation
  bComplexElem out = p1 * (const_i_deriv_coeff_dev[0] * in[0] + const_i_deriv_coeff_dev[1] * in[1] +
                           const_i_deriv_coeff_dev[2] * in[2] + const_i_deriv_coeff_dev[3] * in[3] +
                           const_i_deriv_coeff_dev[4] * in[4]) +
                     p2 * my_ikj * in[2];

  // store computation
  bOut[bFieldIndex][n][m][l][k][j][i] = out;
}

__launch_bounds__(NUM_ELEMENTS_PER_BRICK, max_blocks_per_sm(NUM_ELEMENTS_PER_BRICK)) __global__
    void semi_arakawa_brick_kernel(brick::Array<unsigned, RANK, brick::Padding<>, unsigned> grid,
                                   brick::Array<unsigned, RANK, brick::Padding<>, unsigned> coeffGrid,
                                   FieldBrick_kl bIn,
                                   FieldBrick_kl bOut,
                                   ArakawaCoeffBrick coeff) {
  // compute indices
  unsigned idx[RANK];
  unsigned d0 = grid_iteration_order[0] - 'i';
  idx[d0] = brick_grid_ghost_zones[d0] + blockIdx.x;
  unsigned d1 = grid_iteration_order[1] - 'i';
  idx[d1] = brick_grid_ghost_zones[d1] + blockIdx.y;
  unsigned zIdx = blockIdx.z;
  for (unsigned dimIdx = 2; dimIdx < RANK; ++dimIdx) {
    unsigned d = grid_iteration_order[dimIdx] - 'i';
    idx[d] = brick_grid_ghost_zones[d] + zIdx % grid.extent[d];
    zIdx /= grid.extent[d];
  }
  unsigned bFieldIndex = grid(idx[0], idx[1], idx[2], idx[3], idx[4], idx[5]);
  static_assert(ARAKAWA_COEFF_BRICK_DIM[0] == 1,
                "Arakawa-coefficient brick must have extent 1 in the coefficient index");
  unsigned *bCoeffIndices = &coeffGrid(0, idx[0], idx[2], idx[3], idx[4], idx[5]);

  unsigned *neighbors = bIn.bInfo->adj[bFieldIndex];

  int n = threadIdx.z / (BRICK_DIM[2] * BRICK_DIM[3] * BRICK_DIM[4]);
  int m = threadIdx.z / (BRICK_DIM[2] * BRICK_DIM[3]) % BRICK_DIM[4];
  int l = threadIdx.z / BRICK_DIM[2] % BRICK_DIM[3];
  int k = threadIdx.z % BRICK_DIM[2];
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
  bElem my_coeff = coeff.dat[coeff[0].step * bCoeffIndices[0] + flatIndexNoJ];
  result += in[0] * my_coeff;

  myIndex.shiftInDims<3, 2>(+1, -1);
  in[1] = bIn.dat[bIn.step * neighbors[myIndex.indexInNbrList] + myIndex.getIndexInBrick()];
  my_coeff = coeff.dat[coeff[1].step * bCoeffIndices[1] + flatIndexNoJ];
  result += in[1] * my_coeff;

  myIndex.shiftInDims<2>(+1);
  in[2] = bIn.dat[bIn.step * neighbors[myIndex.indexInNbrList] + myIndex.getIndexInBrick()];
  my_coeff = coeff.dat[coeff[2].step * bCoeffIndices[2] + flatIndexNoJ];
  result += in[2] * my_coeff;

  myIndex.shiftInDims<2>(+1);
  in[3] = bIn.dat[bIn.step * neighbors[myIndex.indexInNbrList] + myIndex.getIndexInBrick()];
  my_coeff = coeff.dat[coeff[3].step * bCoeffIndices[3] + flatIndexNoJ];
  result += in[3] * my_coeff;

  myIndex.shiftInDims<3, 2>(+1, -3);
  in[4] = bIn.dat[bIn.step * neighbors[myIndex.indexInNbrList] + myIndex.getIndexInBrick()];
  my_coeff = coeff.dat[coeff[4].step * bCoeffIndices[4] + flatIndexNoJ];
  result += in[4] * my_coeff;

  myIndex.shiftInDims<2>(+1);
  in[5] = bIn.dat[bIn.step * neighbors[myIndex.indexInNbrList] + myIndex.getIndexInBrick()];
  my_coeff = coeff.dat[coeff[5].step * bCoeffIndices[5] + flatIndexNoJ];
  result += in[5] * my_coeff;

  myIndex.shiftInDims<2>(+1);
  in[6] = bIn.dat[bIn.step * neighbors[myIndex.indexInNbrList] + myIndex.getIndexInBrick()];
  my_coeff = coeff.dat[coeff[6].step * bCoeffIndices[6] + flatIndexNoJ];
  result += in[6] * my_coeff;

  myIndex.shiftInDims<2>(+1);
  in[7] = bIn.dat[bIn.step * neighbors[myIndex.indexInNbrList] + myIndex.getIndexInBrick()];
  my_coeff = coeff.dat[coeff[7].step * bCoeffIndices[7] + flatIndexNoJ];
  result += in[7] * my_coeff;

  myIndex.shiftInDims<2>(+1);
  in[8] = bIn.dat[bIn.step * neighbors[myIndex.indexInNbrList] + myIndex.getIndexInBrick()];
  my_coeff = coeff.dat[coeff[8].step * bCoeffIndices[8] + flatIndexNoJ];
  result += in[8] * my_coeff;

  myIndex.shiftInDims<3, 2>(+1, -3);
  in[9] = bIn.dat[bIn.step * neighbors[myIndex.indexInNbrList] + myIndex.getIndexInBrick()];
  my_coeff = coeff.dat[coeff[9].step * bCoeffIndices[9] + flatIndexNoJ];
  result += in[9] * my_coeff;

  myIndex.shiftInDims<2>(+1);
  in[10] = bIn.dat[bIn.step * neighbors[myIndex.indexInNbrList] + myIndex.getIndexInBrick()];
  my_coeff = coeff.dat[coeff[10].step * bCoeffIndices[10] + flatIndexNoJ];
  result += in[10] * my_coeff;

  myIndex.shiftInDims<2>(+1);
  in[11] = bIn.dat[bIn.step * neighbors[myIndex.indexInNbrList] + myIndex.getIndexInBrick()];
  my_coeff = coeff.dat[coeff[11].step * bCoeffIndices[11] + flatIndexNoJ];
  result += in[11] * my_coeff;

  myIndex.shiftInDims<3, 2>(+1, -1);
  in[12] = bIn.dat[bIn.step * neighbors[myIndex.indexInNbrList] + myIndex.getIndexInBrick()];
  my_coeff = coeff.dat[coeff[12].step * bCoeffIndices[12] + flatIndexNoJ];
  result += in[12] * my_coeff;

  // store result
  unsigned flatIndex = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
  bOut.dat[bFieldIndex * bOut.step + flatIndex] = result;
}
