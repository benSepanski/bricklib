#include "assert.h"
#include "gene-6d-stencils.h"

__constant__ bElem const_i_deriv_coeff_dev[5];
__constant__ char grid_iteration_order[RANK + 1];

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

__launch_bounds__(NUM_ELEMENTS_PER_FIELD_BRICK, max_blocks_per_sm(NUM_ELEMENTS_PER_FIELD_BRICK))
__global__
void semiArakawaBrickKernel(brick::Array<unsigned, RANK, brick::Padding<>, unsigned> grid,
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
      c(0) * input(0, -2) + c(1) * input(-1, -1) + c(2) * input(0, -1) + c(3) * input(1, -1) +
      c(4) * input(-2, 0) + c(5) * input(-1, 0) + c(6) * input(0, 0) + c(7) * input(1, 0) +
      c(8) * input(2, 0) + c(9) * input(-1, 1) + c(10) * input(0, 1) + c(11) * input(1, 1) +
      c(12) * input(0, 2);
}
