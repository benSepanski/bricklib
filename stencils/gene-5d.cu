#include "gene-5d.h"

constexpr unsigned GB_i = GHOST_ZONE_i / BDIM_i;
constexpr unsigned GB_j = GHOST_ZONE_j / BDIM_j;
constexpr unsigned GB_k = GHOST_ZONE_k / BDIM_k;
constexpr unsigned GB_l = GHOST_ZONE_l / BDIM_l;
constexpr unsigned GB_m = GHOST_ZONE_m / BDIM_m;

__global__ void
ij_deriv_brick_kernel(unsigned (*fieldGrid)[BRICK_EXTENT_l][BRICK_EXTENT_k][BRICK_EXTENT_j][BRICK_EXTENT_i],
                      unsigned (*coeffGrid)[BRICK_EXTENT_l][BRICK_EXTENT_k][BRICK_EXTENT_i],
                      FieldBrick bIn,
                      FieldBrick bOut,
                      PreCoeffBrick bP1,
                      PreCoeffBrick bP2,
                      bComplexElem *ikj,
                      bElem *i_deriv_coeff) 
{
  // compute indices
  long tm = GB_m + blockIdx.z / (BRICK_EXTENT_k * BRICK_EXTENT_l);
  long tl = GB_l + (blockIdx.z / BRICK_EXTENT_k) % BRICK_EXTENT_l;
  long tk = GB_k + blockIdx.z % BRICK_EXTENT_k;
  long tj = GB_j + blockIdx.y;
  long ti = GB_i + blockIdx.x;
  long m = threadIdx.z / (BDIM_k * BDIM_l);
  long l = (threadIdx.z / BDIM_k) % BDIM_l;
  long k = threadIdx.z % BDIM_k;
  long j = threadIdx.y;
  long i = threadIdx.x;
  unsigned bFieldIndex = fieldGrid[tm][tl][tk][tj][ti];
  unsigned bCoeffIndex = coeffGrid[tm][tl][tk][ti];
  // perform computation
  bOut[bFieldIndex][m][l][k][j][i] = bIn[bFieldIndex][m][l][k][j][i];
  
  // bP1[bCoeffIndex][m][l][k][i] * (
  //   i_deriv_coeff[0] * bIn[bFieldIndex][m][l][k][j][i - 2] +
  //   i_deriv_coeff[1] * bIn[bFieldIndex][m][l][k][j][i - 1] +
  //   i_deriv_coeff[2] * bIn[bFieldIndex][m][l][k][j][i + 0] +
  //   i_deriv_coeff[3] * bIn[bFieldIndex][m][l][k][j][i + 1] +
  //   i_deriv_coeff[4] * bIn[bFieldIndex][m][l][k][j][i + 2]
  // ) + 
  // bP2[bCoeffIndex][m][l][k][i] * ikj[PADDING_j + tj * BDIM_j + j] * bIn[bFieldIndex][m][l][k][j][i];
}