#include "gene-5d.h"
#include "assert.h"

constexpr unsigned GB_i = GHOST_ZONE_i / BDIM_i;
constexpr unsigned GB_j = GHOST_ZONE_j / BDIM_j;
constexpr unsigned GB_k = GHOST_ZONE_k / BDIM_k;
constexpr unsigned GB_l = GHOST_ZONE_l / BDIM_l;
constexpr unsigned GB_m = GHOST_ZONE_m / BDIM_m;
constexpr unsigned GB_n = GHOST_ZONE_n / BDIM_n;

__global__ void
ij_deriv_brick_kernel(unsigned (*fieldGrid)[GZ_BRICK_EXTENT_m][GZ_BRICK_EXTENT_l][GZ_BRICK_EXTENT_k][GZ_BRICK_EXTENT_j][GZ_BRICK_EXTENT_i],
                      unsigned (*coeffGrid)[GZ_BRICK_EXTENT_m][GZ_BRICK_EXTENT_l][GZ_BRICK_EXTENT_k][GZ_BRICK_EXTENT_i],
                      FieldBrick bIn,
                      FieldBrick bOut,
                      PreCoeffBrick bP1,
                      PreCoeffBrick bP2,
                      bComplexElem *ikj,
                      bElem *i_deriv_coeff) 
{
  // compute indices
  long tn = blockIdx.z / (GZ_BRICK_EXTENT_k * GZ_BRICK_EXTENT_l * GZ_BRICK_EXTENT_m);
  long tm = blockIdx.z / (GZ_BRICK_EXTENT_k * GZ_BRICK_EXTENT_l) % GZ_BRICK_EXTENT_m;
  long tl = (blockIdx.z / GZ_BRICK_EXTENT_k) % GZ_BRICK_EXTENT_l;
  long tk = blockIdx.z % GZ_BRICK_EXTENT_k;
  long tj = blockIdx.y;
  long ti = blockIdx.x;
  long n = threadIdx.z / (BDIM_k * BDIM_l * BDIM_m);
  long m = threadIdx.z / (BDIM_k * BDIM_l) % BDIM_m;
  long l = (threadIdx.z / BDIM_k) % BDIM_l;
  long k = threadIdx.z % BDIM_k;
  long j = threadIdx.y;
  long i = threadIdx.x;

  // bounds check
  assert(0 <= ti && ti < GZ_BRICK_EXTENT_i);
  assert(0 <= tj && tj < GZ_BRICK_EXTENT_j);
  assert(0 <= tk && tk < GZ_BRICK_EXTENT_k);
  assert(0 <= tl && tl < GZ_BRICK_EXTENT_l);
  assert(0 <= tm && tm < GZ_BRICK_EXTENT_m);
  assert(0 <= tn && tn < GZ_BRICK_EXTENT_n);
  assert(0 <= i && i < BDIM_i);
  assert(0 <= j && j < BDIM_j);
  assert(0 <= k && k < BDIM_k);
  assert(0 <= l && l < BDIM_l);
  assert(0 <= m && m < BDIM_m);
  assert(0 <= n && n < BDIM_n);

  unsigned bFieldIndex = fieldGrid[tn][tm][tl][tk][tj][ti];
  unsigned bCoeffIndex = coeffGrid[tn][tm][tl][tk][ti];

  // perform computation
  bOut[bFieldIndex][n][m][l][k][j][i] = bP1[bCoeffIndex][n][m][l][k][i] * (
    i_deriv_coeff[0] * bIn[bFieldIndex][n][m][l][k][j][i - 2] +
    i_deriv_coeff[1] * bIn[bFieldIndex][n][m][l][k][j][i - 1] +
    i_deriv_coeff[2] * bIn[bFieldIndex][n][m][l][k][j][i + 0] +
    i_deriv_coeff[3] * bIn[bFieldIndex][n][m][l][k][j][i + 1] +
    i_deriv_coeff[4] * bIn[bFieldIndex][n][m][l][k][j][i + 2]
  ) + 
  bP2[bCoeffIndex][n][m][l][k][i] * ikj[PADDING_j + tj * BDIM_j + j] * bIn[bFieldIndex][n][m][l][k][j][i];
}