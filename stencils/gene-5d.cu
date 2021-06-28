#include "gene-5d.h"
#include "assert.h"

constexpr unsigned GB_i = GHOST_ZONE_i / BDIM_i;
constexpr unsigned GB_j = GHOST_ZONE_j / BDIM_j;
constexpr unsigned GB_k = GHOST_ZONE_k / BDIM_k;
constexpr unsigned GB_l = GHOST_ZONE_l / BDIM_l;
constexpr unsigned GB_m = GHOST_ZONE_m / BDIM_m;
constexpr unsigned GB_n = GHOST_ZONE_n / BDIM_n;

/**
 * @brief Compute on the non-ghost bricks
 */
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
  long tn = GB_n + blockIdx.z / (BRICK_EXTENT_k * BRICK_EXTENT_l * BRICK_EXTENT_m);
  long tm = GB_m + blockIdx.z / (BRICK_EXTENT_k * BRICK_EXTENT_l) % BRICK_EXTENT_m;
  long tl = GB_l + (blockIdx.z / BRICK_EXTENT_k) % BRICK_EXTENT_l;
  long tk = GB_k + blockIdx.z % BRICK_EXTENT_k;
  long tj = GB_j + blockIdx.y;
  long ti = GB_i + blockIdx.x;
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

/**
 * @brief Compute arakawa on the non-ghost bricks
 * 
 * @param coeff an array of ARAKAWA_STENCIL_SIZE RealCoeffBrick s
 */
__global__ void
semi_arakawa_brick_kernel(unsigned (*fieldGrid)[GZ_BRICK_EXTENT_m][GZ_BRICK_EXTENT_l][GZ_BRICK_EXTENT_k][GZ_BRICK_EXTENT_j][GZ_BRICK_EXTENT_i],
                          unsigned (*coeffGrid)[GZ_BRICK_EXTENT_m][GZ_BRICK_EXTENT_l][GZ_BRICK_EXTENT_k][GZ_BRICK_EXTENT_i],
                          FieldBrick bIn,
                          FieldBrick bOut,
                          RealCoeffBrick *coeff)
{
  // compute indices
  long tn = GB_n + blockIdx.z / (BRICK_EXTENT_k * BRICK_EXTENT_l * BRICK_EXTENT_m);
  long tm = GB_m + blockIdx.z / (BRICK_EXTENT_k * BRICK_EXTENT_l) % BRICK_EXTENT_m;
  long tl = GB_l + (blockIdx.z / BRICK_EXTENT_k) % BRICK_EXTENT_l;
  long tk = GB_k + blockIdx.z % BRICK_EXTENT_k;
  long tj = GB_j + blockIdx.y;
  long ti = GB_i + blockIdx.x;
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
  bOut[bFieldIndex][n][m][l][k][j][i] =
      coeff[ 0][bCoeffIndex][n][m][k][l][i] * bIn[bFieldIndex][n][m][l-2][k+0][j][i] +
      coeff[ 1][bCoeffIndex][n][m][k][l][i] * bIn[bFieldIndex][n][m][l-1][k-1][j][i] +
      coeff[ 2][bCoeffIndex][n][m][k][l][i] * bIn[bFieldIndex][n][m][l-1][k+0][j][i] +
      coeff[ 3][bCoeffIndex][n][m][k][l][i] * bIn[bFieldIndex][n][m][l-1][k+1][j][i] +
      coeff[ 4][bCoeffIndex][n][m][k][l][i] * bIn[bFieldIndex][n][m][l+0][k-2][j][i] +
      coeff[ 5][bCoeffIndex][n][m][k][l][i] * bIn[bFieldIndex][n][m][l+0][k-1][j][i] +
      coeff[ 6][bCoeffIndex][n][m][k][l][i] * bIn[bFieldIndex][n][m][l+0][k+0][j][i] +
      coeff[ 7][bCoeffIndex][n][m][k][l][i] * bIn[bFieldIndex][n][m][l+0][k+1][j][i] +
      coeff[ 8][bCoeffIndex][n][m][k][l][i] * bIn[bFieldIndex][n][m][l+0][k+2][j][i] +
      coeff[ 9][bCoeffIndex][n][m][k][l][i] * bIn[bFieldIndex][n][m][l+1][k-1][j][i] +
      coeff[10][bCoeffIndex][n][m][k][l][i] * bIn[bFieldIndex][n][m][l+1][k+0][j][i] +
      coeff[11][bCoeffIndex][n][m][k][l][i] * bIn[bFieldIndex][n][m][l+1][k+1][j][i] +
      coeff[12][bCoeffIndex][n][m][k][l][i] * bIn[bFieldIndex][n][m][l+2][k+0][j][i];
}
