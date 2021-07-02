#include "gene-6d.h"
#include "assert.h"

constexpr unsigned GB_i = GHOST_ZONE_i / BDIM_i;
constexpr unsigned GB_j = GHOST_ZONE_j / BDIM_j;
constexpr unsigned GB_k = GHOST_ZONE_k / BDIM_k;
constexpr unsigned GB_l = GHOST_ZONE_l / BDIM_l;
constexpr unsigned GB_m = GHOST_ZONE_m / BDIM_m;
constexpr unsigned GB_n = GHOST_ZONE_n / BDIM_n;

constexpr unsigned WARP_SIZE = 32;
constexpr unsigned MAX_BLOCK_SIZE = NUM_ELEMENTS_PER_BRICK;
constexpr unsigned MAX_WARPS_PER_SM = 64;
constexpr unsigned MIN_BLOCKS_PER_MULTIPROCESSOR = MAX_WARPS_PER_SM * WARP_SIZE / MAX_BLOCK_SIZE; 

__constant__ bElem const_i_deriv_coeff_dev[5];

/**
 * @brief convenience structure to convert to/from flat indices
 */
struct index6D
{
  unsigned i, j, k, l, m, n;

  __device__
  index6D(unsigned n, unsigned m, unsigned l, unsigned k, unsigned j, unsigned i)
  : i(i) , j(j) , k(k) , l(l) , m(m) , n(n) { }
};

/**
 * @brief Convert a flat index into a brick of dimensions BRICK_DIM
 *        into a index6D
 * 
 * @param idx the flattened index into the brick
 * @return the 6-dimensional index into the brick
 */
__device__ inline index6D
getIndexInsideBrick(unsigned idx)
{
  return index6D(idx / (BDIM_i * BDIM_j * BDIM_k * BDIM_l * BDIM_m),
                 idx / (BDIM_i * BDIM_j * BDIM_k * BDIM_l) % BDIM_m,
                 idx / (BDIM_i * BDIM_j * BDIM_k) % BDIM_l,
                 idx / (BDIM_i * BDIM_j) % BDIM_k,
                 idx / BDIM_i % BDIM_j,
                 idx % BDIM_i
                 );
}

/**
 * @brief copy i_deriv_coeff_host to constant memoty
 */
__host__
void copy_i_deriv_coeff(const bElem i_deriv_coeff_host[5])
{
  cudaCheck(cudaMemcpyToSymbol(const_i_deriv_coeff_dev, i_deriv_coeff_host, 5 * sizeof(bElem)));
}

/**
 * @brief Compute on the non-ghost bricks
 */
__launch_bounds__(MAX_BLOCK_SIZE, MIN_BLOCKS_PER_MULTIPROCESSOR)
__global__ void
ij_deriv_brick_kernel(unsigned (*fieldGrid)[GZ_BRICK_EXTENT_m][GZ_BRICK_EXTENT_l][GZ_BRICK_EXTENT_k][GZ_BRICK_EXTENT_j][GZ_BRICK_EXTENT_i],
                      unsigned (*coeffGrid)[GZ_BRICK_EXTENT_m][GZ_BRICK_EXTENT_l][GZ_BRICK_EXTENT_k][GZ_BRICK_EXTENT_i],
                      FieldBrick bIn,
                      FieldBrick bOut,
                      PreCoeffBrick bP1,
                      PreCoeffBrick bP2,
                      bComplexElem *ikj) 
{
  // compute brick index
  long tn = GB_n + blockIdx.z / (BRICK_EXTENT_k * BRICK_EXTENT_l * BRICK_EXTENT_m);
  long tm = GB_m + blockIdx.z / (BRICK_EXTENT_k * BRICK_EXTENT_l) % BRICK_EXTENT_m;
  long tl = GB_l + (blockIdx.z / BRICK_EXTENT_k) % BRICK_EXTENT_l;
  long tk = GB_k + blockIdx.z % BRICK_EXTENT_k;
  long tj = GB_j + blockIdx.y;
  long ti = GB_i + blockIdx.x;

  // bounds check
  assert(0 <= ti && ti < GZ_BRICK_EXTENT_i);
  assert(0 <= tj && tj < GZ_BRICK_EXTENT_j);
  assert(0 <= tk && tk < GZ_BRICK_EXTENT_k);
  assert(0 <= tl && tl < GZ_BRICK_EXTENT_l);
  assert(0 <= tm && tm < GZ_BRICK_EXTENT_m);
  assert(0 <= tn && tn < GZ_BRICK_EXTENT_n);

  // get brick indices
  unsigned bFieldIndex = fieldGrid[tn][tm][tl][tk][tj][ti];
  unsigned bCoeffIndex = coeffGrid[tn][tm][tl][tk][ti];

  long n = threadIdx.z / (BDIM_k * BDIM_l * BDIM_m);
  long m = threadIdx.z / (BDIM_k * BDIM_l) % BDIM_m;
  long l = (threadIdx.z / BDIM_k) % BDIM_l;
  long k = threadIdx.z % BDIM_k;
  long j = threadIdx.y;
  long i = threadIdx.x;

  // bounds check
  assert(0 <= i && i < BDIM_i);
  assert(0 <= j && j < BDIM_j);
  assert(0 <= k && k < BDIM_k);
  assert(0 <= l && l < BDIM_l);
  assert(0 <= m && m < BDIM_m);
  assert(0 <= n && n < BDIM_n);

  // load data together so that memory accesses get coalesced
  bComplexElem in[5];
  in[0] = bIn[bFieldIndex][n][m][l][k][j][i - 2];
  in[1] = bIn[bFieldIndex][n][m][l][k][j][i - 1];
  in[2] = bIn[bFieldIndex][n][m][l][k][j][i];
  in[3] = bIn[bFieldIndex][n][m][l][k][j][i + 1];
  in[4] = bIn[bFieldIndex][n][m][l][k][j][i + 2];

  bComplexElem p1 = bP1[bCoeffIndex][n][m][l][k][i],
               p2 = bP2[bCoeffIndex][n][m][l][k][i],
               my_ikj = ikj[PADDING_j + tj * BDIM_j + j];

  // perform computation
  bComplexElem out = p1 * (const_i_deriv_coeff_dev[0] * in[0] +
                           const_i_deriv_coeff_dev[1] * in[1] +
                           const_i_deriv_coeff_dev[2] * in[2] +
                           const_i_deriv_coeff_dev[3] * in[3] +
                           const_i_deriv_coeff_dev[4] * in[4]) + p2 * my_ikj * in[2];

  // store computation
  bOut[bFieldIndex][n][m][l][k][j][i] = out;
}

/**
 * @brief hand-vectorized i-j derivative kernel
 * 
 * Assumes block size is IJ_DERIV_BRICK_KERNEL_VEC_BLOCK_SIZE
 * 
 * Should be invoked as a 3-dimensional grid of 1D blocks
 */
__launch_bounds__(IJ_DERIV_BRICK_KERNEL_VEC_BLOCK_SIZE, MIN_BLOCKS_PER_MULTIPROCESSOR)
__global__ void
ij_deriv_brick_kernel_vec(unsigned (*fieldGrid)[GZ_BRICK_EXTENT_m][GZ_BRICK_EXTENT_l][GZ_BRICK_EXTENT_k][GZ_BRICK_EXTENT_j][GZ_BRICK_EXTENT_i],
                          unsigned (*coeffGrid)[GZ_BRICK_EXTENT_m][GZ_BRICK_EXTENT_l][GZ_BRICK_EXTENT_k][GZ_BRICK_EXTENT_i],
                          FieldBrick bIn,
                          FieldBrick bOut,
                          PreCoeffBrick bP1,
                          PreCoeffBrick bP2,
                          bComplexElem *ikj) 
{
  constexpr unsigned BLOCK_SIZE = IJ_DERIV_BRICK_KERNEL_VEC_BLOCK_SIZE;
  // compute brick index
  long tn = GB_n + blockIdx.z / (BRICK_EXTENT_k * BRICK_EXTENT_l * BRICK_EXTENT_m);
  long tm = GB_m + blockIdx.z / (BRICK_EXTENT_k * BRICK_EXTENT_l) % BRICK_EXTENT_m;
  long tl = GB_l + (blockIdx.z / BRICK_EXTENT_k) % BRICK_EXTENT_l;
  long tk = GB_k + blockIdx.z % BRICK_EXTENT_k;
  long tj = GB_j + blockIdx.y;
  long ti = GB_i + blockIdx.x;

  // bounds check
  assert(0 <= ti && ti < GZ_BRICK_EXTENT_i);
  assert(0 <= tj && tj < GZ_BRICK_EXTENT_j);
  assert(0 <= tk && tk < GZ_BRICK_EXTENT_k);
  assert(0 <= tl && tl < GZ_BRICK_EXTENT_l);
  assert(0 <= tm && tm < GZ_BRICK_EXTENT_m);
  assert(0 <= tn && tn < GZ_BRICK_EXTENT_n);

  // get brick indices
  unsigned bFieldIdx = fieldGrid[tn][tm][tl][tk][tj][ti];
  unsigned bCoeffIdx = coeffGrid[tn][tm][tl][tk][ti];

  // output buffer (corresponds to segments of length BDIM_i
  // at strides of block size elements inside the brick)
  assert(BLOCK_SIZE == blockDim.x);
  static_assert(NUM_ELEMENTS_PER_BRICK % BLOCK_SIZE == 0);
  bComplexElem outputBuf[NUM_ELEMENTS_PER_BRICK / BLOCK_SIZE];

  unsigned leftNeighbor = bIn.bInfo->adj[bFieldIdx][CENTER_OFFSET_6D - 1];
  unsigned rightNeighbor = bIn.bInfo->adj[bFieldIdx][CENTER_OFFSET_6D + 1];

  // perform this operation for each "vector"
  for(unsigned vecIdx = 0; vecIdx < NUM_ELEMENTS_PER_BRICK / BLOCK_SIZE; vecIdx++)
  {
    unsigned flatBrickIdx = vecIdx * BLOCK_SIZE + threadIdx.x;
    bComplexElem result;
    // get value in block to left and in this block
    bComplexElem fieldValueInLeftBlock = bIn.dat[leftNeighbor * bIn.step + flatBrickIdx];
    bComplexElem fieldValueInBlock = bIn.dat[bFieldIdx * bIn.step + flatBrickIdx];
    // += coeff[0] * [...][i-2]
    bComplexElem shiftedValue;
    dev_shl_cplx(shiftedValue, fieldValueInLeftBlock, fieldValueInBlock, 2, BDIM_i, threadIdx.x % BDIM_i);
    result = const_i_deriv_coeff_dev[0] * shiftedValue;
    // += coeff[1] * [...][i-1]
    dev_shl_cplx(shiftedValue, fieldValueInLeftBlock, fieldValueInBlock, 1, BDIM_i, threadIdx.x % BDIM_i);
    result += const_i_deriv_coeff_dev[1] * shiftedValue;
    // grab value from right
    bComplexElem fieldValueInRightBlock = bIn.dat[rightNeighbor * bIn.step + flatBrickIdx];
    // += coeff[2] * [...][i]
    result += const_i_deriv_coeff_dev[2] * fieldValueInBlock;
    // += coeff[3] * [...][i+1]
    dev_shl_cplx(shiftedValue, fieldValueInBlock, fieldValueInRightBlock, BDIM_i - 1, BDIM_i, threadIdx.x % BDIM_i);
    result += const_i_deriv_coeff_dev[3] * shiftedValue;
    // += coeff[4] * [...][i+2]
    dev_shl_cplx(shiftedValue, fieldValueInBlock, fieldValueInRightBlock, BDIM_i - 2, BDIM_i, threadIdx.x % BDIM_i);
    result += const_i_deriv_coeff_dev[4] * shiftedValue;
    // grab p1 and p2
    unsigned flatBrickIdxNoJ = (flatBrickIdx / (BDIM_i * BDIM_j)) * BDIM_i + flatBrickIdx % BDIM_i;
    unsigned flatBrickIdxJ = flatBrickIdx / BDIM_i % BDIM_j;
    bComplexElem p1 = bP1.dat[bCoeffIdx * bIn.step + flatBrickIdxNoJ];
    result *= p1;
    // add y-deriv term
    bComplexElem yDerivTerm = fieldValueInBlock;
    bComplexElem p2 = bP2.dat[bCoeffIdx * bIn.step + flatBrickIdxNoJ];
    yDerivTerm *= p2;
    bComplexElem my_ikj = ikj[PADDING_j + BDIM_j * tj + flatBrickIdxJ];
    result += yDerivTerm * my_ikj;
    // store to buffer
    outputBuf[vecIdx] = result;
  }

  // now write out all the results to the actual output
  for(unsigned vecIdx = 0; vecIdx < NUM_ELEMENTS_PER_BRICK / BLOCK_SIZE; vecIdx++)
  {
    unsigned flatBrickIdx = vecIdx * BLOCK_SIZE + threadIdx.x;
    bOut.dat[bFieldIdx * bIn.step + flatBrickIdx] = outputBuf[vecIdx];
  }
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

  // bounds check
  assert(0 <= ti && ti < GZ_BRICK_EXTENT_i);
  assert(0 <= tj && tj < GZ_BRICK_EXTENT_j);
  assert(0 <= tk && tk < GZ_BRICK_EXTENT_k);
  assert(0 <= tl && tl < GZ_BRICK_EXTENT_l);
  assert(0 <= tm && tm < GZ_BRICK_EXTENT_m);
  assert(0 <= tn && tn < GZ_BRICK_EXTENT_n);

  unsigned bFieldIndex = fieldGrid[tn][tm][tl][tk][tj][ti];
  unsigned bCoeffIndex = coeffGrid[tn][tm][tl][tk][ti];

  long n = threadIdx.z / (BDIM_k * BDIM_l * BDIM_m);
  long m = threadIdx.z / (BDIM_k * BDIM_l) % BDIM_m;
  long l = (threadIdx.z / BDIM_k) % BDIM_l;
  long k = threadIdx.z % BDIM_k;
  long j = threadIdx.y;
  long i = threadIdx.x;

  // bounds check
  assert(0 <= i && i < BDIM_i);
  assert(0 <= j && j < BDIM_j);
  assert(0 <= k && k < BDIM_k);
  assert(0 <= l && l < BDIM_l);
  assert(0 <= m && m < BDIM_m);
  assert(0 <= n && n < BDIM_n);

  // load in data
  bComplexElem in[13];
  in[ 0] = bIn[bFieldIndex][n][m][l-2][k+0][j][i];
  in[ 1] = bIn[bFieldIndex][n][m][l-1][k-1][j][i];
  in[ 2] = bIn[bFieldIndex][n][m][l-1][k+0][j][i];
  in[ 3] = bIn[bFieldIndex][n][m][l-1][k+1][j][i];
  in[ 4] = bIn[bFieldIndex][n][m][l+0][k-2][j][i];
  in[ 5] = bIn[bFieldIndex][n][m][l+0][k-1][j][i];
  in[ 6] = bIn[bFieldIndex][n][m][l+0][k+0][j][i];
  in[ 7] = bIn[bFieldIndex][n][m][l+0][k+1][j][i];
  in[ 8] = bIn[bFieldIndex][n][m][l+0][k+2][j][i];
  in[ 9] = bIn[bFieldIndex][n][m][l+1][k-1][j][i];
  in[10] = bIn[bFieldIndex][n][m][l+1][k+0][j][i];
  in[11] = bIn[bFieldIndex][n][m][l+1][k+1][j][i];
  in[12] = bIn[bFieldIndex][n][m][l+2][k+0][j][i];

  bElem myCoeff[13];
  for(unsigned stencil_index = 0; stencil_index < 13; ++stencil_index) 
  {
    myCoeff[stencil_index] = coeff[stencil_index][bCoeffIndex][n][m][l][k][i];
  }

  // perform computation
  bOut[bFieldIndex][n][m][l][k][j][i] = myCoeff[ 0] * in[ 0] +
                                        myCoeff[ 1] * in[ 1] +
                                        myCoeff[ 2] * in[ 2] +
                                        myCoeff[ 3] * in[ 3] +
                                        myCoeff[ 4] * in[ 4] +
                                        myCoeff[ 5] * in[ 5] +
                                        myCoeff[ 6] * in[ 6] +
                                        myCoeff[ 7] * in[ 7] +
                                        myCoeff[ 8] * in[ 8] +
                                        myCoeff[ 9] * in[ 9] +
                                        myCoeff[10] * in[10] +
                                        myCoeff[11] * in[11] +
                                        myCoeff[12] * in[12];
}
