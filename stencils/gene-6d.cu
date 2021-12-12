#include "assert.h"
#include "gene-6d-stencils.h"

constexpr unsigned GB_i = GHOST_ZONE_i / BDIM_i;
constexpr unsigned GB_j = GHOST_ZONE_j / BDIM_j;
constexpr unsigned GB_k = GHOST_ZONE_k / BDIM_k;
constexpr unsigned GB_l = GHOST_ZONE_l / BDIM_l;
constexpr unsigned GB_m = GHOST_ZONE_m / BDIM_m;
constexpr unsigned GB_n = GHOST_ZONE_n / BDIM_n;

__constant__ bElem const_i_deriv_coeff_dev[5];
__constant__ unsigned brick_grid_extents[DIM] = {BRICK_EXTENT};
__constant__ unsigned brick_grid_strides[DIM] = {1,
                                                 GZ_BRICK_EXTENT_i,
                                                 GZ_BRICK_EXTENT_i*GZ_BRICK_EXTENT_j,
                                                 GZ_BRICK_EXTENT_i*GZ_BRICK_EXTENT_j*GZ_BRICK_EXTENT_k,
                                                 GZ_BRICK_EXTENT_i*GZ_BRICK_EXTENT_j*GZ_BRICK_EXTENT_k*GZ_BRICK_EXTENT_l,
                                                 GZ_BRICK_EXTENT_i*GZ_BRICK_EXTENT_j*GZ_BRICK_EXTENT_k*GZ_BRICK_EXTENT_l*GZ_BRICK_EXTENT_m
                                                 };
__constant__ unsigned brick_grid_ghost_zones[DIM] = {GB_i, GB_j, GB_k, GB_l, GB_m, GB_n};
__constant__ char grid_iteration_order[DIM+1];

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
  gpuCheck(cudaMemcpyToSymbol(const_i_deriv_coeff_dev, i_deriv_coeff_host, 5 * sizeof(bElem)));
}

/**
 * @brief copy gird_iteration_order to constant memory
 */
__host__
void copy_grid_iteration_order(const char * grid_iteration_order_host)
{
  gpuCheck(cudaMemcpyToSymbol(grid_iteration_order, grid_iteration_order_host, (DIM+1) * sizeof(char)));
}

/**
 * @brief Compute on the non-ghost bricks
 */
__launch_bounds__(NUM_ELEMENTS_PER_BRICK, max_blocks_per_sm(NUM_ELEMENTS_PER_BRICK))
__global__ void
ij_deriv_brick_kernel(unsigned (*fieldGrid)[GZ_BRICK_EXTENT_m][GZ_BRICK_EXTENT_l][GZ_BRICK_EXTENT_k][GZ_BRICK_EXTENT_j][GZ_BRICK_EXTENT_i],
                      unsigned (*coeffGrid)[GZ_BRICK_EXTENT_m][GZ_BRICK_EXTENT_l][GZ_BRICK_EXTENT_k][GZ_BRICK_EXTENT_i],
                      FieldBrick_i bIn,
                      FieldBrick_i bOut,
                      PreCoeffBrick bP1,
                      PreCoeffBrick bP2,
                      bComplexElem *ikj) 
{
  // compute brick index
  unsigned tn = GB_n + blockIdx.z / (BRICK_EXTENT_k * BRICK_EXTENT_l * BRICK_EXTENT_m);
  unsigned tm = GB_m + blockIdx.z / (BRICK_EXTENT_k * BRICK_EXTENT_l) % BRICK_EXTENT_m;
  unsigned tl = GB_l + (blockIdx.z / BRICK_EXTENT_k) % BRICK_EXTENT_l;
  unsigned tk = GB_k + blockIdx.z % BRICK_EXTENT_k;
  unsigned tj = GB_j + blockIdx.y;
  unsigned ti = GB_i + blockIdx.x;

  // bounds check
  assert(ti < GZ_BRICK_EXTENT_i);
  assert(tj < GZ_BRICK_EXTENT_j);
  assert(tk < GZ_BRICK_EXTENT_k);
  assert(tl < GZ_BRICK_EXTENT_l);
  assert(tm < GZ_BRICK_EXTENT_m);
  assert(tn < GZ_BRICK_EXTENT_n);

  // get brick indices
  unsigned bFieldIndex = fieldGrid[tn][tm][tl][tk][tj][ti];
  unsigned bCoeffIndex = coeffGrid[tn][tm][tl][tk][ti];

  unsigned n = threadIdx.z / (BDIM_k * BDIM_l * BDIM_m);
  unsigned m = threadIdx.z / (BDIM_k * BDIM_l) % BDIM_m;
  unsigned l = (threadIdx.z / BDIM_k) % BDIM_l;
  unsigned k = threadIdx.z % BDIM_k;
  unsigned j = threadIdx.y;
  unsigned i = threadIdx.x;

  // bounds check
  assert(i < BDIM_i);
  assert(j < BDIM_j);
  assert(k < BDIM_k);
  assert(l < BDIM_l);
  assert(m < BDIM_m);
  assert(n < BDIM_n);

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
__launch_bounds__(IJ_DERIV_BRICK_KERNEL_VEC_BLOCK_SIZE, max_blocks_per_sm(IJ_DERIV_BRICK_KERNEL_VEC_BLOCK_SIZE))
__global__ void
ij_deriv_brick_kernel_vec(unsigned (*fieldGrid)[GZ_BRICK_EXTENT_m][GZ_BRICK_EXTENT_l][GZ_BRICK_EXTENT_k][GZ_BRICK_EXTENT_j][GZ_BRICK_EXTENT_i],
                          unsigned (*coeffGrid)[GZ_BRICK_EXTENT_m][GZ_BRICK_EXTENT_l][GZ_BRICK_EXTENT_k][GZ_BRICK_EXTENT_i],
                          FieldBrick_i bIn,
                          FieldBrick_i bOut,
                          PreCoeffBrick bP1,
                          PreCoeffBrick bP2,
                          bComplexElem *ikj) 
{
  constexpr unsigned BLOCK_SIZE = IJ_DERIV_BRICK_KERNEL_VEC_BLOCK_SIZE;
  // compute brick index
  unsigned tn = GB_n + blockIdx.z / (BRICK_EXTENT_k * BRICK_EXTENT_l * BRICK_EXTENT_m);
  unsigned tm = GB_m + blockIdx.z / (BRICK_EXTENT_k * BRICK_EXTENT_l) % BRICK_EXTENT_m;
  unsigned tl = GB_l + (blockIdx.z / BRICK_EXTENT_k) % BRICK_EXTENT_l;
  unsigned tk = GB_k + blockIdx.z % BRICK_EXTENT_k;
  unsigned tj = GB_j + blockIdx.y;
  unsigned ti = GB_i + blockIdx.x;

  // bounds check
  assert(ti < GZ_BRICK_EXTENT_i);
  assert(tj < GZ_BRICK_EXTENT_j);
  assert(tk < GZ_BRICK_EXTENT_k);
  assert(tl < GZ_BRICK_EXTENT_l);
  assert(tm < GZ_BRICK_EXTENT_m);
  assert(tn < GZ_BRICK_EXTENT_n);

  // get brick indices
  unsigned bFieldIdx = fieldGrid[tn][tm][tl][tk][tj][ti];
  unsigned bCoeffIdx = coeffGrid[tn][tm][tl][tk][ti];

  // output buffer (corresponds to segments of length BDIM_i
  // at strides of block size elements inside the brick)
  assert(BLOCK_SIZE == blockDim.x);
  static_assert(NUM_ELEMENTS_PER_BRICK % BLOCK_SIZE == 0);
  bComplexElem outputBuf[NUM_ELEMENTS_PER_BRICK / BLOCK_SIZE];
  bComplexElem yDerivBuf[NUM_ELEMENTS_PER_BRICK / BLOCK_SIZE];

  unsigned leftNeighbor = bIn.bInfo->adj[bFieldIdx][0];
  unsigned rightNeighbor = bIn.bInfo->adj[bFieldIdx][2];

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
    outputBuf[vecIdx] = result;
  }

  for(unsigned vecIdx = 0; vecIdx < NUM_ELEMENTS_PER_BRICK / BLOCK_SIZE; vecIdx++)
  {
    unsigned flatBrickIdx = vecIdx * BLOCK_SIZE + threadIdx.x;
    // add y-deriv term
    unsigned j = flatBrickIdx / BDIM_i % BDIM_j;
    bComplexElem my_ikj = ikj[PADDING_j + BDIM_j * tj + j];
    yDerivBuf[vecIdx] = bIn.dat[bFieldIdx * bIn.step + flatBrickIdx] * my_ikj;
  }

  // now write out all the results to the actual output
  for(unsigned vecIdx = 0; vecIdx < NUM_ELEMENTS_PER_BRICK / BLOCK_SIZE; vecIdx++)
  {
    unsigned flatBrickIdx = vecIdx * BLOCK_SIZE + threadIdx.x;
    // grab p1 and p2
    unsigned flatBrickIdxNoJ = (flatBrickIdx / (BDIM_i * BDIM_j)) * BDIM_i + flatBrickIdx % BDIM_i;
    bComplexElem p1 = bP1.dat[bCoeffIdx * bP1.step + flatBrickIdxNoJ];
    bComplexElem p2 = bP2.dat[bCoeffIdx * bP2.step + flatBrickIdxNoJ];
    bOut.dat[bFieldIdx * bIn.step + flatBrickIdx] = p1 * outputBuf[vecIdx] + p2 * yDerivBuf[vecIdx];
  }
}

/**
 * @brief Compute arakawa on the non-ghost bricks
 * 
 * @param coeff an array of ARAKAWA_STENCIL_SIZE RealCoeffBrick s
 */
__launch_bounds__(NUM_ELEMENTS_PER_BRICK, max_blocks_per_sm(NUM_ELEMENTS_PER_BRICK))
__global__ void
semi_arakawa_brick_kernel(unsigned *fieldGrid,
                          unsigned *coeffGrid,
                          FieldBrick_kl bIn,
                          FieldBrick_kl bOut,
                          RealCoeffBrick *coeff)
{
  // compute indices
  unsigned bFieldIndex, bCoeffIndex;
  {
    unsigned d0 = grid_iteration_order[0] - 'i';
    unsigned fieldGridIndex = (brick_grid_ghost_zones[d0] + (blockIdx.x % brick_grid_extents[d0])) * brick_grid_strides[d0];
    unsigned d1 = grid_iteration_order[1] - 'i';
    fieldGridIndex += (brick_grid_ghost_zones[d1] + (blockIdx.y % brick_grid_extents[d1])) * brick_grid_strides[d1];
    unsigned idx = blockIdx.z;
    for(unsigned dim_idx = 2; dim_idx < DIM; ++dim_idx)
    {
      unsigned d = grid_iteration_order[dim_idx] - 'i';
      fieldGridIndex += (brick_grid_ghost_zones[d] + (idx % brick_grid_extents[d])) * brick_grid_strides[d];
      idx /= brick_grid_extents[d];
    }
    bFieldIndex = ((unsigned *)fieldGrid)[fieldGridIndex];
    unsigned coeffGridIndex = fieldGridIndex / (GZ_BRICK_EXTENT_i * GZ_BRICK_EXTENT_j) * GZ_BRICK_EXTENT_i
                            + fieldGridIndex % GZ_BRICK_EXTENT_i;
    bCoeffIndex = ((unsigned *)coeffGrid)[coeffGridIndex];
  }

  // put neighbors in shared memory
  // constexpr unsigned NUM_NEIGHBORS = static_power<3, FieldBrick_kl::myCommDims::numCommunicatingDims(DIM)>::value;
  // static_assert(NUM_NEIGHBORS < NUM_ELEMENTS_PER_BRICK);
  // __shared__ unsigned neighbors[NUM_NEIGHBORS];
  // unsigned idx = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
  // if(idx < NUM_NEIGHBORS)
  // {
    // neighbors[idx] = bIn.bInfo->adj[bFieldIndex][idx];
  // }
  // __syncthreads();
  unsigned *neighbors = bIn.bInfo->adj[bFieldIndex];

  int n = threadIdx.z / (BDIM_k *  BDIM_l  * BDIM_m);
  int m = threadIdx.z / (BDIM_k *  BDIM_l) % BDIM_m;
  int l = threadIdx.z /  BDIM_k % BDIM_l;
  int k = threadIdx.z %  BDIM_k;
  int j = threadIdx.y;
  int i = threadIdx.x;

  // bounds check
  assert(i < BDIM_i);
  assert(j < BDIM_j);
  assert(k < BDIM_k);
  assert(l < BDIM_l);
  assert(m < BDIM_m);
  assert(n < BDIM_n);

  // load in data
  bComplexElem in[13], result = 0.0;
  auto myIndex = BrickIndex<FieldBrick_kl>(n, m, l, k, j, i);
  unsigned flatIndexNoJ = i + BDIM_i * (k + BDIM_k * (l + BDIM_l * (m + BDIM_m * n)));

  myIndex.shiftInDims<3>(-2);
  in[0] = bIn.dat[bIn.step * neighbors[myIndex.indexInNbrList] + myIndex.getIndexInBrick()];
  bElem my_coeff = coeff[0].dat[coeff[0].step * bCoeffIndex + flatIndexNoJ];
  result += in[0] * my_coeff;

  myIndex.shiftInDims<3, 2>(+1, -1);
  in[1] = bIn.dat[bIn.step * neighbors[myIndex.indexInNbrList] + myIndex.getIndexInBrick()];
  my_coeff = coeff[1].dat[coeff[1].step * bCoeffIndex + flatIndexNoJ];
  result += in[1] * my_coeff;

  myIndex.shiftInDims<2>(+1);
  in[2] = bIn.dat[bIn.step * neighbors[myIndex.indexInNbrList] + myIndex.getIndexInBrick()];
  my_coeff = coeff[2].dat[coeff[2].step * bCoeffIndex + flatIndexNoJ];
  result += in[2] * my_coeff;

  myIndex.shiftInDims<2>(+1);
  in[3] = bIn.dat[bIn.step * neighbors[myIndex.indexInNbrList] + myIndex.getIndexInBrick()];
  my_coeff = coeff[3].dat[coeff[3].step * bCoeffIndex + flatIndexNoJ];
  result += in[3] * my_coeff;

  myIndex.shiftInDims<3, 2>(+1, -3);
  in[4] = bIn.dat[bIn.step * neighbors[myIndex.indexInNbrList] + myIndex.getIndexInBrick()];
  my_coeff = coeff[4].dat[coeff[4].step * bCoeffIndex + flatIndexNoJ];
  result += in[4] * my_coeff;

  myIndex.shiftInDims<2>(+1);
  in[5] = bIn.dat[bIn.step * neighbors[myIndex.indexInNbrList] + myIndex.getIndexInBrick()];
  my_coeff = coeff[5].dat[coeff[5].step * bCoeffIndex + flatIndexNoJ];
  result += in[5] * my_coeff;

  myIndex.shiftInDims<2>(+1);
  in[6] = bIn.dat[bIn.step * neighbors[myIndex.indexInNbrList] + myIndex.getIndexInBrick()];
  my_coeff = coeff[6].dat[coeff[6].step * bCoeffIndex + flatIndexNoJ];
  result += in[6] * my_coeff;

  myIndex.shiftInDims<2>(+1);
  in[7] = bIn.dat[bIn.step * neighbors[myIndex.indexInNbrList] + myIndex.getIndexInBrick()];
  my_coeff = coeff[7].dat[coeff[7].step * bCoeffIndex + flatIndexNoJ];
  result += in[7] * my_coeff;

  myIndex.shiftInDims<2>(+1);
  in[8] = bIn.dat[bIn.step * neighbors[myIndex.indexInNbrList] + myIndex.getIndexInBrick()];
  my_coeff = coeff[8].dat[coeff[8].step * bCoeffIndex + flatIndexNoJ];
  result += in[8] * my_coeff;

  myIndex.shiftInDims<3, 2>(+1, -3);
  in[9] = bIn.dat[bIn.step * neighbors[myIndex.indexInNbrList] + myIndex.getIndexInBrick()];
  my_coeff = coeff[9].dat[coeff[9].step * bCoeffIndex + flatIndexNoJ];
  result += in[9] * my_coeff;

  myIndex.shiftInDims<2>(+1);
  in[10] = bIn.dat[bIn.step * neighbors[myIndex.indexInNbrList] + myIndex.getIndexInBrick()];
  my_coeff = coeff[10].dat[coeff[10].step * bCoeffIndex + flatIndexNoJ];
  result += in[10] * my_coeff;

  myIndex.shiftInDims<2>(+1);
  in[11] = bIn.dat[bIn.step * neighbors[myIndex.indexInNbrList] + myIndex.getIndexInBrick()];
  my_coeff = coeff[11].dat[coeff[11].step * bCoeffIndex + flatIndexNoJ];
  result += in[11] * my_coeff;

  myIndex.shiftInDims<3, 2>(+1, -1);
  in[12] = bIn.dat[bIn.step * neighbors[myIndex.indexInNbrList] + myIndex.getIndexInBrick()];
  my_coeff = coeff[12].dat[coeff[12].step * bCoeffIndex + flatIndexNoJ];
  result += in[12] * my_coeff;

  // store result
  unsigned flatIndex = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
  bOut.dat[bFieldIndex * bOut.step + flatIndex] = result;
}

/**
 * @brief Compute arakawa on the non-ghost bricks
 * 
 * @param coeff an array of ARAKAWA_STENCIL_SIZE RealCoeffBrick s
 * @param grid_iteration_order "ijklmn" for contiguous
 */
__launch_bounds__(SEMI_ARAKAWA_BRICK_KERNEL_VEC_BLOCK_SIZE)
__global__ void
semi_arakawa_brick_kernel_vec(unsigned *fieldGrid,
                              unsigned *coeffGrid,
                              FieldBrick_kl bIn,
                              FieldBrick_kl bOut,
                              RealCoeffBrick *coeff
                              )
{
  // build our output buffer and input buffer
  constexpr unsigned BLOCK_SIZE = SEMI_ARAKAWA_BRICK_KERNEL_VEC_BLOCK_SIZE;
  bComplexElem outputBuf[NUM_ELEMENTS_PER_BRICK / BLOCK_SIZE];
  bComplexElem in[ARAKAWA_STENCIL_SIZE];

  // move grid-info into shared memory
  __shared__ unsigned extent[DIM],
                      stride[DIM],
                      ghost[DIM];
  if(threadIdx.x < DIM)
  {
    unsigned my_d = grid_iteration_order[threadIdx.x] - 'i';
    assert(my_d < DIM);
    extent[threadIdx.x] = brick_grid_extents[my_d];
    ghost[threadIdx.x] = brick_grid_ghost_zones[my_d];
    unsigned my_stride = 1;
    for(unsigned d = 0; d < my_d; ++d)
    {
      my_stride *= brick_grid_extents[d] + 2 * brick_grid_ghost_zones[d];
    }
    stride[threadIdx.x] = my_stride;
  }
  __syncthreads();

  // For each brick
  for(unsigned idx = blockIdx.x; idx < NUM_BRICKS ; idx += gridDim.x)
  {
    // // compute indices
    unsigned bFieldIndex, bCoeffIndex;
    {
      unsigned fieldGridIndex = 0;
      unsigned copy_of_idx = idx;
      for(unsigned d = 0; d < DIM; ++d)
      {
        fieldGridIndex += (ghost[d] + (copy_of_idx % extent[d])) * stride[d];
        copy_of_idx /= extent[d];
      }
      bFieldIndex = fieldGrid[fieldGridIndex];
      unsigned coeffGridIndex = fieldGridIndex / (GZ_BRICK_EXTENT_i * GZ_BRICK_EXTENT_j) * GZ_BRICK_EXTENT_i
                              + fieldGridIndex % GZ_BRICK_EXTENT_i;
      bCoeffIndex = coeffGrid[coeffGridIndex];
    }

    // currently assume that block size is brick size
    static_assert(NUM_ELEMENTS_PER_BRICK % BLOCK_SIZE == 0);
    static_assert(BLOCK_SIZE % WARP_SIZE == 0);

    // compute for each warp (assume bricks stored as scalar)
    unsigned n, m, l, k, j, i;
    for(unsigned vec_idx = 0; vec_idx < NUM_ELEMENTS_PER_BRICK / BLOCK_SIZE; vec_idx++)
    {
      unsigned flatIdxInBrick = vec_idx * BLOCK_SIZE + threadIdx.x;

      n = flatIdxInBrick / (BDIM_i * BDIM_j  * BDIM_k *  BDIM_l  * BDIM_m);
      m = flatIdxInBrick / (BDIM_i * BDIM_j  * BDIM_k *  BDIM_l) % BDIM_m;
      l = flatIdxInBrick / (BDIM_i * BDIM_j  * BDIM_k) % BDIM_l;
      k = flatIdxInBrick / (BDIM_i * BDIM_j) % BDIM_k;
      j = flatIdxInBrick /  BDIM_i % BDIM_j;
      i = flatIdxInBrick %  BDIM_i;

      // bounds check
      assert(i < BDIM_i);
      assert(j < BDIM_j);
      assert(k < BDIM_k);
      assert(l < BDIM_l);
      assert(m < BDIM_m);
      assert(n < BDIM_n);

      // load in data
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

      outputBuf[vec_idx] = 0.0;
      for(unsigned stencil_index = 0; stencil_index < 13; ++stencil_index) 
      {
        bElem my_coeff = coeff[stencil_index][bCoeffIndex][n][m][l][k][i];
        outputBuf[vec_idx] += my_coeff * in[stencil_index];
      }
    }

    // store data out from buffer
    #pragma unroll
    for(unsigned vec_idx = 0; vec_idx < NUM_ELEMENTS_PER_BRICK / BLOCK_SIZE; vec_idx++)
    {
      unsigned flatIdxInBrick = vec_idx * BLOCK_SIZE + threadIdx.x;
      bOut.dat[bFieldIndex * bOut.step + flatIdxInBrick] = outputBuf[vec_idx];
    }
  }
}
