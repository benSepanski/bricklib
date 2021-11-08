#include "gene-6d.h"
#include "assert.h"

__constant__ bElem const_i_deriv_coeff_dev[5];
__constant__ unsigned brick_grid_extents[RANK] =
    {BRICK_GRID_EXTENT[0], BRICK_GRID_EXTENT[1], BRICK_GRID_EXTENT[2],
     BRICK_GRID_EXTENT[3], BRICK_GRID_EXTENT[4], BRICK_GRID_EXTENT[5]};
__constant__ unsigned brick_grid_strides[RANK] =
    {1,
     BRICK_GRID_EXTENT_WITH_GZ[0],
     BRICK_GRID_EXTENT_WITH_GZ[0] * BRICK_GRID_EXTENT_WITH_GZ[1],
     BRICK_GRID_EXTENT_WITH_GZ[0] * BRICK_GRID_EXTENT_WITH_GZ[1] *
         BRICK_GRID_EXTENT_WITH_GZ[2],
     BRICK_GRID_EXTENT_WITH_GZ[0] * BRICK_GRID_EXTENT_WITH_GZ[1] *
         BRICK_GRID_EXTENT_WITH_GZ[2] * BRICK_GRID_EXTENT_WITH_GZ[3],
     BRICK_GRID_EXTENT_WITH_GZ[0] * BRICK_GRID_EXTENT_WITH_GZ[1] *
         BRICK_GRID_EXTENT_WITH_GZ[2] * BRICK_GRID_EXTENT_WITH_GZ[3] *
         BRICK_GRID_EXTENT_WITH_GZ[4]};
__constant__ unsigned brick_grid_ghost_zones[RANK] =
    {GHOST_ZONE_BRICK[0], GHOST_ZONE_BRICK[1], GHOST_ZONE_BRICK[2],
     GHOST_ZONE_BRICK[3], GHOST_ZONE_BRICK[4], GHOST_ZONE_BRICK[5]};
__constant__ char grid_iteration_order[RANK+1];

/**
 * @brief copy i_deriv_coeff_host to constant memoty
 */
__host__
    void copy_i_deriv_coeff(const bElem i_deriv_coeff_host[5])
{
  cudaCheck(cudaMemcpyToSymbol(const_i_deriv_coeff_dev, i_deriv_coeff_host, 5 * sizeof(bElem)));
}

/**
 * @brief copy gird_iteration_order to constant memory
 */
__host__
    void copy_grid_iteration_order(const char * grid_iteration_order_host)
{
  cudaCheck(cudaMemcpyToSymbol(grid_iteration_order, grid_iteration_order_host, (RANK+1) * sizeof(char)));
}

__global__ __launch_bounds__(NUM_ELEMENTS_PER_BRICK, max_blocks_per_sm(NUM_ELEMENTS_PER_BRICK))
void semi_arakawa_brick_kernel(brick::Array<unsigned, RANK, brick::Padding<>, unsigned> fieldGrid,
                               brick::Array<unsigned, RANK-1, brick::Padding<>, unsigned> coeffGrid,
                               FieldBrick_kl bIn,
                               FieldBrick_kl bOut,
                               RealCoeffBrick *coeff) {
  // compute indices
  unsigned bFieldIndex, bCoeffIndex;
  {
    unsigned d0 = grid_iteration_order[0] - 'i';
    unsigned fieldGridIndex = (brick_grid_ghost_zones[d0] + (blockIdx.x % brick_grid_extents[d0])) * brick_grid_strides[d0];
    unsigned d1 = grid_iteration_order[1] - 'i';
    fieldGridIndex += (brick_grid_ghost_zones[d1] + (blockIdx.y % brick_grid_extents[d1])) * brick_grid_strides[d1];
    unsigned idx = blockIdx.z;
    for(unsigned dim_idx = 2; dim_idx < RANK; ++dim_idx)
    {
      unsigned d = grid_iteration_order[dim_idx] - 'i';
      fieldGridIndex += (brick_grid_ghost_zones[d] + (idx % brick_grid_extents[d])) * brick_grid_strides[d];
      idx /= brick_grid_extents[d];
    }
    bFieldIndex = fieldGrid.atFlatIndex(fieldGridIndex);
    unsigned coeffGridIndex = fieldGridIndex / (BRICK_GRID_EXTENT_WITH_GZ[0] * BRICK_GRID_EXTENT_WITH_GZ[1]) * BRICK_GRID_EXTENT_WITH_GZ[0]
                              + fieldGridIndex % BRICK_GRID_EXTENT_WITH_GZ[0];
    bCoeffIndex = coeffGrid.atFlatIndex(coeffGridIndex);
  }

  unsigned *neighbors = bIn.bInfo->adj[bFieldIndex];

  int n = threadIdx.z / (BDIM[2] *  BDIM[3]  * BDIM[4]);
  int m = threadIdx.z / (BDIM[2] *  BDIM[3]) % BDIM[4];
  int l = threadIdx.z /  BDIM[2] % BDIM[3];
  int k = threadIdx.z %  BDIM[2];
  int j = threadIdx.y;
  int i = threadIdx.x;

  // TODO: bounds check
//  assert(i < BDIM[0]);
//  assert(j < BDIM[1]);
//  assert(k < BDIM[2]);
//  assert(l < BDIM[3]);
//  assert(m < BDIM[4]);
//  assert(n < BDIM[5]);

  // load in data
  bComplexElem in[13], result = 0.0;
  auto myIndex = BrickIndex<FieldBrick_kl>(n, m, l, k, j, i);
  unsigned flatIndexNoJ = i + BDIM[0] * (k + BDIM[2] * (l + BDIM[3] * (m + BDIM[4] * n)));

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
//  result = coeff[1][bCoeffIndex][n][m][l][k][i];

  // store result
  unsigned flatIndex = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
  bOut.dat[bFieldIndex * bOut.step + flatIndex] = result;
}
