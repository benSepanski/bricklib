/**
 * @file transpose-cu.h
 * @author Ben Sepanski (ben_sepanski@utexas.edu)
 * @brief batched matrix transposes
 * @version 0.1
 * @date 2021-07-23
 * 
 * @copyright Copyright (c) 2021
 */
#include <cassert>
#include <cuda_runtime.h>

/**
 * @brief flatten index [k][j][i]
 * 
 * @param k index in outermost dimension (least contiguous)
 * @param j index in middle dimension
 * @param i index in innermost dimension (most contiguous)
 * @param J extent in J direction
 * @param I extent in I direction
 * @return the flattened index
 */
__device__ __forceinline__
size_t flatten_index(size_t k, size_t j, size_t i, size_t J, size_t I)
{
  return i + I * (j + J * k);
}

/**
 * @brief transpose the i-j dimensions of in_mat and store in out_mat
 * 
 * Each thread-block process one tile of size [TileJDim][TileIDim]
 * 
 * @tparam elemType the type of elements in the matrix
 * @tparam TileJDim the j-extent of the shared-memory tiles used 
 * @tparam TileIDim the i-extent of the shared-memory tiles used 
 * @param[in] in_mat pointer to array of shape [gridDim.z][J][I]
 * @param[out] out_mat pointer to array of shape [gridDim.z][I][J]
 * @param[in] J an extent of the matrices
 * @param[in] I an extent of the matrices
 */
// Based on https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/
template<unsigned TileJDim, unsigned TileIDim, typename elemType>
__global__
void transpose_ij(elemType * __restrict__ in_mat, elemType * __restrict__ out_mat, const size_t J, const size_t I)
{
  assert(TileJDim * gridDim.y >= J);
  assert(TileIDim * gridDim.x >= J);

  // avoid bank conflicts
  static constexpr unsigned WARP_SIZE = 32;
  static constexpr unsigned BANK_SHIFT = (1 - (TileJDim % WARP_SIZE) + WARP_SIZE) % WARP_SIZE;
  __shared__ elemType tile[TileJDim][TileIDim + BANK_SHIFT];
  // find tile
  size_t k = blockDim.z * blockIdx.z + threadIdx.z;
  const size_t tile_j_start_idx = min((size_t) TileJDim * blockIdx.y, J);
  const size_t tile_i_start_idx = min((size_t) TileIDim * blockIdx.x, I);

  // get upper bounds on tile
  const size_t upper_bound_j = min((size_t) TileJDim, J - tile_j_start_idx),
               upper_bound_i = min((size_t) TileIDim, I - tile_i_start_idx);

  // load tile into shared memory
  for(size_t j = threadIdx.y; j < upper_bound_j; j += blockDim.y)
  {
    for(size_t i = threadIdx.x; i < upper_bound_i; i += blockDim.x)
    {
      tile[j][i] = in_mat[flatten_index(k, tile_j_start_idx + j, tile_i_start_idx + i, J, I)];
      printf("tile[%lu][%lu] = %d\n", j, i, tile[j][i]);
    }
  }
  __syncthreads();
  // store transposed tile
  for(size_t i = threadIdx.y; i < upper_bound_i; i += blockDim.y)
  {
    for(size_t j = threadIdx.x; j < upper_bound_j; j += blockDim.x)
    {
      out_mat[flatten_index(k, tile_i_start_idx + i, tile_j_start_idx + j, I, J)] = tile[j][i];
    }
  }
}