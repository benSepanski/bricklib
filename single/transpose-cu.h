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
#include <numeric>
#include <vector>

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
 * Each thread-block process blockDim.z tiles of size [TileJDim][TileIDim]
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
void transpose_ij(const elemType * __restrict__ in_mat, elemType * __restrict__ out_mat, const size_t J, const size_t I)
{
  assert(TileJDim * gridDim.y >= J);
  assert(TileIDim * gridDim.x >= I);

  // avoid bank conflicts
  static constexpr unsigned BANK_SIZE = 32;
  static constexpr unsigned BANK_WIDTH_IN_BYTES = 32 / 4;
  /// which bank is (tile + TileJDim)?
  static constexpr unsigned BANK_OF_TileJDim = (TileJDim * sizeof(elemType) / BANK_WIDTH_IN_BYTES) % BANK_SIZE;
  // shift so that bank-shift of one row is one elemnt 
  static constexpr unsigned BANK_SHIFT = (sizeof(elemType) / BANK_WIDTH_IN_BYTES - BANK_OF_TileJDim + BANK_SIZE) % BANK_SIZE; 
  __shared__ elemType tile[TileJDim][TileIDim + BANK_SHIFT * BANK_WIDTH_IN_BYTES / sizeof(elemType)];
  // find tile
  size_t k = blockDim.z * blockIdx.z + threadIdx.z;
  const size_t tile_j_start_idx = min(((size_t) TileJDim) * blockIdx.y, J);
  const size_t tile_i_start_idx = min(((size_t) TileIDim) * blockIdx.x, I);

  // get upper bounds on tile
  const size_t upper_bound_j = min((size_t) TileJDim, J - tile_j_start_idx),
               upper_bound_i = min((size_t) TileIDim, I - tile_i_start_idx);

  // load tile into shared memory
  for(size_t j = threadIdx.y; j < upper_bound_j; j += blockDim.y)
  {
    for(size_t i = threadIdx.x; i < upper_bound_i; i += blockDim.x)
    {
      size_t index = flatten_index(k, tile_j_start_idx + j, tile_i_start_idx + i, J, I);
      assert(index < gridDim.z * blockDim.z * J * I);
      tile[j][i] = in_mat[index];
    }
  }
  __syncthreads();
  // store transposed tile
  for(size_t i = threadIdx.y; i < upper_bound_i; i += blockDim.y)
  {
    for(size_t j = threadIdx.x; j < upper_bound_j; j += blockDim.x)
    {
      size_t index = flatten_index(k, tile_i_start_idx + i, tile_j_start_idx + j, I, J);
      assert(index < gridDim.z * blockDim.z * J * I);
      out_mat[index] = tile[j][i];
    }
  }
}

namespace // anonymous namespace
{
  /**
   * @brief gcd from https://www.geeksforgeeks.org/euclidean-algorithms-basic-and-extended/
   * std::gcd not included until C++17
   */
  int gcd(int a, int b)
  {
      if (a == 0)
          return b;
      return gcd(b % a, a);
  }

  /**
   * @brief transpose in-brick into out-brick
   * 
   * @tparam BrickType 
   * @param[in] in_brick the brick
   * @param[out] out_brick the brick to transpose into
   */
  template<typename BrickType>
  __global__
  void transpose_brick_ij(const BrickType &in_brick, BrickType *out_brick)
  {
    constexpr unsigned BDIM_j = BrickType::template getBrickDim<1>(),
                       BDIM_i = BrickType::template getBrickDim<0>();
    
    for(unsigned flat_idx = threadIdx.x; flat_idx < BDIM_j * BDIM_i; flat_idx += blockDim.x)
    {
      unsigned i = flat_idx % BDIM_i;
      unsigned j = (i / BDIM_i) % BDIM_j;
      unsigned index = blockDim.x * out_brick->step + BDIM_i * j + i;
      unsigned index_tr = index + (BDIM_j - 1) * i
                                + (1 - BDIM_i) * j;
      unsigned val = out_brick->dat[index];
      out_brick->dat[index_tr] = val;
    }
  }
} // end anonymous namespace

/**
 * @brief transpose i-j dimenions of grid and brick
 * 
 * Transpose i-j dimensions of a grid and associated brick on
 * a CUDA device
 * 
 * @tparam TileJDim the j-dimension of the tiles to use when transposing the grid
 * @tparam TileIDim the i-dimension of the tiles to use when transposing the grid
 * @tparam BrickType type of the brick
 * @param[in] in_grid_ptr pointer to in-grid
 * @param[in] in_brick input brick
 * @param[out] out_grid_ptr pointer to output grid
 * @param[out] out_brick pointer to output brick
 * @param[in] grid_size extent of grid in each dimension
 */
template<unsigned TileJDim, unsigned TileIDim, typename BrickType>
void transpose_brick_ij_on_dev(const unsigned *in_grid_ptr, const BrickType &in_brick,
                               unsigned *out_grid_ptr, BrickType *out_brick,
                               const std::vector<size_t> grid_size)
{
  // launch transpose of grid
  constexpr size_t BLOCK_SIZE = 256;
  const size_t grid_x = (grid_size[0] + TileIDim - 1) / TileIDim;
  const size_t grid_y = (grid_size[1] + TileJDim - 1) / TileJDim;
  const size_t collapsed_dims = std::accumulate(grid_size.begin() + 2, grid_size.end(), 1);
  size_t tilesPerBlock = gcd(BLOCK_SIZE / TileIDim / TileJDim, collapsed_dims);
  const dim3 grid(grid_x, grid_y, collapsed_dims / tilesPerBlock),
            block(TileIDim, TileJDim, tilesPerBlock);
  transpose_ij<TileJDim, TileIDim><< <grid, block>> >(in_grid_ptr, out_grid_ptr, grid_size[1], grid_size[0]);
  transpose_brick_ij<< <in_brick.bInfo->nbricks, 64>> >(in_brick, out_brick);
}