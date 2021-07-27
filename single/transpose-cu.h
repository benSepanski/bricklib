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
#include "brick.h"

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
 * 1-dimensional kernel
 * 
 * @tparam elemType the type of elements in the matrix
 * @tparam TileJDim the j-extent of the shared-memory tiles used 
 * @tparam TileIDim the i-extent of the shared-memory tiles used 
 * @param[in] in_mat pointer to array of shape [gridDim.z*blockDim.z][J][I]
 * @param[out] out_mat pointer to array of shape [gridDim.z*blockDim.z][I][J]
 * @param[in] K number of matrices
 * @param[in] J an extent of the matrices
 * @param[in] I an extent of the matrices
 */
// Based on https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/
template<unsigned TileJDim, unsigned TileIDim, typename elemType>
__global__
void transpose_ij(const elemType * __restrict__ in_mat, elemType * __restrict__ out_mat, const size_t K, const size_t J, const size_t I)
{
  // avoid bank conflicts
  static constexpr unsigned BANK_SIZE = 32;
  static constexpr unsigned BANK_WIDTH_IN_BYTES = 32 / 4;
  /// which bank is (tile + TileJDim)?
  static constexpr unsigned BANK_OF_TileJDim = (TileJDim * sizeof(elemType) / BANK_WIDTH_IN_BYTES) % BANK_SIZE;
  // shift so that bank-shift of one row is one elemnt 
  static constexpr unsigned BANK_SHIFT = (sizeof(elemType) / BANK_WIDTH_IN_BYTES - BANK_OF_TileJDim + BANK_SIZE) % BANK_SIZE; 
  __shared__ elemType tile[TileJDim][TileIDim + BANK_SHIFT * BANK_WIDTH_IN_BYTES / sizeof(elemType)];
  // set up grid-stride loop over tiles
  const size_t num_tiles_in_i = (I + TileIDim - 1) / TileIDim;
  const size_t num_tiles_in_j = (J + TileJDim - 1) / TileJDim;
  const size_t num_tiles = num_tiles_in_i * num_tiles_in_j * K;
  for(size_t tile_idx = blockIdx.x ; tile_idx < num_tiles; tile_idx += gridDim.x)
  {
    size_t tile_i_idx = tile_idx % num_tiles_in_i;
    size_t tile_j_idx = (tile_idx / num_tiles_in_i) % num_tiles_in_j;
    size_t k = tile_idx / num_tiles_in_i / num_tiles_in_j;
    assert(k < K);

    size_t tile_i_offset = min(tile_i_idx * TileIDim, I);
    size_t tile_j_offset = min(tile_j_idx * TileJDim, J);

    // get upper bounds on tile
    const size_t upper_bound_j = min((size_t) TileJDim, J - tile_j_offset),
                upper_bound_i = min((size_t) TileIDim, I - tile_i_offset);
    const size_t tile_size = upper_bound_j * upper_bound_i;
    
    // load tile into shared memory
    for(unsigned flat_idx = threadIdx.x ; flat_idx < tile_size; flat_idx += blockDim.x)
    {
      unsigned i = flat_idx % upper_bound_i;
      unsigned j = flat_idx / upper_bound_i;
      size_t index = flatten_index(k, tile_j_offset + j, tile_i_offset + i, J, I);
      assert(j < upper_bound_j);
      assert(index < K * J * I);
      tile[j][i] = in_mat[index];
    }
    __syncthreads();
    // load tile into shared memory
    for(unsigned flat_idx = threadIdx.x ; flat_idx < tile_size; flat_idx += blockDim.x)
    {
      unsigned j = flat_idx % upper_bound_j;
      unsigned i = flat_idx / upper_bound_j;
      size_t index = flatten_index(k, tile_i_offset + i, tile_j_offset + j, I, J);
      assert(i < upper_bound_i);
      assert(index < K * J * I);
      out_mat[index] = tile[j][i];
    }
    __syncthreads();
  }
}

/**
 * @brief transpose in-brick into out-brick
 * 
 * Only modifies the data in out_brick's associated BrickStorage. Does
 * not transpose meta-data.
 * 
 * Should be invoked as a 1-dimensional kernel
 * 
 * @tparam BrickType type of in brick
 * @tparam BrickTypeTransposed type of out brick
 * @param[in] in_brick to the brick
 * @param[out] out_brick to the brick to transpose into
 */
template<typename BrickType, typename BrickTypeTransposed>
__global__
void transpose_brick_ij(const BrickType in_brick, BrickTypeTransposed out_brick)
{
  constexpr unsigned BDIM_j = BrickType::template getBrickDim<1>(),
                     BDIM_i = BrickType::template getBrickDim<0>(),
                     VFOLD_j = BrickType::template getVectorFold<1>(),
                     VFOLD_i = BrickType::template getVectorFold<0>();
  static_assert(BDIM_i == BrickTypeTransposed::template getBrickDim<1>());
  static_assert(BDIM_j == BrickTypeTransposed::template getBrickDim<0>());
  // grid-strided loop over bricks
  for(unsigned brick_idx = blockIdx.x ; brick_idx < in_brick.bInfo->nbricks; brick_idx += gridDim.x)
  {
    // transpose data
    for(unsigned flat_idx = threadIdx.x; flat_idx < BrickType::BRICKLEN; flat_idx += blockDim.x)
    {
      unsigned in_vec = flat_idx % BrickType::VECLEN;
      unsigned of_vec = flat_idx / BrickType::VECLEN;
      unsigned i_in_vec = in_vec % VFOLD_i;
      unsigned j_in_vec = (in_vec / VFOLD_i) % VFOLD_j;
      unsigned i_of_vec = of_vec % (BDIM_i / VFOLD_i);
      unsigned j_of_vec = (of_vec / (BDIM_i / VFOLD_i)) % (BDIM_j / VFOLD_j);
      unsigned new_in_vec_idx = in_vec + (VFOLD_j - 1) * i_in_vec
                                       + (1 - VFOLD_i) * j_in_vec;
      unsigned new_of_vec_idx = of_vec + (BDIM_j / VFOLD_j - 1) * i_of_vec
                                       + (1 - BDIM_i / VFOLD_i) * j_of_vec;
      assert(new_in_vec_idx < BrickType::VECLEN);
      assert(new_of_vec_idx < BrickType::BRICKLEN / BrickType::VECLEN);
      unsigned new_in_brick_idx = new_of_vec_idx * BrickType::VECLEN + new_in_vec_idx;

      unsigned index = flat_idx + in_brick.step * brick_idx;
      unsigned index_tr = new_in_brick_idx + out_brick.step * brick_idx;
      assert(index < in_brick.bInfo->nbricks * in_brick.step);
      assert(index_tr < out_brick.bInfo->nbricks * out_brick.step);
      typename BrickType::elemType val = in_brick.dat[index];
      out_brick.dat[index_tr] = val;
    }
  } 
}

/**
 * @brief transpose in-brick info into out-brick info
 * 
 * Should be invoked as a 1-dimensional kernel
 * 
 * @tparam BrickInfoType must include neighbors in dimensions 0 and 1.
 * @param[in] in_brick_info the input brick info
 * @param[out] out_brick_info pointer to the brick-info to transpose into
 */
template<typename BrickInfoType>
__global__
void transpose_brick_info_ij(const BrickInfoType *in_brick_info, BrickInfoType *out_brick_info)
{
  static_assert(BrickInfoType::adjListIncludesDim(0) && BrickInfoType::adjListIncludesDim(1),
                "i-j metadata transposition is a nop unless BrickInfoType communicates in dimensions 0 and 1");

  // grid-strided loop over bricks
  for(unsigned brick_idx = blockIdx.x ; brick_idx < in_brick_info->nbricks; brick_idx += gridDim.x)
  {
    constexpr unsigned nneighbors = static_power<3, BrickInfoType::numDimsWithRecordedNeighbors>::value;
    // transpose data
    for(size_t idx = threadIdx.x; idx < nneighbors; idx += blockDim.x)
    {
      unsigned neighbor = in_brick_info->adj[idx];
      unsigned first_ternary_digit = idx % 3;
      unsigned second_ternary_digit = (idx / 3) % 3;
      unsigned idx_with_flipped_ternary_digits = idx + 2 * first_ternary_digit - 2 * second_ternary_digit;
      out_brick_info->adj[idx_with_flipped_ternary_digits] = neighbor;
    }
  } 
}

/**
 * @brief invoke cuda-kernel to transpose i-j dimenions.
 * 
 * Case where adjacency list includes neighbors in i and j dimensions
 * 
 * @param in_brick_info[in] pointer to input brick-info (on device)
 * @param out_brick_info[out] pointer to output brick-info (on device)
 * @see transpose_brick_info_ij
 */
template<typename BrickInfoType>
typename std::enable_if<BrickInfoType::adjListIncludesDim(0) && BrickInfoType::adjListIncludesDim(1)>::type
transpose_brick_info_ij_on_device(unsigned gridSize, unsigned blockSize, const BrickInfoType *in_brick_info, BrickInfoType *out_brick_info)
{
  transpose_brick_info_ij<< <gridSize, blockSize>> >(in_brick_info, out_brick_info);
}

/**
 * @brief nop case of transpose in i-j dimensions.
 * 
 * @param in_brick_info[in] pointer to input brick-info (on device)
 * @param out_brick_info[out] pointer to output brick-info (on device)
 * @see transpose_brick_info_ij
 */
template<typename BrickInfoType>
typename std::enable_if<!(BrickInfoType::adjListIncludesDim(0) && BrickInfoType::adjListIncludesDim(1))>::type
transpose_brick_info_ij_on_device(unsigned gridSize, unsigned blockSize, const BrickInfoType *in_brick_info, BrickInfoType *out_brick_info) { }