//
// Created by Benjamin Sepanski on 12/2/21.
//

#ifndef BRICK_GENE_6D_STENCILS_H
#define BRICK_GENE_6D_STENCILS_H

#include <iomanip>
#include <iostream>

#include "Array.h"
#include "BrickedArray.h"
#include "array-mpi.h"
#include "brick-cuda.h"
#include "brickcompare.h"
#include "bricksetup.h"
#include "multiarray.h"

// useful constants
constexpr unsigned RANK = 6;
constexpr std::array<unsigned, RANK> BRICK_DIM = {2, 16, 2, 2, 1, 1};
constexpr std::array<unsigned, RANK> ARAKAWA_COEFF_BRICK_DIM = {
    1, BRICK_DIM[0], BRICK_DIM[2], BRICK_DIM[3], BRICK_DIM[4], BRICK_DIM[5]};
constexpr unsigned NUM_GHOST_ZONES = 1;
constexpr std::array<unsigned, RANK> GHOST_ZONE = {0, 0, 2 * NUM_GHOST_ZONES,
                                                   2 * NUM_GHOST_ZONES, 0, 0};
constexpr std::array<unsigned, RANK> PADDING = {0,0,0,0,0,0};
//    GHOST_ZONE[0] > 0 ? 2 : 0, GHOST_ZONE[1] > 0 ? 2 : 0, GHOST_ZONE[2] > 0 ? 2 : 0,
//    GHOST_ZONE[3] > 0 ? 2 : 0, GHOST_ZONE[4] > 0 ? 2 : 0, GHOST_ZONE[5] > 0 ? 2 : 0};
constexpr unsigned TILE_SIZE = 8;
constexpr unsigned ARAKAWA_STENCIL_SIZE = 13;
constexpr unsigned NUM_ELEMENTS_PER_FIELD_BRICK =
    BRICK_DIM[0] * BRICK_DIM[1] * BRICK_DIM[2] * BRICK_DIM[3] * BRICK_DIM[4] * BRICK_DIM[5];

// check constants
static_assert(GHOST_ZONE[0] % BRICK_DIM[0] == 0, "GHOST_ZONE must be a multiple of BRICK_DIM");
static_assert(GHOST_ZONE[1] % BRICK_DIM[1] == 0, "GHOST_ZONE must be a multiple of BRICK_DIM");
static_assert(GHOST_ZONE[2] % BRICK_DIM[2] == 0, "GHOST_ZONE must be a multiple of BRICK_DIM");
static_assert(GHOST_ZONE[3] % BRICK_DIM[3] == 0, "GHOST_ZONE must be a multiple of BRICK_DIM");
static_assert(GHOST_ZONE[4] % BRICK_DIM[4] == 0, "GHOST_ZONE must be a multiple of BRICK_DIM");
static_assert(GHOST_ZONE[5] % BRICK_DIM[5] == 0, "GHOST_ZONE must be a multiple of BRICK_DIM");

// useful types
typedef Dim<BRICK_DIM[5], BRICK_DIM[4], BRICK_DIM[3], BRICK_DIM[2], BRICK_DIM[1], BRICK_DIM[0]>
    FieldBrickDimsType;
typedef Dim<BRICK_DIM[5], BRICK_DIM[4], BRICK_DIM[3], BRICK_DIM[2], BRICK_DIM[0]>
    PreCoeffBrickDimsType;
typedef Dim<ARAKAWA_COEFF_BRICK_DIM[5], ARAKAWA_COEFF_BRICK_DIM[4], ARAKAWA_COEFF_BRICK_DIM[3],
            ARAKAWA_COEFF_BRICK_DIM[2], ARAKAWA_COEFF_BRICK_DIM[1], ARAKAWA_COEFF_BRICK_DIM[0]>
    ArakawaCoeffBrickDimsType;
typedef Dim<1> VectorFoldType;
typedef CommDims<false, false, false, false, false, true> CommIn_i;
typedef CommDims<false, false, true, true, false, false> CommIn_kl;
typedef CommDims<false, false, false, false, false, false> NoComm;
typedef Brick<FieldBrickDimsType, VectorFoldType, true, CommIn_i> FieldBrick_i;
typedef Brick<FieldBrickDimsType, VectorFoldType, true, CommIn_kl> FieldBrick_kl;
typedef Brick<PreCoeffBrickDimsType, Dim<1>, true, NoComm> PreCoeffBrick;
typedef Brick<ArakawaCoeffBrickDimsType, VectorFoldType, false, NoComm> ArakawaCoeffBrick;

typedef brick::Padding<PADDING[5], PADDING[4], PADDING[3], PADDING[2], PADDING[1], PADDING[0]>
    Padding_kl6D;
typedef brick::Array<bComplexElem, 6, Padding_kl6D> complexArray6D;
typedef brick::Array<bElem, 6> realArray6D;
typedef brick::BrickedArray<bComplexElem, FieldBrickDimsType, VectorFoldType> BrickedFieldArray;
typedef brick::BrickedArray<bElem, ArakawaCoeffBrickDimsType, VectorFoldType>
    BrickedArakawaCoeffArray;

// some CUDA constants
constexpr unsigned WARP_SIZE = 32;
#if __CUDA_ARCH__ <= 370 || __CUDA_ARCH__ == 750 || __CUDA_ARCH__ == 860
constexpr unsigned MAX_BLOCKS_PER_SM = 16;
#else
constexpr unsigned MAX_BLOCKS_PER_SM = 32;
#endif
#if __CUDA_ARCH__ <= 720 || __CUDA_ARCH__ == 800
constexpr unsigned MAX_WARPS_PER_SM = 64;
#elif __CUDA_ARCH__ == 750
constexpr unsigned MAX_WARPS_PER_SM = 32;
#elif __CUDA__ == 860
constexpr unsigned MAX_WARPS_PER_SM = 48;
#else
#error Unexpected compute capability #__CUDA_ARCH__
#endif

constexpr unsigned max_blocks_per_sm(unsigned max_block_size) {
  unsigned max_num_warps_per_block = max_block_size / WARP_SIZE;
  unsigned max_blocks_per_sm = MAX_WARPS_PER_SM / max_num_warps_per_block;
  if (max_blocks_per_sm > MAX_BLOCKS_PER_SM) {
    max_blocks_per_sm = MAX_BLOCKS_PER_SM;
  }
  return max_blocks_per_sm;
}

// used to copy i derivative coefficients into constant memory
void copy_i_deriv_coeff(const bElem i_deriv_coeff_host[5]);

// declare cuda kernels
/**
 * @brief Compute on the non-ghost bricks
 *
 * Assumes that grid-size is I x J x KLMN.
 * Assumes i-deriv coeff has been copied to constant memory
 * @see copy_i_deriv_coeff
 */
__global__ void
ijDerivBrickKernel(brick::Array<unsigned, RANK, brick::Padding<>, unsigned> grid,
                   brick::Array<unsigned, RANK, brick::Padding<>, unsigned> coeffGrid,
                   FieldBrick_i bIn, FieldBrick_i bOut, PreCoeffBrick bP1, PreCoeffBrick bP2,
                   bComplexElem *ikj);

/**
 * @brief Compute on the non-ghost bricks
 *
 * Assumes that grid-size is determined by grid-iteration order
 * @see copy_grid_iteration_order
 */
__global__ void
semiArakawaBrickKernel(brick::Array<unsigned, RANK, brick::Padding<>, unsigned> grid,
                       brick::Array<unsigned, RANK, brick::Padding<>, unsigned> coeffGrid,
                       FieldBrick_kl bIn, FieldBrick_kl bOut, ArakawaCoeffBrick coeff);

#endif // BRICK_GENE_6D_STENCILS_H
