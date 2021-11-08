#ifndef BRICK_GENE_5D_H
#define BRICK_GENE_5D_H

#define GTENSOR_DEFAULT_DEVICE_ALLOCATOR(T) gt::device_allocator<T>

#include <algorithm>
#include <cuComplex.h>
#include <gtensor/gtensor.h>
#include <iostream>
#include <random>
#include "bricked-array.h"
#include "../../stencils/stencils.h"
#include "../../stencils/stencils_cu.h"
#undef TILE
#undef BDIM
#undef PADDING
#undef N

// blocking dimensions
constexpr unsigned RANK = 6;
constexpr unsigned TILE = 2;
// set brick sizes
constexpr unsigned BDIM[RANK] = {2, 32, 2, 2, 1, 1};
constexpr unsigned NUM_ELEMENTS_PER_BRICK = BDIM[5] * BDIM[4] * BDIM[3]
                                          * BDIM[2] * BDIM[1] * BDIM[0];
// num elements in each direction
//constexpr unsigned EXTENT[RANK] = {72, 32, 24, 24, 32, 2};
constexpr unsigned EXTENT[RANK] = {2, 32, 2, 2, 1, 1};
constexpr unsigned NUM_ELEMENTS = EXTENT[5] * EXTENT[4] * EXTENT[3]
                                * EXTENT[2] * EXTENT[1] * EXTENT[0];
// padding (for arrays only)
constexpr unsigned PADDING[RANK] = {BDIM[0] > 1 ? BDIM[0] : 0,
                                    0,
                                    BDIM[2] > 1 ? BDIM[2] : 0,
                                    BDIM[3] > 1 ? BDIM[3] : 0,
                                    0,
                                    0};
// ghost zone (for arrays and bricks)
constexpr unsigned GHOST_ZONE[RANK] = {PADDING[0], PADDING[1], PADDING[2],
                                       PADDING[3], PADDING[4], PADDING[5]};
// ghost-zone in units of bricks
constexpr unsigned GHOST_ZONE_BRICK[RANK] = {
    GHOST_ZONE[0] / BDIM[0], GHOST_ZONE[1] / BDIM[1],
    GHOST_ZONE[2] / BDIM[2], GHOST_ZONE[3] / BDIM[3],
    GHOST_ZONE[4] / BDIM[4], GHOST_ZONE[5] / BDIM[5]
};
// extent with ghost-zones
constexpr std::array<unsigned, RANK> EXTENT_WITH_GHOST_ZONE = {
    EXTENT[0] + 2 * GHOST_ZONE[0],
    EXTENT[1] + 2 * GHOST_ZONE[1],
    EXTENT[2] + 2 * GHOST_ZONE[2],
    EXTENT[3] + 2 * GHOST_ZONE[3],
    EXTENT[4] + 2 * GHOST_ZONE[4],
    EXTENT[5] + 2 * GHOST_ZONE[5]
};
// extent with ghost-zones and padding
constexpr std::array<unsigned, RANK> PADDED_EXTENT = {
    EXTENT_WITH_GHOST_ZONE[0] + 2 * PADDING[0],
    EXTENT_WITH_GHOST_ZONE[1] + 2 * PADDING[1],
    EXTENT_WITH_GHOST_ZONE[2] + 2 * PADDING[2],
    EXTENT_WITH_GHOST_ZONE[3] + 2 * PADDING[3],
    EXTENT_WITH_GHOST_ZONE[4] + 2 * PADDING[4],
    EXTENT_WITH_GHOST_ZONE[5] + 2 * PADDING[5]
};
constexpr std::array<unsigned, RANK> BRICK_GRID_EXTENT = {
    EXTENT[0] / BDIM[0],
    EXTENT[1] / BDIM[1],
    EXTENT[2] / BDIM[2],
    EXTENT[3] / BDIM[3],
    EXTENT[4] / BDIM[4],
    EXTENT[5] / BDIM[5]
};
constexpr unsigned NUM_BRICKS = BRICK_GRID_EXTENT[0] * BRICK_GRID_EXTENT[1]
                              * BRICK_GRID_EXTENT[2] * BRICK_GRID_EXTENT[3]
                              * BRICK_GRID_EXTENT[4] * BRICK_GRID_EXTENT[5];
constexpr std::array<unsigned, RANK> BRICK_GRID_EXTENT_WITH_GZ = {
    EXTENT_WITH_GHOST_ZONE[0] / BDIM[0],
    EXTENT_WITH_GHOST_ZONE[1] / BDIM[1],
    EXTENT_WITH_GHOST_ZONE[2] / BDIM[2],
    EXTENT_WITH_GHOST_ZONE[3] / BDIM[3],
    EXTENT_WITH_GHOST_ZONE[4] / BDIM[4],
    EXTENT_WITH_GHOST_ZONE[5] / BDIM[5]
};

// set our brick types
typedef CommDims<false, false, false, false, false, true> CommIn_i;
typedef CommDims<false, false, true, true, false, false> CommIn_kl;
typedef CommDims<false, false, false, false, false, false> NoComm;
typedef Dim<BDIM[5], BDIM[4], BDIM[3], BDIM[2], BDIM[1], BDIM[0]> BrickDims6D;
typedef Dim<BDIM[5], BDIM[4], BDIM[3], BDIM[2], BDIM[0]> BrickDims5D;
typedef brick::Padding<PADDING[5], PADDING[4], PADDING[3], PADDING[2], PADDING[1], PADDING[0]> Padding6D;
typedef brick::Padding<PADDING[5] + GHOST_ZONE[5],
                       PADDING[4] + GHOST_ZONE[4],
                       PADDING[3] + GHOST_ZONE[3],
                       PADDING[2] + GHOST_ZONE[2],
                       PADDING[1] + GHOST_ZONE[1],
                       PADDING[0] + GHOST_ZONE[0]> PaddingAndGhostZone6D;
typedef brick::Padding<PADDING[5], PADDING[4], PADDING[3], PADDING[2], PADDING[0]> Padding5D;
typedef Brick<BrickDims6D , Dim<1>, true, CommIn_i> FieldBrick_i ;
typedef Brick<BrickDims6D, Dim<1>, true, CommIn_kl> FieldBrick_kl ;
typedef Brick<BrickDims5D , Dim<1>, true, NoComm> PreCoeffBrick;
typedef Brick<BrickDims5D , Dim<1>, false, NoComm> RealCoeffBrick;

// some convenient array types
typedef brick::Array<bComplexElem, 6, Padding6D> complexArray6D;
typedef brick::Array<bElem, 6, brick::Padding<PADDING[5], PADDING[4], PADDING[3], PADDING[2], PADDING[0], 0> > realArray6D;
typedef brick::Array<bElem, 5, Padding5D> realArray5D;

// gtensor types
template<typename Space>
using gtensor6D = gt::gtensor<gt::complex<bElem>, 6, Space>;

// useful constants for stencil computations
constexpr unsigned ARAKAWA_STENCIL_SIZE = 13;
constexpr unsigned I_NBR_OFFSET = 1;
constexpr unsigned J_NBR_OFFSET = 3 * I_NBR_OFFSET;
constexpr unsigned K_NBR_OFFSET = 3 * J_NBR_OFFSET;
constexpr unsigned L_NBR_OFFSET = 3 * K_NBR_OFFSET;
constexpr unsigned M_NBR_OFFSET = 3 * L_NBR_OFFSET;
constexpr unsigned N_NBR_OFFSET = 3 * M_NBR_OFFSET;
constexpr unsigned CENTER_OFFSET_6D = I_NBR_OFFSET + J_NBR_OFFSET + K_NBR_OFFSET + L_NBR_OFFSET + M_NBR_OFFSET + N_NBR_OFFSET;
constexpr unsigned WARP_SIZE = 32;
#if __CUDA_ARCH__<=370 || __CUDA_ARCH__==750 || __CUDA_ARCH__==860
constexpr unsigned MAX_BLOCKS_PER_SM = 16;
#else
constexpr unsigned MAX_BLOCKS_PER_SM = 32;
#endif
#if __CUDA_ARCH__<=720 || __CUDA_ARCH__==800
constexpr unsigned MAX_WARPS_PER_SM = 64;
#elif __CUDA_ARCH__==750
constexpr unsigned MAX_WARPS_PER_SM = 32;
#elif __CUDA__==860
constexpr unsigned MAX_WARPS_PER_SM = 48;
#else
#error Unexpected compute capability #__CUDA_ARCH__
#endif

constexpr unsigned max_blocks_per_sm(unsigned max_block_size) {
    unsigned max_num_warps_per_block = max_block_size / WARP_SIZE;
    unsigned max_blocks_per_sm = MAX_WARPS_PER_SM / max_num_warps_per_block; 
    if(max_blocks_per_sm > MAX_BLOCKS_PER_SM)
    {
        max_blocks_per_sm = MAX_BLOCKS_PER_SM;
    }
    return max_blocks_per_sm;
}

// used to copy i derivative coefficients into constant memory
void copy_i_deriv_coeff(const bElem i_deriv_coeff_host[5]);

// used to copy the iteration order of the grid (for the semi-arkawa kernel (vec))
void copy_grid_iteration_order(const char * grid_iteration_order_host);

// declare cuda kernels
__global__
void semi_arakawa_brick_kernel(brick::Array<unsigned, RANK, brick::Padding<>, unsigned> fieldGrid,
                               brick::Array<unsigned, RANK-1, brick::Padding<>, unsigned> coeffGrid,
                               FieldBrick_kl bIn,
                               FieldBrick_kl bOut,
                               RealCoeffBrick *coeff);

// declare functions which time gtensor vs bricks
void semi_arakawa(bool run_bricks, bool run_gtensor);
void ij_deriv(bool run_bricks, bool run_gtensor);

// constants in gene-6d.cpp
extern unsigned NUM_WARMUP_ITERS;
extern unsigned NUM_ITERS;

#endif // BRICK_GENE_5D_H