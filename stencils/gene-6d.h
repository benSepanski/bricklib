#ifndef BRICK_GENE_5D_H
#define BRICK_GENE_5D_H

#define GTENSOR_DEFAULT_DEVICE_ALLOCATOR(T) gt::device_allocator<T>

#include <algorithm>
#include <cuComplex.h>
#include <gtensor/gtensor.h>
#include <iostream>
#include <random>
#include "bricksetup.h"
#include "multiarray.h"
#include "brickcompare.h"
#include "gpuvfold.h"
#include "stencils_cu.h"

#undef N
#undef TILE
#undef GZ
#undef PADDING
#undef STRIDE
#undef STRIDEG
#undef NB
#undef GB
#undef STRIDEB
#undef BDIM
#undef VFOLD
#undef _TILEFOR

// useful constants
constexpr bElem pi = 3.14159265358979323846;

// blocking dimensions
constexpr unsigned DIM = 6;
constexpr unsigned TILE = 8;
// set brick sizes
constexpr unsigned BDIM_i = 2;
constexpr unsigned BDIM_j = 16;
constexpr unsigned BDIM_k = 2;
constexpr unsigned BDIM_l = 2;
constexpr unsigned BDIM_m = 1;
constexpr unsigned BDIM_n = 1;
constexpr std::array<unsigned, DIM> BDIM_arr = {BDIM_i, BDIM_j, BDIM_k, BDIM_l, BDIM_m, BDIM_n};
constexpr unsigned NUM_ELEMENTS_PER_BRICK = BDIM_i * BDIM_j * BDIM_k * BDIM_l * BDIM_m * BDIM_n;
// num elements in each direction
constexpr unsigned EXTENT_i = 72;
constexpr unsigned EXTENT_j = 32;
constexpr unsigned EXTENT_k = 24;
constexpr unsigned EXTENT_l = 24;
constexpr unsigned EXTENT_m = 32;
constexpr unsigned EXTENT_n = 2;
constexpr unsigned NUM_ELEMENTS = EXTENT_i * EXTENT_j * EXTENT_k * EXTENT_l * EXTENT_m * EXTENT_n;
#define EXTENT EXTENT_i,EXTENT_j,EXTENT_k,EXTENT_l,EXTENT_m,EXTENT_n
// padding (for arrays only)
constexpr unsigned PADDING_i = BDIM_i > 1 ? BDIM_i : 0;
constexpr unsigned PADDING_j = 0;
constexpr unsigned PADDING_k = BDIM_k > 1 ? BDIM_k : 0;
constexpr unsigned PADDING_l = BDIM_l > 1 ? BDIM_l : 0;
constexpr unsigned PADDING_m = 0;
constexpr unsigned PADDING_n = 0;
#define PADDING PADDING_i,PADDING_j,PADDING_k,PADDING_l,PADDING_m,PADDING_n
constexpr std::array<unsigned, DIM> PADDING_arr = {PADDING};
// ghost zone (for arrays and bricks)
constexpr unsigned GHOST_ZONE_i = PADDING_i;
constexpr unsigned GHOST_ZONE_j = PADDING_j;
constexpr unsigned GHOST_ZONE_k = PADDING_k;
constexpr unsigned GHOST_ZONE_l = PADDING_l;
constexpr unsigned GHOST_ZONE_m = PADDING_m;
constexpr unsigned GHOST_ZONE_n = PADDING_n;
#define GHOST_ZONE GHOST_ZONE_i,GHOST_ZONE_j,GHOST_ZONE_k,GHOST_ZONE_l,GHOST_ZONE_m,GHOST_ZONE_n
constexpr std::array<unsigned, DIM> GHOST_ZONE_arr = {GHOST_ZONE};

// num elements in each direction + ghosts
constexpr unsigned GZ_EXTENT_i = (EXTENT_i + 2 * GHOST_ZONE_i);
constexpr unsigned GZ_EXTENT_j = (EXTENT_j + 2 * GHOST_ZONE_j);
constexpr unsigned GZ_EXTENT_k = (EXTENT_k + 2 * GHOST_ZONE_k);
constexpr unsigned GZ_EXTENT_l = (EXTENT_l + 2 * GHOST_ZONE_l);
constexpr unsigned GZ_EXTENT_m = (EXTENT_m + 2 * GHOST_ZONE_m);
constexpr unsigned GZ_EXTENT_n = (EXTENT_n + 2 * GHOST_ZONE_n);
#define GZ_EXTENT GZ_EXTENT_i,GZ_EXTENT_j,GZ_EXTENT_k,GZ_EXTENT_l,GZ_EXTENT_m,GZ_EXTENT_n
// num elements in each direction + padding + ghosts
constexpr unsigned PADDED_EXTENT_i = GZ_EXTENT_i + 2 * PADDING_i;
constexpr unsigned PADDED_EXTENT_j = GZ_EXTENT_j + 2 * PADDING_j;
constexpr unsigned PADDED_EXTENT_k = GZ_EXTENT_k + 2 * PADDING_k;
constexpr unsigned PADDED_EXTENT_l = GZ_EXTENT_l + 2 * PADDING_l;
constexpr unsigned PADDED_EXTENT_m = GZ_EXTENT_m + 2 * PADDING_m;
constexpr unsigned PADDED_EXTENT_n = GZ_EXTENT_n + 2 * PADDING_n;
constexpr unsigned NUM_PADDED_ELEMENTS = (PADDED_EXTENT_i * PADDED_EXTENT_j * PADDED_EXTENT_k * PADDED_EXTENT_l * PADDED_EXTENT_m * PADDED_EXTENT_n);
#define PADDED_EXTENT PADDED_EXTENT_i,PADDED_EXTENT_j,PADDED_EXTENT_k,PADDED_EXTENT_l,PADDED_EXTENT_m,PADDED_EXTENT_n
// number of non-ghost bricks in each direction
constexpr unsigned BRICK_EXTENT_i = EXTENT_i / BDIM_i;
constexpr unsigned BRICK_EXTENT_j = EXTENT_j / BDIM_j;
constexpr unsigned BRICK_EXTENT_k = EXTENT_k / BDIM_k;
constexpr unsigned BRICK_EXTENT_l = EXTENT_l / BDIM_l;
constexpr unsigned BRICK_EXTENT_m = EXTENT_m / BDIM_m;
constexpr unsigned BRICK_EXTENT_n = EXTENT_n / BDIM_n;
constexpr unsigned NUM_BRICKS = BRICK_EXTENT_i * BRICK_EXTENT_j * BRICK_EXTENT_k * BRICK_EXTENT_l * BRICK_EXTENT_m * BRICK_EXTENT_n;
#define BRICK_EXTENT BRICK_EXTENT_i,BRICK_EXTENT_j,BRICK_EXTENT_k,BRICK_EXTENT_l,BRICK_EXTENT_m,BRICK_EXTENT_n
// number of bricks + ghost bricks in each direction
constexpr unsigned GZ_BRICK_EXTENT_i = GZ_EXTENT_i / BDIM_i;
constexpr unsigned GZ_BRICK_EXTENT_j = GZ_EXTENT_j / BDIM_j;
constexpr unsigned GZ_BRICK_EXTENT_k = GZ_EXTENT_k / BDIM_k;
constexpr unsigned GZ_BRICK_EXTENT_l = GZ_EXTENT_l / BDIM_l;
constexpr unsigned GZ_BRICK_EXTENT_m = GZ_EXTENT_m / BDIM_m;
constexpr unsigned GZ_BRICK_EXTENT_n = GZ_EXTENT_n / BDIM_n;
constexpr unsigned NUM_GZ_BRICKS = GZ_BRICK_EXTENT_i * GZ_BRICK_EXTENT_j * GZ_BRICK_EXTENT_k * GZ_BRICK_EXTENT_l * GZ_BRICK_EXTENT_m * GZ_BRICK_EXTENT_n;
#define GZ_BRICK_EXTENT GZ_BRICK_EXTENT_i,GZ_BRICK_EXTENT_j,GZ_BRICK_EXTENT_k,GZ_BRICK_EXTENT_l,GZ_BRICK_EXTENT_m,GZ_BRICK_EXTENT_n

// set our brick types
typedef CommDims<false, false, false, false, false, true> CommIn_i;
typedef CommDims<false, false, true, true, false, false> CommIn_kl;
typedef CommDims<false, false, false, false, false, false> NoComm;
typedef Brick<Dim<BDIM_n, BDIM_m, BDIM_l, BDIM_k, BDIM_j, BDIM_i>, Dim<1>, true, CommIn_i> FieldBrick_i ;
typedef Brick<Dim<BDIM_n, BDIM_m, BDIM_l, BDIM_k, BDIM_j, BDIM_i>, Dim<1>, true, CommIn_kl> FieldBrick_kl ;
typedef Brick<Dim<BDIM_n, BDIM_m, BDIM_l, BDIM_k, BDIM_i>, Dim<1>, true, NoComm> PreCoeffBrick;
typedef Brick<Dim<BDIM_n, BDIM_m, BDIM_l, BDIM_k, BDIM_i>, Dim<1>, false, NoComm> RealCoeffBrick;

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
__global__ void
ij_deriv_brick_kernel(unsigned (*fieldGrid)[GZ_BRICK_EXTENT_m][GZ_BRICK_EXTENT_l][GZ_BRICK_EXTENT_k][GZ_BRICK_EXTENT_j][GZ_BRICK_EXTENT_i],
                      unsigned (*coeffGrid)[GZ_BRICK_EXTENT_m][GZ_BRICK_EXTENT_l][GZ_BRICK_EXTENT_k][GZ_BRICK_EXTENT_i],
                      FieldBrick_i bIn,
                      FieldBrick_i bOut,
                      PreCoeffBrick bP1,
                      PreCoeffBrick bP2,
                      bComplexElem *ikj);

constexpr unsigned IJ_DERIV_BRICK_KERNEL_VEC_BLOCK_SIZE = NUM_ELEMENTS_PER_BRICK;
static_assert(IJ_DERIV_BRICK_KERNEL_VEC_BLOCK_SIZE % WARP_SIZE == 0);
static_assert(NUM_ELEMENTS_PER_BRICK % IJ_DERIV_BRICK_KERNEL_VEC_BLOCK_SIZE == 0);
__global__ void
ij_deriv_brick_kernel_vec(unsigned (*fieldGrid)[GZ_BRICK_EXTENT_m][GZ_BRICK_EXTENT_l][GZ_BRICK_EXTENT_k][GZ_BRICK_EXTENT_j][GZ_BRICK_EXTENT_i],
                          unsigned (*coeffGrid)[GZ_BRICK_EXTENT_m][GZ_BRICK_EXTENT_l][GZ_BRICK_EXTENT_k][GZ_BRICK_EXTENT_i],
                          FieldBrick_i bIn,
                          FieldBrick_i bOut,
                          PreCoeffBrick bP1,
                          PreCoeffBrick bP2,
                          bComplexElem *ikj) ;

__global__ void
semi_arakawa_brick_kernel(unsigned *fieldGrid,
                          unsigned *coeffGrid,
                          FieldBrick_kl bIn,
                          FieldBrick_kl bOut,
                          RealCoeffBrick *coeff);

constexpr unsigned SEMI_ARAKAWA_BRICK_KERNEL_VEC_BLOCK_SIZE = std::min(128U, NUM_ELEMENTS_PER_BRICK);
static_assert(SEMI_ARAKAWA_BRICK_KERNEL_VEC_BLOCK_SIZE % WARP_SIZE == 0);
static_assert(NUM_ELEMENTS_PER_BRICK % SEMI_ARAKAWA_BRICK_KERNEL_VEC_BLOCK_SIZE == 0);
__global__ void
semi_arakawa_brick_kernel_vec(unsigned *fieldGrid,
                              unsigned *coeffGrid,
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