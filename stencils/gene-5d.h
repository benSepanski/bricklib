#ifndef BRICK_GENE_5D_H
#define BRICK_GENE_5D_H

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

// tiled for loop, ignoring padded regions
#define _TILEFOR5D _Pragma("omp parallel for collapse(4)") \
for (long tm = PADDING_m; tm < PADDING_m + EXTENT_m; tm += TILE) \
for (long tl = PADDING_l; tl < PADDING_l + EXTENT_l; tl += TILE) \
for (long tk = PADDING_k; tk < PADDING_k + EXTENT_k; tk += TILE) \
for (long tj = PADDING_j; tj < PADDING_j + EXTENT_j; tj += TILE) \
for (long ti = PADDING_i; ti < PADDING_i + EXTENT_i; ti += TILE) \
for (long m = tm; m < tm + TILE; ++m) \
for (long l = tl; l < tl + TILE; ++l) \
for (long k = tk; k < tk + TILE; ++k) \
for (long j = tj; j < tj + TILE; ++j) \
_Pragma("omp simd") \
for (long i = ti; i < ti + TILE; ++i)

// useful constants
constexpr bElem pi = 3.14159265358979323846;

#define PADDING_i 8
#define PADDING_j 0
#define PADDING_k 8
#define PADDING_l 8
#define PADDING_m 0
#define PADDING PADDING_i,PADDING_j,PADDING_k,PADDING_l,PADDING_m
#define GHOST_ZONE_i 0
#define GHOST_ZONE_j 0
#define GHOST_ZONE_k 0
#define GHOST_ZONE_l 0
#define GHOST_ZONE_m 0
#define GHOST_ZONE GHOST_ZONE_i,GHOST_ZONE_j,GHOST_ZONE_k,GHOST_ZONE_l,GHOST_ZONE_m

// blocking dimensions
constexpr unsigned DIM = 5;
constexpr unsigned TILE = 8;
#define BDIM_i 4
#define BDIM_j 4
#define BDIM_k 4
#define BDIM_l 4
#define BDIM_m 4
#define BDIM BDIM_i,BDIM_j,BDIM_k,BDIM_l,BDIM_m
#define VFOLD 2,4,4
// num elements in each direction
#define EXTENT_i 72
#define EXTENT_j 32
#define EXTENT_k 24
#define EXTENT_l 24
#define EXTENT_m 32
#define EXTENT EXTENT_i,EXTENT_j,EXTENT_k,EXTENT_l,EXTENT_m
#define NUM_ELEMENTS EXTENT_i * EXTENT_j * EXTENT_k * EXTENT_l * EXTENT_m
// num elements in each direction + padding + ghosts
#define PADDED_EXTENT_i EXTENT_i + 2 * (PADDING_i + GHOST_ZONE_i)
#define PADDED_EXTENT_j EXTENT_j + 2 * (PADDING_j + GHOST_ZONE_j)
#define PADDED_EXTENT_k EXTENT_k + 2 * (PADDING_k + GHOST_ZONE_k)
#define PADDED_EXTENT_l EXTENT_l + 2 * (PADDING_l + GHOST_ZONE_l)
#define PADDED_EXTENT_m EXTENT_m + 2 * (PADDING_m + GHOST_ZONE_m)
#define PADDED_EXTENT PADDED_EXTENT_i,PADDED_EXTENT_j,PADDED_EXTENT_k,PADDED_EXTENT_l,PADDED_EXTENT_m
#define NUM_PADDED_ELEMENTS PADDED_EXTENT_i * PADDED_EXTENT_j * PADDED_EXTENT_k * PADDED_EXTENT_l * PADDED_EXTENT_m
// number of blocks (including padded blocks)
#define BRICK_GRID_EXTENT_i (EXTENT_i + 2 * GHOST_ZONE_i) / BDIM_i
#define BRICK_GRID_EXTENT_j (EXTENT_j + 2 * GHOST_ZONE_j) / BDIM_j
#define BRICK_GRID_EXTENT_k (EXTENT_k + 2 * GHOST_ZONE_k) / BDIM_k
#define BRICK_GRID_EXTENT_l (EXTENT_l + 2 * GHOST_ZONE_l) / BDIM_l
#define BRICK_GRID_EXTENT_m (EXTENT_m + 2 * GHOST_ZONE_m) / BDIM_m
#define BRICK_GRID_EXTENT BRICK_GRID_EXTENT_i,BRICK_GRID_EXTENT_j,BRICK_GRID_EXTENT_k,BRICK_GRID_EXTENT_l,BRICK_GRID_EXTENT_m
#define NUM_BRICKS BRICK_GRID_EXTENT_i * BRICK_GRID_EXTENT_j * BRICK_GRID_EXTENT_k * BRICK_GRID_EXTENT_l * BRICK_GRID_EXTENT_m

void cudaDouble(gt::complex<bElem> *inPtr, gt::complex<bElem> *outPtr);

#endif // BRICK_GENE_5D_H