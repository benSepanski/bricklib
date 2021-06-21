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

#define PADDING_i 4
#define PADDING_j 0
#define PADDING_k 4
#define PADDING_l 4
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
// set brick sizes
constexpr unsigned BDIM_i = 4;
constexpr unsigned BDIM_j = 4;
constexpr unsigned BDIM_k = 4;
constexpr unsigned BDIM_l = 4;
constexpr unsigned BDIM_m = 1;
// num elements in each direction
constexpr unsigned EXTENT_i = 72;
constexpr unsigned EXTENT_j = 32;
constexpr unsigned EXTENT_k = 24;
constexpr unsigned EXTENT_l = 24;
constexpr unsigned EXTENT_m = 32;
#define EXTENT EXTENT_i,EXTENT_j,EXTENT_k,EXTENT_l,EXTENT_m
constexpr unsigned NUM_ELEMENTS = EXTENT_i * EXTENT_j * EXTENT_k * EXTENT_l * EXTENT_m;
// num elements in each direction + ghosts
constexpr unsigned GZ_EXTENT_i = (EXTENT_i + 2 * GHOST_ZONE_i);
constexpr unsigned GZ_EXTENT_j = (EXTENT_j + 2 * GHOST_ZONE_j);
constexpr unsigned GZ_EXTENT_k = (EXTENT_k + 2 * GHOST_ZONE_k);
constexpr unsigned GZ_EXTENT_l = (EXTENT_l + 2 * GHOST_ZONE_l);
constexpr unsigned GZ_EXTENT_m = (EXTENT_m + 2 * GHOST_ZONE_m);
#define GZ_EXTENT GZ_EXTENT_i,GZ_EXTENT_j,GZ_EXTENT_k,GZ_EXTENT_l,GZ_EXTENT_m
// num elements in each direction + padding + ghosts
constexpr unsigned PADDED_EXTENT_i = GZ_EXTENT_i + 2 * PADDING_i;
constexpr unsigned PADDED_EXTENT_j = GZ_EXTENT_j + 2 * PADDING_j;
constexpr unsigned PADDED_EXTENT_k = GZ_EXTENT_k + 2 * PADDING_k;
constexpr unsigned PADDED_EXTENT_l = GZ_EXTENT_l + 2 * PADDING_l;
constexpr unsigned PADDED_EXTENT_m = GZ_EXTENT_m + 2 * PADDING_m;
constexpr unsigned NUM_PADDED_ELEMENTS = (PADDED_EXTENT_i * PADDED_EXTENT_j * PADDED_EXTENT_k * PADDED_EXTENT_l * PADDED_EXTENT_m);
#define PADDED_EXTENT PADDED_EXTENT_i,PADDED_EXTENT_j,PADDED_EXTENT_k,PADDED_EXTENT_l,PADDED_EXTENT_m
// figure out number of bricks in each direction
constexpr unsigned BRICK_EXTENT_i = GZ_EXTENT_i / BDIM_i;
constexpr unsigned BRICK_EXTENT_j = GZ_EXTENT_j / BDIM_j;
constexpr unsigned BRICK_EXTENT_k = GZ_EXTENT_k / BDIM_k;
constexpr unsigned BRICK_EXTENT_l = GZ_EXTENT_l / BDIM_l;
constexpr unsigned BRICK_EXTENT_m = GZ_EXTENT_m / BDIM_m;

// set our brick types
typedef Brick<Dim<BDIM_m, BDIM_l, BDIM_k, BDIM_j, BDIM_i>, Dim<2,4,4>, true> FieldBrick ;
typedef Brick<Dim<BDIM_m, BDIM_l, BDIM_k, BDIM_i>, Dim<2,4,4>, true> PreCoeffBrick;

// declare cuda kernels
__global__ void
ij_deriv_brick_kernel(unsigned (*fieldGrid)[BRICK_EXTENT_l][BRICK_EXTENT_k][BRICK_EXTENT_j][BRICK_EXTENT_i],
                      unsigned (*coeffGrid)[BRICK_EXTENT_l][BRICK_EXTENT_k][BRICK_EXTENT_i],
                      FieldBrick bIn,
                      FieldBrick bOut,
                      PreCoeffBrick bP1,
                      PreCoeffBrick bP2,
                      bComplexElem *ikj,
                      bElem *i_deriv_coeff);

#endif // BRICK_GENE_5D_H