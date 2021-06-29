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

constexpr unsigned PADDING_i = 8;
constexpr unsigned PADDING_j = 0;
constexpr unsigned PADDING_k = 8;
constexpr unsigned PADDING_l = 4;
constexpr unsigned PADDING_m = 0;
constexpr unsigned PADDING_n = 0;
#define PADDING PADDING_i,PADDING_j,PADDING_k,PADDING_l,PADDING_m,PADDING_n
constexpr unsigned GHOST_ZONE_i = 8;
constexpr unsigned GHOST_ZONE_j = 0;
constexpr unsigned GHOST_ZONE_k = 8;
constexpr unsigned GHOST_ZONE_l = 4;
constexpr unsigned GHOST_ZONE_m = 0;
constexpr unsigned GHOST_ZONE_n = 0;
#define GHOST_ZONE GHOST_ZONE_i,GHOST_ZONE_j,GHOST_ZONE_k,GHOST_ZONE_l,GHOST_ZONE_m,GHOST_ZONE_n

// blocking dimensions
constexpr unsigned DIM = 6;
constexpr unsigned TILE = 8;
// set brick sizes
constexpr unsigned BDIM_i = 8;
constexpr unsigned BDIM_j = 1;
constexpr unsigned BDIM_k = 8;
constexpr unsigned BDIM_l = 4;
constexpr unsigned BDIM_m = 1;
constexpr unsigned BDIM_n = 1;
constexpr unsigned NUM_ELEMENTS_PER_BRICK = BDIM_i * BDIM_j * BDIM_k * BDIM_l * BDIM_m * BDIM_n;
// num elements in each direction
constexpr unsigned EXTENT_i = 72;
constexpr unsigned EXTENT_j = 32;
constexpr unsigned EXTENT_k = 24;
constexpr unsigned EXTENT_l = 24;
constexpr unsigned EXTENT_m = 32;
constexpr unsigned EXTENT_n = 2;
#define EXTENT EXTENT_i,EXTENT_j,EXTENT_k,EXTENT_l,EXTENT_m,EXTENT_n
constexpr unsigned NUM_ELEMENTS = EXTENT_i * EXTENT_j * EXTENT_k * EXTENT_l * EXTENT_m * EXTENT_n;
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
typedef Brick<Dim<BDIM_n, BDIM_m, BDIM_l, BDIM_k, BDIM_j, BDIM_i>, Dim<4,4,1,4>, true> FieldBrick ;
typedef Brick<Dim<BDIM_n, BDIM_m, BDIM_l, BDIM_k, BDIM_i>, Dim<2,4,4>, true> PreCoeffBrick;
typedef Brick<Dim<BDIM_n, BDIM_m, BDIM_l, BDIM_k, BDIM_i>, Dim<2,4,4>> RealCoeffBrick;

// useful constants for stencil computations
constexpr unsigned ARAKAWA_STENCIL_SIZE = 13;

// used to copy i derivative coefficients into constant memory
void copy_i_deriv_coeff(const bElem i_deriv_coeff_host[5]);

// declare cuda kernels
__global__ void
ij_deriv_brick_kernel(unsigned (*fieldGrid)[GZ_BRICK_EXTENT_m][GZ_BRICK_EXTENT_l][GZ_BRICK_EXTENT_k][GZ_BRICK_EXTENT_j][GZ_BRICK_EXTENT_i],
                      unsigned (*coeffGrid)[GZ_BRICK_EXTENT_m][GZ_BRICK_EXTENT_l][GZ_BRICK_EXTENT_k][GZ_BRICK_EXTENT_i],
                      FieldBrick bIn,
                      FieldBrick bOut,
                      PreCoeffBrick bP1,
                      PreCoeffBrick bP2,
                      bComplexElem *ikj);

__global__ void
ij_deriv_brick_kernel_vec(unsigned (*fieldGrid)[GZ_BRICK_EXTENT_m][GZ_BRICK_EXTENT_l][GZ_BRICK_EXTENT_k][GZ_BRICK_EXTENT_j][GZ_BRICK_EXTENT_i],
                          unsigned (*coeffGrid)[GZ_BRICK_EXTENT_m][GZ_BRICK_EXTENT_l][GZ_BRICK_EXTENT_k][GZ_BRICK_EXTENT_i],
                          FieldBrick bIn,
                          FieldBrick bOut,
                          PreCoeffBrick bP1,
                          PreCoeffBrick bP2,
                          bComplexElem *ikj) ;

__global__ void
semi_arakawa_brick_kernel(unsigned (*fieldGrid)[GZ_BRICK_EXTENT_m][GZ_BRICK_EXTENT_l][GZ_BRICK_EXTENT_k][GZ_BRICK_EXTENT_j][GZ_BRICK_EXTENT_i],
                          unsigned (*coeffGrid)[GZ_BRICK_EXTENT_m][GZ_BRICK_EXTENT_l][GZ_BRICK_EXTENT_k][GZ_BRICK_EXTENT_i],
                          FieldBrick bIn,
                          FieldBrick bOut,
                          RealCoeffBrick *coeff);

#endif // BRICK_GENE_5D_H