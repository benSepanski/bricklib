//
// Created by Tuowen Zhao on 12/5/18.
//

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

#define PADDING_i 8
#define PADDING_j 8
#define PADDING_k 8
#define PADDING_l 8
#define PADDING_m 8
#define PADDING PADDING_i,PADDING_j,PADDING_k,PADDING_l,PADDING_m
#define GHOST_ZONE_i 0
#define GHOST_ZONE_j 0
#define GHOST_ZONE_k 0
#define GHOST_ZONE_l 0
#define GHOST_ZONE_m 0
#define GHOST_ZONE GHOST_ZONE_i,GHOST_ZONE_j,GHOST_ZONE_k,GHOST_ZONE_l,GHOST_ZONE_m
// tiling
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
// num elements in each direction + padding + ghosts
#define PADDED_EXTENT_i EXTENT_i + 2 * (PADDING_i + GHOST_ZONE_i)
#define PADDED_EXTENT_j EXTENT_j + 2 * (PADDING_j + GHOST_ZONE_j)
#define PADDED_EXTENT_k EXTENT_k + 2 * (PADDING_k + GHOST_ZONE_k)
#define PADDED_EXTENT_l EXTENT_l + 2 * (PADDING_l + GHOST_ZONE_l)
#define PADDED_EXTENT_m EXTENT_m + 2 * (PADDING_m + GHOST_ZONE_m)
#define PADDED_EXTENT PADDED_EXTENT_i,PADDED_EXTENT_j,PADDED_EXTENT_k,PADDED_EXTENT_l,PADDED_EXTENT_m
// number of blocks (including padded blocks)
#define BRICK_GRID_EXTENT_i (EXTENT_i + 2 * GHOST_ZONE_i) / BDIM_i
#define BRICK_GRID_EXTENT_j (EXTENT_j + 2 * GHOST_ZONE_j) / BDIM_j
#define BRICK_GRID_EXTENT_k (EXTENT_k + 2 * GHOST_ZONE_k) / BDIM_k
#define BRICK_GRID_EXTENT_l (EXTENT_l + 2 * GHOST_ZONE_l) / BDIM_l
#define BRICK_GRID_EXTENT_m (EXTENT_m + 2 * GHOST_ZONE_m) / BDIM_m
#define BRICK_GRID_EXTENT BRICK_GRID_EXTENT_i,BRICK_GRID_EXTENT_j,BRICK_GRID_EXTENT_k,BRICK_GRID_EXTENT_l,BRICK_GRID_EXTENT_m

// relevant types
using ComplexElement5DArray = bComplexElem(*) [PADDED_EXTENT_l][PADDED_EXTENT_k][PADDED_EXTENT_j][PADDED_EXTENT_i];
using Brick5DGrid = unsigned(*) [BRICK_GRID_EXTENT_l][BRICK_GRID_EXTENT_k][BRICK_GRID_EXTENT_j][BRICK_GRID_EXTENT_i];
using Brick5D = Brick<Dim<BDIM>, Dim<VFOLD> >;
using ComplexBrick5D = Brick<Dim<BDIM>, Dim<VFOLD>, true>;


/**
 * @brief 1-D stencil fused with multiplication by 1-D array
 * 
 * Based on https://github.com/wdmapp/gtensor/blob/d07000b15d253cdeb44942b52f3d2caf4522faa0/benchmarks/ij_deriv.cxx
 */
void ij_deriv() {
  // setup brick-info and raray of bricks on the device
  unsigned *grid_ptr;
  const unsigned long NUM_ELEMENTS = EXTENT_i * EXTENT_j * EXTENT_k * EXTENT_l * EXTENT_m;
  const unsigned long NUM_BRICKS = BRICK_GRID_EXTENT_i *
                                   BRICK_GRID_EXTENT_j *
                                   BRICK_GRID_EXTENT_k *
                                   BRICK_GRID_EXTENT_l *
                                   BRICK_GRID_EXTENT_m;

  auto bInfo = init_grid<DIM>(grid_ptr, {BRICK_GRID_EXTENT});
  unsigned *grid_dev;
  {
    unsigned size = NUM_BRICKS * sizeof(unsigned);
    cudaMalloc(&grid_dev, size);
    cudaMemcpy(grid_dev, grid_ptr, size, cudaMemcpyHostToDevice);
  }
  Brick5DGrid grid = (Brick5DGrid) grid_dev;
  BrickInfo<DIM> *bInfo_dev;
  BrickInfo<DIM> _bInfo_dev = movBrickInfo(bInfo, cudaMemcpyHostToDevice);
  {
    unsigned size = sizeof(BrickInfo < DIM > );
    cudaMalloc(&bInfo_dev, size);
    cudaMemcpy(bInfo_dev, &_bInfo_dev, size, cudaMemcpyHostToDevice);
  }

  // build arrays to hold our input/output fields
  unsigned size = NUM_ELEMENTS * sizeof(bComplexElem);
  bComplexElem *in_ptr = randomComplexArray({PADDED_EXTENT});
  bComplexElem *out_ptr = zeroComplexArray({PADDED_EXTENT});
  ComplexElement5DArray arr_in = (ComplexElement5DArray) in_ptr;
  ComplexElement5DArray arr_out = (ComplexElement5DArray) out_ptr;

  // move our coefficients and arrays onto the device
  // (4th order centered difference)
  bElem i_deriv_coeff[] = {1.0 / 12.0, -2.0 / 3.0, 0.0, 2.0 / 3.0, -1.0 / 12.0};
  bComplexElem *i_deriv_coeff_dev;
  {
    cudaMalloc(&i_deriv_coeff_dev, sizeof(i_deriv_coeff));
    cudaMemcpy(i_deriv_coeff_dev, i_deriv_coeff, sizeof(i_deriv_coeff), cudaMemcpyHostToDevice);
  }

  // build coefficients for Fourier derivatives and put on device
  bComplexElem j_scale[EXTENT_j];
  #pragma omp parallel
  for(int j = 0; j < EXTENT_j; ++j)
  {
    j_scale[j] = bComplexElem(0, 2 * pi * j);
  }
  bComplexElem *j_scale_dev;
  {
    cudaMalloc(&j_scale_dev, sizeof(j_scale));
    cudaMemcpy(j_scale_dev, j_scale, sizeof(j_scale), cudaMemcpyHostToDevice);
  }

  // copy input/output fields to device
  bComplexElem *in_dev, *out_dev;
  {
    cudaMalloc(&in_dev, size);
    cudaMemcpy(in_dev, in_ptr, size, cudaMemcpyHostToDevice);
  }
  {
    cudaMalloc(&out_dev, size);
    cudaMemcpy(out_dev, out_ptr, size, cudaMemcpyHostToDevice);
  }

  // move input field into bricks and copy to device
  auto bSize = ComplexBrick5D::BRICKSIZE;
  auto bStorage = BrickStorage::allocate(bInfo.nbricks, bSize * 2);
  ComplexBrick5D bIn(&bInfo, bStorage, 0);
  ComplexBrick5D bOut(&bInfo, bStorage, bSize);

  copyToBrick<DIM>({EXTENT}, {PADDING}, {GHOST_ZONE}, in_ptr, grid_ptr, bIn);

  BrickStorage bStorage_dev = movBrickStorage(bStorage, cudaMemcpyHostToDevice);

  auto arr_func = [&arr_in, &arr_out, &i_deriv_coeff, &j_scale]() -> void {
    _TILEFOR5D {
      arr_out[m][l][k][j][i] = j_scale[j] * (
        i_deriv_coeff[0] * arr_in[m][l][k][j][i - 2] +
        i_deriv_coeff[1] * arr_in[m][l][k][j][i - 1] +
        i_deriv_coeff[2] * arr_in[m][l][k][j][i] +
        i_deriv_coeff[3] * arr_in[m][l][k][j][i + 1] +
        i_deriv_coeff[4] * arr_in[m][l][k][j][i + 2]
        );
    }
  };

  std::cout << "i-j derivative" << std::endl;
  // perform stencil computations
  arr_func();
  std::cout << "done" << std::endl;

  // free memory
  free(in_ptr);
  free(out_ptr);
  free(grid_ptr);
  free(bInfo.adj);
  cudaFree(_bInfo_dev.adj);
  cudaFree(in_dev);
  cudaFree(out_dev);
}

int main() {
  // std::random_device r;
  // std::mt19937_64 mt(r());
  // std::uniform_real_distribution<bElem> u(0, 1);
  ij_deriv();
  return 0;
}