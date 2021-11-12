//
// Created by Ben_Sepanski on 11/11/2021.
//

#ifndef BRICK_CUDA_TEST_UTILS_H
#define BRICK_CUDA_TEST_UTILS_H

#include "BrickedArray.h"
#include <gtest/gtest.h>

namespace { // begin anonymous namespace
/**
 * Store dst into src
 *
 * @tparam BrickType type of the brick
 * @param dst destination
 * @param src source
 */
template <typename BrickType>
__global__ void
brickedArrayCopy(brick::Array<unsigned, 3, brick::Padding<>, unsigned> grid,
                 BrickType dst, BrickType src) {
  unsigned b = grid(blockIdx.x, blockIdx.y, blockIdx.z);
  for (unsigned k = 0; k < blockDim.z; ++k) {
    for (unsigned j = 0; j < blockDim.y; ++j) {
      for (unsigned i = 0; i < blockDim.x; ++i) {
        assert(src[b][k][j][i] != 0.0);
        dst[b][k][j][i] = src[b][k][j][i];
      }
    }
  }
}
} // end anonymous namespace

/**
 * Copy dst and src to cuda, store src to dst on the device,
 * and copy dst back from the device
 *
 * @tparam CommunicatingDims see BrickInfo
 * @param layout layout of src and dst
 * @param dst destination array
 * @param src source array
 */
template<typename CommunicatingDims, typename DataType, typename BrickDims>
void testCopyToCuda(brick::BrickLayout<3> layout,
                    brick::BrickedArray<DataType, BrickDims> dst,
                    brick::BrickedArray<DataType, BrickDims> src) {
  typedef brick::BrickedArray<DataType, BrickDims> BrickedArray3D;
  // make sure arrays start out unequal
  std::array<unsigned, 3> extent{};
  for(unsigned d = 0; d < 3; ++d) {
    extent[d] = layout.indexInStorage.extent[d] * BrickedArray3D::BRICK_DIMS[d];
  }
  int index = 1;
  for(unsigned k = 0; k < extent[2]; ++k) {
    for(unsigned j = 0; j < extent[1]; ++j) {
      for(unsigned i = 0; i < extent[0]; ++i) {
        src(i, j, k) = index++;
        dst(i, j, k) = -1;
        EXPECT_NE(src(i, j, k), dst(i, j, k));
      }
    }
  }

  // Copy to device and reset src
  src.copyToDevice();
  for(unsigned k = 0; k < extent[2]; ++k) {
    for(unsigned j = 0; j < extent[1]; ++j) {
      for(unsigned i = 0; i < extent[0]; ++i) {
        src(i, j, k) = -2;
      }
    }
  }
  // look at src/dst from device perspective
  auto bricksSrc_dev = src.template viewBricksOnDevice<CommunicatingDims>(),
       bricksDst_dev = dst.template viewBricksOnDevice<CommunicatingDims>();
  dim3 gridSize{layout.indexInStorage.extent[0],
                layout.indexInStorage.extent[1],
                layout.indexInStorage.extent[2]};
  dim3 blockSize{BrickedArray3D::BRICK_DIMS[0],
                 BrickedArray3D::BRICK_DIMS[1],
                 BrickedArray3D::BRICK_DIMS[2]} ;
  brick::Array<unsigned, 3, brick::Padding<>, unsigned> grid_dev = layout.indexInStorage.allocateOnDevice();
  layout.indexInStorage.copyToDevice(grid_dev);
  brickedArrayCopy<< <gridSize, blockSize>> >(grid_dev, bricksDst_dev, bricksSrc_dev);
  src.copyFromDevice();
  dst.copyFromDevice();

  index = 1;
  for(unsigned k = 0; k < extent[2]; ++k) {
    for(unsigned j = 0; j < extent[1]; ++j) {
      for(unsigned i = 0; i < extent[0]; ++i) {
        EXPECT_EQ(src(i, j, k), index++); //< src should not have changed
        EXPECT_EQ(src(i, j, k), dst(i, j, k));
      }
    }
  }
}

#endif // BRICK_CUDA_TEST_UTILS_H
