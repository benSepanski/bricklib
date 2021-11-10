//
// Created by Ben_Sepanski on 11/4/2021.
//

#include "Array.h"
#include "BrickedArray.h"
#include "InterleavedBrickedArrays.h"
#include <gtest/gtest.h>

typedef brick::Array<int, 3> Array3D;
typedef brick::BrickedArray<double, Dim<2,2,2> > BrickedArray3D;
typedef BrickedArray3D::BrickType<> Brick3D;

__global__
void arrayCopy(Array3D dst, Array3D src)
{
  for(unsigned k = 0; k < blockDim.z; ++k) {
    for(unsigned j = 0; j < blockDim.y; ++j) {
      for(unsigned i = 0; i < blockDim.x; ++i) {
        dst(i, j, k) = src(i, j, k);
      }
    }
  }
}

TEST(CudaCopyTests, ArrayToCudaTest) {
  std::array<unsigned, 3> extent = {3, 3, 3};
  Array3D src(extent), dst(extent);
  int index = 0;
  for(auto &val : src) {
    val = index++;
  }
  for(auto &val : dst) {
    val = -1;
  }
  // make sure arrays start out unequal
  for(unsigned k = 0; k < extent[2]; ++k) {
    for(unsigned j = 0; j < extent[1]; ++j) {
      for(unsigned i = 0; i < extent[0]; ++i) {
        EXPECT_NE(src(i, j, k), dst(i, j, k));
      }
    }
  }

  Array3D src_dev = src.allocateOnDevice(),
          dst_dev = dst.allocateOnDevice();
  src.copyToDevice(src_dev);
  dim3 blockSize;
  blockSize.z = extent[2];
  blockSize.y = extent[1];
  blockSize.x = extent[0];
  arrayCopy<< <1, blockSize>> >(dst_dev, src_dev);
  dst.copyFromDevice(dst_dev);

  for(unsigned k = 0; k < extent[2]; ++k) {
    for(unsigned j = 0; j < extent[1]; ++j) {
      for(unsigned i = 0; i < extent[0]; ++i) {
        EXPECT_EQ(src(i, j, k), dst(i, j, k));
      }
    }
  }
}

__global__
void brickedArrayCopy(Brick3D dst, Brick3D src)
{
  unsigned b = blockIdx.x;
  for(unsigned k = 0; k < blockDim.z; ++k) {
    for(unsigned j = 0; j < blockDim.y; ++j) {
      for(unsigned i = 0; i < blockDim.x; ++i) {
        dst[b][k][j][i] = src[b][k][j][i];
      }
    }
  }
}

TEST(CudaCopyTests, BrickToCudaTest) {
  std::array<unsigned, 3> brickGridExtent = {2, 2, 2};
  brick::BrickLayout<3> layout(brickGridExtent);
  BrickedArray3D src(layout), dst(layout);
  // make sure arrays start out unequal
  std::array<unsigned, 3> extent{};
  for(unsigned d = 0; d < brickGridExtent.size(); ++d) {
    extent[d] = brickGridExtent[d] * BrickedArray3D::BRICK_DIMS[d];
  }
  int index = 0;
  for(unsigned k = 0; k < extent[2]; ++k) {
    for(unsigned j = 0; j < extent[1]; ++j) {
      for(unsigned i = 0; i < extent[0]; ++i) {
        src(i, j, k) = index++;
        dst(i, j, k) = -1;
        EXPECT_NE(src(i, j, k), dst(i, j, k));
      }
    }
  }

  // Copy to device
  src.copyToDevice();
  dst.copyToDevice();
  Brick3D bricksSrc_dev = src.template viewBricksOnDevice<>(),
          bricksDst_dev = dst.template viewBricksOnDevice<>();
  dim3 blockSize{BrickedArray3D::BRICK_DIMS[0],
                 BrickedArray3D::BRICK_DIMS[1],
                 BrickedArray3D::BRICK_DIMS[2]} ;
  brickedArrayCopy<< <layout.size(), blockSize>> >(bricksDst_dev, bricksSrc_dev);
  dst.copyFromDevice();

  for(unsigned k = 0; k < extent[2]; ++k) {
    for(unsigned j = 0; j < extent[1]; ++j) {
      for(unsigned i = 0; i < extent[0]; ++i) {
        EXPECT_EQ(src(i, j, k), dst(i, j, k));
      }
    }
  }
}

TEST(CudaCopyTests, InterleavedBrickToCudaTest) {
  std::array<unsigned, 3> brickGridExtent = {2, 2, 2};
  brick::BrickLayout<3> layout(brickGridExtent);
  brick::InterleavedBrickedArrays<Dim<2,2,2>,
                                  brick::DataTypeVectorFoldPair<bElem>
                                  > srcAndDst(layout, 2);
  BrickedArray3D src = std::get<0>(srcAndDst.fields).front(),
                 dst = std::get<0>(srcAndDst.fields).back();
  // make sure arrays start out unequal
  std::array<unsigned, 3> extent{};
  for(unsigned d = 0; d < brickGridExtent.size(); ++d) {
    extent[d] = brickGridExtent[d] * BrickedArray3D::BRICK_DIMS[d];
  }
  int index = 0;
  for(unsigned k = 0; k < extent[2]; ++k) {
    for(unsigned j = 0; j < extent[1]; ++j) {
      for(unsigned i = 0; i < extent[0]; ++i) {
        src(i, j, k) = index++;
        dst(i, j, k) = -1;
        EXPECT_NE(src(i, j, k), dst(i, j, k));
      }
    }
  }

  // Copy to device
  src.copyToDevice();
  Brick3D bricksSrc_dev = src.viewBricksOnDevice(),
          bricksDst_dev = dst.viewBricksOnDevice();
  dim3 blockSize{BrickedArray3D::BRICK_DIMS[0],
                 BrickedArray3D::BRICK_DIMS[1],
                 BrickedArray3D::BRICK_DIMS[2]} ;
  brickedArrayCopy<< <layout.size(), blockSize>> >(bricksDst_dev, bricksSrc_dev);
  dst.copyFromDevice();

  for(unsigned k = 0; k < extent[2]; ++k) {
    for(unsigned j = 0; j < extent[1]; ++j) {
      for(unsigned i = 0; i < extent[0]; ++i) {
        EXPECT_EQ(src(i, j, k), dst(i, j, k));
      }
    }
  }
}
