//
// Created by Ben_Sepanski on 11/4/2021.
//

#include "cuda-test-utils.h"
#include "Array.h"
#include "BrickedArray.h"
#include "InterleavedBrickedArrays.h"
#include <gtest/gtest.h>

// useful type-defs
typedef Dim<4, 2, 1> BrickDims;
typedef brick::Array<int, 3> Array3D;
typedef brick::BrickedArray<double, BrickDims > BrickedArray3D;

/**
 * Test suite for consistency with communication in different
 * dimensions
 */
template<typename T>
class CopyToCudaTests : public testing::Test {
public:
  using Brick3D = BrickedArray3D::BrickType<T>;
};

using Comms = ::testing::Types<CommDims<false, false, false>,
                               CommDims<false, false, true>,
                               CommDims<false, true, false>,
                               CommDims<false, true, true>,
                               CommDims<true, false, false>,
                               CommDims<true, false, true>,
                               CommDims<true, true, false>,
                               CommDims<true, true, true>>;
TYPED_TEST_SUITE(CopyToCudaTests, Comms);

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

TYPED_TEST(CopyToCudaTests, ArrayToCudaTest) {
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

TYPED_TEST(CopyToCudaTests, BrickToCudaTest) {
  std::array<unsigned, 3> brickGridExtent = {2, 2, 2};
  brick::BrickLayout<3> layout(brickGridExtent);
  BrickedArray3D src(layout), dst(layout);
  testCopyToCuda<TypeParam>(layout, dst, src);
}

TYPED_TEST(CopyToCudaTests, BrickToCudaTest_mmap) {
  std::array<unsigned, 3> brickGridExtent = {2, 2, 2};
  brick::BrickLayout<3> layout(brickGridExtent);
  BrickedArray3D src(layout, nullptr), dst(layout, nullptr);
  testCopyToCuda<TypeParam>(layout, dst, src);
}

TYPED_TEST(CopyToCudaTests, InterleavedBrickToCudaTest) {
  std::array<unsigned, 3> brickGridExtent = {2, 2, 2};
  brick::BrickLayout<3> layout(brickGridExtent);
  brick::InterleavedBrickedArrays<BrickDims,
                                  brick::DataTypeVectorFoldPair<bElem>
                                  > srcAndDst(layout, 2);
  BrickedArray3D src = std::get<0>(srcAndDst.fields).front(),
                 dst = std::get<0>(srcAndDst.fields).back();
  testCopyToCuda<TypeParam>(layout, dst, src);
}
