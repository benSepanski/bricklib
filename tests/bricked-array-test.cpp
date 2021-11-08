#pragma clang diagnostic push
#pragma ide diagnostic ignored "cert-err58-cpp"
//
// Created by Ben_Sepanski on 10/15/2021.
//

#include <array>
#include <gtest/gtest.h>

#include "Array.h"
#include "bricked-array.h"

TEST(BrickedArrayTests, AssignmentTest) {
  constexpr unsigned RANK = 3;
  typedef brick::BrickedArray<bElem, Dim<2,2,2>> BrickedArray;
  typedef brick::Array<bElem, RANK> Array;

  // Build our bricked array
  std::array<unsigned, RANK> brickGridExtent = {3, 3, 3};
  brick::BrickLayout<RANK> layout(brickGridExtent);
  BrickedArray brickedArray(layout);

  // Build an array to copy into the brick
  std::array<unsigned, RANK> extent{};
  for(unsigned d = 0; d < RANK; ++d) {
    extent[d] = brickGridExtent[d] * BrickedArray::BRICK_DIMS[d];
  }
  Array arr(extent);
  for(unsigned k = 0; k < extent[2]; ++k) {
    for (unsigned j = 0; j < extent[1]; ++j) {
      for (unsigned i = 0; i < extent[0]; ++i) {
        arr(i, j, k) = i + extent[0] * (j + extent[1] * k);
      }
    }
  }

  // Test the copy from array into the brick
  brickedArray.loadFrom(arr);
  for(unsigned k = 0; k < extent[2]; ++k) {
    for(unsigned j = 0; j < extent[1]; ++j) {
      for (unsigned i = 0; i < extent[0]; ++i) {
        bElem value = brickedArray(i, j, k);
        EXPECT_EQ(value, i + extent[0] * (j + extent[1] * k));
      }
    }
  }

  // test store-to
  Array arrCopy(extent);
  brickedArray.storeTo(arrCopy);
  for(unsigned k = 0; k < extent[2]; ++k)
  for(unsigned j = 0; j < extent[1]; ++j)
  for(unsigned i = 0; i < extent[0]; ++i) {
    EXPECT_EQ(arr(i, j, k), arrCopy(i, j, k));
  }
}

TEST(BrickedArrayTests, InterleavedFieldTest) {
  constexpr unsigned RANK = 3;
  typedef Dim<2,2,2> BDims;

  // Build our interleaved interleavedBrickedArrays
  std::array<unsigned, RANK> brickGridExtent = {3, 3, 3};
  brick::BrickLayout<RANK> layout(brickGridExtent);
  typedef
    brick::InterleavedBrickedArrays<BDims,
                                    brick::DataTypeVectorFoldPair<bElem, Dim<2> >,
                                    brick::DataTypeVectorFoldPair<bComplexElem>
                                    >
    InterleavedBrickedArrays;
  InterleavedBrickedArrays interleavedBrickedArrays(layout, 1, 2);
  brick::BrickedArray<bElem, BDims, Dim<2>> realArr =
      std::get<0>(interleavedBrickedArrays.fields).front();
  brick::BrickedArray<bComplexElem, BDims> complexArr1 =
      std::get<1>(interleavedBrickedArrays.fields).front(),
     complexArr2 = std::get<1>(interleavedBrickedArrays.fields).back();

  for(unsigned k = 0; k < realArr.extent[2]; ++k) {
    for(unsigned j = 0; j < realArr.extent[1]; ++j) {
      for(unsigned i = 0; i < realArr.extent[0]; ++i) {
        realArr(i, j, k) = i % brick::BrickedArray<double, BDims, Dim<2>>::BRICK_DIMS[0]
                         + 10 * (j % brick::BrickedArray<double, BDims, Dim<2>>::BRICK_DIMS[1])
                         + 100 * (k % brick::BrickedArray<double, BDims, Dim<2>>::BRICK_DIMS[2]);
        complexArr1(i, j, k) = std::complex<bElem>(0, -realArr(i, j, k));
        complexArr2(i, j, k) = std::complex<bElem>(-realArr(i, j, k), 0);
      }
    }
  }

  Brick<BDims,Dim<2> > realBricks = realArr.viewBricks();
  Brick<BDims,Dim<1>,true>  complexBricks1 = complexArr1.viewBricks(),
                            complexBricks2 = complexArr2.viewBricks();
  for(unsigned b = 0; b < layout.size(); ++b) {
    for (unsigned k = 0; k < brick::BrickedArray<double, BDims, Dim<2>>::BRICK_DIMS[2]; ++k) {
      for (unsigned j = 0; j < brick::BrickedArray<double, BDims, Dim<2>>::BRICK_DIMS[1]; ++j) {
        for (unsigned i = 0; i < brick::BrickedArray<double, BDims, Dim<2>>::BRICK_DIMS[0]; ++i) {
          EXPECT_EQ(realBricks[b][k][j][i], i + 10 * j + 100 * k);
          EXPECT_EQ(complexBricks1[b][k][j][i], bComplexElem(0, -realBricks[b][k][j][i]));
          EXPECT_EQ(complexBricks2[b][k][j][i], bComplexElem(-realBricks[b][k][j][i], 0));
        }
      }
    }
  }
}
