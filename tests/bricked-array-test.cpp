#pragma clang diagnostic push
#pragma ide diagnostic ignored "cert-err58-cpp"
//
// Created by Ben_Sepanski on 10/15/2021.
//

#include <array>
#include <gtest/gtest.h>

#include "Array.h"
#include "BrickedArray.h"
#include "InterleavedBrickedArrays.h"

template<typename C, typename D>
struct CommunicatingDimsAndDataType {
  typedef C CommunicatingDims;
  typedef D DataType;
};

template<typename T>
class BrickedArray3DTests : public ::testing::Test {
protected:
  typedef typename T::DataType DataType;
  typedef Dim<4, 2, 1> BrickDims;
  typedef Dim<2, 1, 1> VectorFold;
  typedef typename T::CommunicatingDims CommunicatingDims;
  typedef brick::BrickedArray<DataType, BrickDims, VectorFold> BrickedArray3D;
};

typedef CommDims<false, false, false> NoComm;
typedef CommDims<false, false, true> Comm_i;
typedef CommDims<false, true, false> Comm_j;
typedef CommDims<false, true, true> Comm_ij;
typedef CommDims<true, false, false> Comm_k;
typedef CommDims<true, false, true> Comm_ik;
typedef CommDims<true, true, false> Comm_jk;
typedef CommDims<true, true, true> Comm_ijk;

using CommunicatingDimsAndDataTypes = ::testing::Types<
    CommunicatingDimsAndDataType<NoComm, bElem>,
    CommunicatingDimsAndDataType<NoComm, bComplexElem>,
    CommunicatingDimsAndDataType<Comm_i, bElem>,
    CommunicatingDimsAndDataType<Comm_i, bComplexElem>,
    CommunicatingDimsAndDataType<Comm_j, bElem>,
    CommunicatingDimsAndDataType<Comm_j, bComplexElem>,
    CommunicatingDimsAndDataType<Comm_ij, bElem>,
    CommunicatingDimsAndDataType<Comm_ij, bComplexElem>,
    CommunicatingDimsAndDataType<Comm_k, bElem>,
    CommunicatingDimsAndDataType<Comm_k, bComplexElem>,
    CommunicatingDimsAndDataType<Comm_ik, bElem>,
    CommunicatingDimsAndDataType<Comm_ik, bComplexElem>,
    CommunicatingDimsAndDataType<Comm_jk, bElem>,
    CommunicatingDimsAndDataType<Comm_jk, bComplexElem>,
    CommunicatingDimsAndDataType<Comm_ijk, bElem>,
    CommunicatingDimsAndDataType<Comm_ijk, bComplexElem>
    >;
TYPED_TEST_SUITE(BrickedArray3DTests, CommunicatingDimsAndDataTypes);

TYPED_TEST(BrickedArray3DTests, MemoryLayoutTest) {
  // convenient typedefs
  typedef typename TestFixture::BrickedArray3D BrickedArray3D;
  typedef typename TestFixture::BrickDims BrickDims;
  typedef typename TestFixture::CommunicatingDims CommunicatingDims;
  typedef typename TestFixture::DataType DataType;
  // set up bricked array
  std::array<unsigned, BrickedArray3D::RANK> extent{};
  for(unsigned d = 0; d < extent.size(); ++d) {
    extent[d] = 3 * BrickedArray3D::BRICK_DIMS[d];
  }
  brick::BrickLayout<3> layout(extent);
  BrickedArray3D bArr(layout);

  // assign each brickedArray entry to unique value
  int index = 1;
  for (unsigned k = 0; k < extent[2]; ++k) {
    for (unsigned j = 0; j < extent[1]; ++j) {
      for (unsigned i = 0; i < extent[0]; ++i) {
        bArr(i, j, k) = index;
        index++;
      }
    }
  }

  // Get a handle on the actual bricks
  auto bricks = bArr.template viewBricks<CommunicatingDims>();
  // make sure the values stored in the bricks are what we expect
  auto BDIM = BrickedArray3D::BRICK_DIMS;
  for(unsigned bk = 0; bk < layout.indexInStorage.extent[2]; ++bk) {
    for(unsigned bj = 0; bj < layout.indexInStorage.extent[1]; ++bj) {
      for(unsigned bi = 0; bi < layout.indexInStorage.extent[0]; ++bi) {
        unsigned b = layout.indexInStorage.get(bi, bj, bk);
        for(unsigned k = 0; k < BDIM[2]; ++k) {
          for(unsigned j = 0; j < BDIM[1]; ++j) {
            for(unsigned i = 0; i < BDIM[0]; ++i) {
              DataType value = bricks[b][k][j][i];
              EXPECT_EQ(value, bArr(bi * BDIM[0] + i, bj * BDIM[1] + j, bk * BDIM[2] + k));
            }
          }
        }
      }
    }
  }
}

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
