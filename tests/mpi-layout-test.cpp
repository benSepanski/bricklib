//
// Created by Ben_Sepanski on 11/9/2021.
//

#include "MPILayout.h"
#include "MPIEnvironment.h"
#include "MPICartesianSetup_3D.h"

using Comms = ::testing::Types<CommDims<false, false, true>,
                               CommDims<false, true, false>,
                               CommDims<false, true, true>,
                               CommDims<true, false, false>,
                               CommDims<true, false, true>,
                               CommDims<true, true, false>,
                               CommDims<true, true, true>>;
TYPED_TEST_SUITE(MPI_CartesianTest3D, Comms);

TYPED_TEST(MPI_CartesianTest3D, ExchangeNoMMAPTest) {
  typedef typename TestFixture::BrickDims BrickDims;
  typedef typename TestFixture::Region Region;
  typedef TypeParam CommunicatingDims;
  // build a brick from the layout
  brick::MPILayout<BrickDims, CommunicatingDims> mpiLayout(this->buildMPILayout());
  brick::BrickedArray<bElem, BrickDims> bArr(mpiLayout.getBrickLayout());
  this->template assignEntriesToRegion(bArr);

  // exchange
  mpiLayout.exchangeWithoutMMAP(bArr);
  // Now make sure that ghosts received the appropriate regionTag
  // (NOTE: THIS RELIES ON THE CARTESIAN COMM BEING PERIODIC)
  for (unsigned k = 0; k < this->extentWithGZ[2]; ++k) {
    for (unsigned j = 0; j < this->extentWithGZ[1]; ++j) {
      for (unsigned i = 0; i < this->extentWithGZ[0]; ++i) {
        Region r = this->getRegion(i, j, k);
        EXPECT_EQ(bArr(i, j, k), r.regionID);
      }
    }
  }
}

TYPED_TEST(MPI_CartesianTest3D, ExchangeMMAPTest) {
  typedef typename TestFixture::BrickDims BrickDims;
  typedef typename TestFixture::Region Region;
  typedef TypeParam CommunicatingDims;

  brick::MPILayout<BrickDims, CommunicatingDims> mpiLayout(this->buildMPILayout());
  // build a brick from the layout using mmap
  brick::BrickedArray<bElem, BrickDims> bArr(mpiLayout.getBrickLayout(), nullptr);
  this->template assignEntriesToRegion(bArr);
  // exchange
  ExchangeView ev = mpiLayout.buildExchangeView(bArr);
  ev.exchange();
  // Now make sure that ghosts received the appropriate regionTag
  // (NOTE: THIS RELIES ON THE CARTESIAN COMM BEING PERIODIC)
  for (unsigned k = 0; k < this->extentWithGZ[2]; ++k) {
    for (unsigned j = 0; j < this->extentWithGZ[1]; ++j) {
      for (unsigned i = 0; i < this->extentWithGZ[0]; ++i) {
        Region r = this->getRegion(i, j, k);
        EXPECT_EQ(bArr(i, j, k), r.regionID);
      }
    }
  }
}

TYPED_TEST(MPI_CartesianTest3D, LoadFromStoreToTest) {
  typedef typename TestFixture::BrickDims BrickDims;
  typedef TypeParam CommunicatingDims;

  // build a brick from the layout
  brick::MPILayout<BrickDims, CommunicatingDims> mpiLayout(this->buildMPILayout());
  brick::BrickLayout<3> layout = mpiLayout.getBrickLayout();
  brick::BrickedArray<bElem, BrickDims> bArr1(layout), bArr2(layout);
  brick::Array<bElem, 3> arr(this->extentWithGZ);

  int index = 1;
  for (unsigned k = 0; k < this->extentWithGZ[2]; ++k) {
    for (unsigned j = 0; j <  this->extentWithGZ[1]; ++j) {
      for (unsigned i = 0; i < this->extentWithGZ[0]; ++i) {
        arr(i, j, k) = ++index;
        bArr1(i, j, k) = -index;
      }
    }
  }
  bArr2.loadFrom(arr);
  bArr1.storeTo(arr);
  index = 1;
  for (unsigned k = 0; k < this->extentWithGZ[2]; ++k) {
    for (unsigned j = 0; j < this->extentWithGZ[1]; ++j) {
      for (unsigned i = 0; i < this->extentWithGZ[0]; ++i) {
        index++;
        EXPECT_EQ(arr(i, j, k), -index);
        EXPECT_EQ(bArr2(i, j, k), index);
      }
    }
  }
}

TYPED_TEST(MPI_CartesianTest3D, GridIsCorrectTest) {
  typedef typename TestFixture::BrickDims BrickDims;
  typedef TypeParam CommunicatingDims;

  // build a layout using MPI
  brick::MPILayout<BrickDims, CommunicatingDims> mpiLayout(this->buildMPILayout());
  brick::BrickLayout<3> layout = mpiLayout.getBrickLayout();

  // make sure the grids match
  auto brickDecompPtr = mpiLayout.getBrickDecompPtr();
  for(unsigned k = 0; k < layout.indexInStorage.extent[2]; ++k) {
    for(unsigned j = 0; j < layout.indexInStorage.extent[1]; ++j) {
      for(unsigned i = 0; i < layout.indexInStorage.extent[0]; ++i) {
        unsigned expectedIndex = (*brickDecompPtr)[k][j][i];
        unsigned actualIndex = layout.indexInStorage.get(i, j, k);
        EXPECT_EQ(actualIndex, expectedIndex);
      }
    }
  }
}

TYPED_TEST(MPI_CartesianTest3D, MemoryLayoutTest) {
  typedef typename TestFixture::BrickDims BrickDims;
  typedef TypeParam CommunicatingDims;

  // build a brick from the layout
  brick::MPILayout<BrickDims, CommunicatingDims> mpiLayout(this->buildMPILayout());
  brick::BrickLayout<3> layout = mpiLayout.getBrickLayout();
  brick::BrickedArray<bElem, BrickDims> bArr(layout);

  // assign each brickedArray entry to unique value
  int index = 1;
  for (unsigned k = 0; k < bArr.extent[2]; ++k) {
    for (unsigned j = 0; j < bArr.extent[1]; ++j) {
      for (unsigned i = 0; i < bArr.extent[0]; ++i) {
        bArr(i, j, k) = index;
        index++;
      }
    }
  }

  // Get a handle on the actual bricks
  auto bricks = bArr.template viewBricks<CommunicatingDims>();
  auto bricksNoComm = bArr.template viewBricks<CommDims<false, false, false> >();
  // make sure the values stored in the bricks are what we expect
  auto BDIM = brick::BrickedArray<bElem, BrickDims>::BRICK_DIMS;
  for(unsigned bk = 0; bk < layout.indexInStorage.extent[2]; ++bk) {
    for(unsigned bj = 0; bj < layout.indexInStorage.extent[1]; ++bj) {
      for(unsigned bi = 0; bi < layout.indexInStorage.extent[0]; ++bi) {
        unsigned b = layout.indexInStorage.get(bi, bj, bk);
        ASSERT_EQ(b, (*mpiLayout.getBrickDecompPtr())[bk][bj][bi]);
        for(unsigned k = 0; k < BDIM[2]; ++k) {
          for(unsigned j = 0; j < BDIM[1]; ++j) {
            for(unsigned i = 0; i < BDIM[0]; ++i) {
              bElem *addressFromBrickList = &bricks[b][k][j][i];
              bElem *addressFromNoCommBrickList = &bricksNoComm[b][k][j][i];
              bElem *addressFromBrickedArray = &bArr(bi * BDIM[0] + i,
                                                     bj * BDIM[1] + j,
                                                     bk * BDIM[2] + k);
              EXPECT_EQ(addressFromBrickList, addressFromNoCommBrickList);
              EXPECT_EQ(addressFromNoCommBrickList, addressFromBrickedArray);
            }
          }
        }
      }
    }
  }
}

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
  return RUN_ALL_TESTS();
}