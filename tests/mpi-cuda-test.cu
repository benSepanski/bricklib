//
// Created by Ben_Sepanski on 11/9/2021.
//

#include "cuda-test-utils.h"
#include "MPILayout.h"
#include "MPIEnvironment.h"
#include "MPICartesianSetup_3D.h"
#include <gtest/gtest.h>

using Comms = ::testing::Types<CommDims<false, false, true>,
                               CommDims<false, true, false>,
                               CommDims<false, true, true>,
                               CommDims<true, false, false>,
                               CommDims<true, false, true>,
                               CommDims<true, true, false>,
                               CommDims<true, true, true>>;
TYPED_TEST_SUITE(MPI_CartesianTest3D, Comms);

TYPED_TEST(MPI_CartesianTest3D, CopyToFromCudaIsIdentity) {
  typedef typename TestFixture::BrickDims BrickDims;
  typedef TypeParam CommunicatingDims;

  // build bricked arrays from the layout
  brick::MPILayout<BrickDims, CommunicatingDims> mpiLayout(this->buildMPILayout());
  brick::BrickLayout<3> layout = mpiLayout.getBrickLayout();
  brick::BrickedArray<bElem, BrickDims> arr(layout);
  int index = 1;
  for(unsigned k = 0; k < arr.extent[2]; ++k) {
    for(unsigned j = 0; j < arr.extent[1]; ++j) {
      for(unsigned i = 0; i < arr.extent[0]; ++i) {
        arr(i, j, k) = index++;
      }
    }
  }
  arr.copyToDevice();
  // clear host storage
  index = 1;
  for(unsigned k = 0; k < arr.extent[2]; ++k) {
    for(unsigned j = 0; j < arr.extent[1]; ++j) {
      for(unsigned i = 0; i < arr.extent[0]; ++i) {
        arr(i, j, k) = 0.0;
        EXPECT_NE(arr(i, j, k), index++);
      }
    }
  }
  // copy back from device
  arr.copyFromDevice();
  // test that we got the expected data
  index = 1;
  for(unsigned k = 0; k < arr.extent[2]; ++k) {
    for(unsigned j = 0; j < arr.extent[1]; ++j) {
      for(unsigned i = 0; i < arr.extent[0]; ++i) {
        EXPECT_EQ(arr(i, j, k), index++);
      }
    }
  }
}

TYPED_TEST(MPI_CartesianTest3D, CopyToCuda) {
  typedef typename TestFixture::BrickDims BrickDims;
  typedef TypeParam CommunicatingDims;

  // build bricked arrays from the layout
  brick::MPILayout<BrickDims, CommunicatingDims> mpiLayout(this->buildMPILayout());
  brick::BrickLayout<3> layout = mpiLayout.getBrickLayout();
  brick::BrickedArray<bElem, BrickDims> src(layout), dst(layout);

  // test the equality
  testCopyToCuda<CommunicatingDims>(layout, dst, src);
}

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
  return RUN_ALL_TESTS();
}