//
// Created by Ben_Sepanski on 11/9/2021.
//

#include "cuda-test-utils.h"
#include "MPILayout.h"
#include "MPIEnvironment.h"
#include "MPICartesianSetup_3D.h"
#include <gtest/gtest.h>
#include <cuda_runtime.h>

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

#if !defined(MPIX_CUDA_AWARE_SUPPORT) || MPIX_CUDA_AWARE_SUPPORT
#if !defined(MPIX_CUDA_AWARE_SUPPORT)
#warning "This MPI library cannot determine if there is CUDA-aware support."
#endif
TYPED_TEST(MPI_CartesianTest3D, CudaAwareArray) {
  typedef typename TestFixture::BrickDims BrickDims;
  typedef TypeParam CommunicatingDims;

  // runtime-check to make sure we have cuda-aware
#ifdef MPIX_CUDA_AWARE_SUPPORT
  ASSERT_EQ(MPIX_Query_cuda_support(), 1);
#endif

  // build an array
  brick::MPILayout<BrickDims, CommunicatingDims> mpiLayout(this->buildMPILayout());
  brick::BrickLayout<3> layout = mpiLayout.getBrickLayout();
  brick::Array<bElem, 3> arr({this->extentWithGZ[0], this->extentWithGZ[1], this->extentWithGZ[2]});
  this->template fill3DArray(arr);
  // copy it to the device
  auto arr_dev = arr.allocateOnDevice();
  arr.copyToDevice(arr_dev);

  // build a types handle and run the MPI transfer
  auto typesHandle = mpiLayout.buildArrayTypesHandle(arr);
  mpiLayout.exchangeArray(arr, typesHandle);
  mpiLayout.exchangeArray(arr_dev, typesHandle);
  // copy back and make sure it's what we expect
  brick::Array<bElem, 3> arr2({arr.extent[0], arr.extent[1], arr.extent[2]});
  arr2.copyFromDevice(arr_dev);

  for(unsigned k = 0; k < this->extentWithGZ[2]; ++k) {
    for(unsigned j = 0; j < this->extentWithGZ[1]; ++j) {
      for(unsigned i = 0; i < this->extentWithGZ[0]; ++i) {
        EXPECT_EQ(arr.get(i, j, k), arr2.get(i, j, k));
      }
    }
  }
}
#endif

int main(int argc, char *argv[]) {
  gpuCheck(cudaSetDevice(0));
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(
      new MPIEnvironment(argc, argv));
  return RUN_ALL_TESTS();
}