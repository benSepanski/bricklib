//
// Created by Ben_Sepanski on 10/15/2021.
//

#include <array>
#include <gtest/gtest.h>

#include "array.h"
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
  for(unsigned k = 0; k < extent[2]; ++k)
  for(unsigned j = 0; j < extent[1]; ++j)
  for(unsigned i = 0; i < extent[0]; ++i) {
    arr(i, j, k) = i + extent[0] * (j + extent[1] * k);
  }

  // Test the copy from array into the brick
  brickedArray.loadFrom(arr);
  for(unsigned k = 0; k < extent[2]; ++k)
  for(unsigned j = 0; j < extent[1]; ++j)
  for(unsigned i = 0; i < extent[0]; ++i) {
    bElem value = brickedArray(i, j, k);
    EXPECT_EQ(value, i + extent[0] * (j + extent[1] * k));
  }

  // test store-to
  Array arrCopy(extent);
  brickedArray.storeTo(arrCopy);
  for(unsigned k = 0; k < extent[2]; ++k)
  for(unsigned j = 0; j < extent[1]; ++j)
  for(unsigned i = 0; i < extent[0]; ++i) {
    EXPECT_EQ(arr(i, j, k), arrCopy(i, j, k));
  }

  // Just to test compilation
  auto brick = brickedArray.getBricks();
}
