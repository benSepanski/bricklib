//
// Created by Ben_Sepanski on 10/15/2021.
//

#include <array>
#include <gtest/gtest.h>

#include "array.h"
#include "bricked-array.h"

TEST(BrickedArrayTests, AssignmentTest) {
  constexpr unsigned RANK = 3;
  std::array<unsigned, RANK> brickGridExtent = {3, 3, 3};
  brick::BrickLayout<RANK> layout(brickGridExtent);
  brick::BrickedArray<bElem, Dim<2,2,2>> brickedArray(layout);
}
