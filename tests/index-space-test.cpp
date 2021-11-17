//
// Created by Benjamin Sepanski on 11/17/21.
//

#include <gtest/gtest.h>
#include <gtest/gtest-printers.h>

#include "IndexSpace.h"

class IndexSpaceTests : public ::testing::TestWithParam<std::array<unsigned, 3> > {
protected:
  std::array<unsigned, 3> extent{};
  void SetUp() override {
    for(unsigned d = 0; d < 3; ++d) {
      extent[d] = GetParam()[d];
    }
  }
};

std::array<unsigned, 3> extents[] = {
    {1, 2, 3}, {2, 1, 3}, {2, 3, 1}, {3, 3, 3}
};
INSTANTIATE_TEST_SUITE_P(AllExtents, IndexSpaceTests, testing::ValuesIn(extents));

TEST_P(IndexSpaceTests, TestIncrementDecrement) {
  brick::IndexSpace<3> ispace(extent);

  auto ispaceForwardIt = ispace.begin();
  auto ispaceBackwardIt = ispace.end();
  for(unsigned k = 0; k < extent[2]; ++k) {
    for(unsigned j = 0; j < extent[1]; ++j) {
      for(unsigned i = 0; i < extent[0]; ++i) {
        EXPECT_EQ(ispaceForwardIt->k(), k);
        EXPECT_EQ(ispaceForwardIt->j(), j);
        EXPECT_EQ(ispaceForwardIt->i(), i);
        ispaceForwardIt++;
        ispaceBackwardIt--;
        EXPECT_EQ(ispaceBackwardIt->k(), extent[2] - 1 - k);
        EXPECT_EQ(ispaceBackwardIt->j(), extent[1] - 1 - j);
        EXPECT_EQ(ispaceBackwardIt->i(), extent[0] - 1 - i);
      }
    }
  }
  EXPECT_EQ(ispaceForwardIt, ispace.end());
  EXPECT_EQ(ispaceBackwardIt, ispace.begin());
}

TEST_P(IndexSpaceTests, TestPlusEquals) {
  brick::IndexSpace<3> ispace(extent);

  unsigned index = 0;
  for(unsigned k = 0; k < extent[2]; ++k) {
    for(unsigned j = 0; j < extent[1]; ++j) {
      for(unsigned i = 0; i < extent[0]; ++i) {
        auto itFromFront = ispace.begin() + index;
        index++;
        auto itFromBack = ispace.end() - index;
        EXPECT_EQ(itFromFront->k(), k);
        EXPECT_EQ(itFromFront->j(), j);
        EXPECT_EQ(itFromFront->i(), i);
        EXPECT_EQ(itFromBack->k(), extent[2] - 1 - k);
        EXPECT_EQ(itFromBack->j(), extent[1] - 1 - j);
        EXPECT_EQ(itFromBack->i(), extent[0] - 1 - i);
      }
    }
  }
  EXPECT_EQ(ispace.begin() + index, ispace.end());
  EXPECT_EQ(ispace.end() - index, ispace.begin());
}

TEST_P(IndexSpaceTests, TestNamedAxis) {
  brick::IndexSpace<3> ispace(extent);
  bool threwError = false;
  try {
    ispace.begin()->l();
  } catch(std::runtime_error &e) {
    threwError = true;
  }
  EXPECT_TRUE(threwError);
}