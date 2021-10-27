//
// Created by Ben_Sepanski on 10/15/2021.
//

#include <array>
#include <gtest/gtest.h>

#include "array.h"

template<typename T>
class BasicArrayConsistencyTests : public testing::Test {
};

template<unsigned PADDING_k, unsigned PADDING_j, unsigned PADDING_i>
struct Padding {
  static constexpr std::array<unsigned, 3> PADDING = {PADDING_i, PADDING_j, PADDING_k};
};

using Paddings = ::testing::Types<Padding<0, 0, 0>, Padding<0, 0, 1>,
                                  Padding<0, 1, 0>, Padding<0, 1, 1>,
                                  Padding<1, 0, 0>, Padding<1, 0, 1>,
                                  Padding<1, 1, 0>, Padding<1, 1, 1>>;
TYPED_TEST_SUITE(BasicArrayConsistencyTests, Paddings);

template<typename Padding>
void testConsistency3D(std::array<unsigned, 3> extent) {
  constexpr unsigned RANK = 3;
  // Build up array
  using Array3D = brick::Array<int, RANK, Padding>;
  std::array<unsigned, RANK> extentWithPadding{};
  for(unsigned i = 0; i < RANK; ++i) {
    extentWithPadding[i] = extent[i] + 2 * Array3D::PADDING[i];
  }
  unsigned numElements = std::accumulate(extentWithPadding.begin(),
                                         extentWithPadding.end(),
                                         1,
                                         std::multiplies<unsigned>());
  int *data = new int[numElements];
  unsigned i = 0, j = 0, k = 0;
  int paddingIndex = -1;
  for(unsigned index = 0; index < numElements; ++index) {
    if(i < Array3D::PADDING[0] || j < Array3D::PADDING[1] || k < Array3D::PADDING[2]
        || i >= extent[0] + Array3D::PADDING[0]
        || j >= extent[1] + Array3D::PADDING[1]
        || k >= extent[2] + Array3D::PADDING[2]) {
      data[index] = paddingIndex--;
    } else {
      data[index] = (i - Array3D::PADDING[0])
                    + 10 * (j - Array3D::PADDING[1])
                    + 100 * (k - Array3D::PADDING[2]);
    }
    i += 1;
    j += i / extentWithPadding[0];
    i %= extentWithPadding[0];
    k += j / extentWithPadding[1];
    j %= extentWithPadding[1];
    ASSERT_TRUE(k < extentWithPadding[2] || index >= numElements - 1);
  }
  Array3D arr(extent, data);
  for(int k = 0; k < extent[2]; ++k) {
    for(int j = 0; j < extent[1]; ++j) {
      for(int i = 0; i < extent[0]; ++i) {
        EXPECT_EQ(arr(i, j, k), i + 10 * j + 100 * k);
      }
    }
  }
  delete []data;
}

template<typename Padding>
void testConsistency2D(std::array<unsigned, 2> extent) {
  constexpr unsigned RANK = 2;
  // Build up array
  using Array2D = brick::Array<int, RANK, Padding>;
  std::array<unsigned, RANK> extentWithPadding{};
  for(unsigned i = 0; i < RANK; ++i) {
    extentWithPadding[i] = extent[i] + 2 * Array2D::PADDING[i];
  }
  unsigned numElements = std::accumulate(extentWithPadding.begin(),
                                         extentWithPadding.end(),
                                         1,
                                         std::multiplies<unsigned>());
  int *data = new int[numElements];
  unsigned i = 0, j = 0;
  int paddingIndex = -1;
  for(unsigned index = 0; index < numElements; ++index) {
    if(i < Array2D::PADDING[0] || j < Array2D::PADDING[1]
        || i >= extent[0] + Array2D::PADDING[0]
        || j >= extent[1] + Array2D::PADDING[1]) {
      data[index] = paddingIndex--;
    } else {
      data[index] = (i - Array2D::PADDING[0])
                    + 10 * (j - Array2D::PADDING[1]);
    }
    i += 1;
    j += i / extentWithPadding[0];
    i %= extentWithPadding[0];
    ASSERT_TRUE(j < extentWithPadding[1] || index >= numElements - 1);
  }
  Array2D arr(extent, data);
  for(int j = 0; j < extent[1]; ++j) {
    for(int i = 0; i < extent[0]; ++i) {
      EXPECT_EQ(arr(i, j), i + 10 * j);
    }
  }
  delete []data;
}

template<typename Padding>
void testConsistency1D(unsigned extent) {
  constexpr unsigned RANK = 1;
  // Build up array
  using Array1D = brick::Array<int, RANK, Padding>;
  unsigned extentWithPadding = extent + 2 * Array1D::PADDING[0];
  int *data = new int[extentWithPadding];
  int paddingIndex = -1;
  for(unsigned index = 0; index < extentWithPadding; ++index) {
    if(index < Array1D::PADDING[0] || index >= extent + Array1D::PADDING[0]) {
      data[index] = paddingIndex--;
    } else {
      data[index] = (index - Array1D::PADDING[0]);
    }
  }
  Array1D arr({extent}, data);
  for(int i = 0; i < extent; ++i) {
    EXPECT_EQ(arr(i), i);
  }
  delete []data;
}

TYPED_TEST(BasicArrayConsistencyTests, CheckDataLayout) {
  typedef brick::Padding<TypeParam::PADDING[2], TypeParam::PADDING[1], TypeParam::PADDING[0]> PADDING3;
  typedef brick::Padding<TypeParam::PADDING[1], TypeParam::PADDING[0]> PADDING2;
  typedef brick::Padding<TypeParam::PADDING[0]> PADDING1;
  for(unsigned k = 1; k < 3; ++k) {
    for(unsigned j = 1; j < 3; ++j) {
      for(unsigned i = 1; i < 3; ++i) {
        testConsistency3D<PADDING3>({i, j, k});
      }
      testConsistency2D<PADDING2>({j, k});
    }
    testConsistency1D<PADDING1>(k);
  }
}