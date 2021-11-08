#pragma clang diagnostic push
#pragma ide diagnostic ignored "cert-err58-cpp"
//
// Created by Ben_Sepanski on 10/15/2021.
//

#include <array>
#include <gtest/gtest.h>

#include "Array.h"

/**
 * Test suite for consistency with multiple dimensions/paddings
 */
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
  #pragma unroll
  for(unsigned i = 0; i < RANK; ++i) {
    extentWithPadding[i] = extent[i] + 2 * Array3D::PADDING(i);
  }
  unsigned numElements = std::accumulate(extentWithPadding.begin(),
                                         extentWithPadding.end(),
                                         1,
                                         std::multiplies<unsigned>());
  std::shared_ptr<int> data((int*)malloc(numElements * sizeof(int)), free);
  unsigned i = 0, j = 0, k = 0;
  int paddingIndex = -1;
  for(unsigned index = 0; index < numElements; ++index) {
    if(i < Array3D::PADDING(0) || j < Array3D::PADDING(1) || k < Array3D::PADDING(2)
        || i >= extent[0] + Array3D::PADDING(0)
        || j >= extent[1] + Array3D::PADDING(1)
        || k >= extent[2] + Array3D::PADDING(2)) {
      data.get()[index] = paddingIndex--;
    } else {
      data.get()[index] = (i - Array3D::PADDING(0))
                    + 10 * (j - Array3D::PADDING(1))
                    + 100 * (k - Array3D::PADDING(2));
    }
    i += 1;
    j += i / extentWithPadding[0];
    i %= extentWithPadding[0];
    k += j / extentWithPadding[1];
    j %= extentWithPadding[1];
    ASSERT_TRUE(k < extentWithPadding[2] || index >= numElements - 1);
  }
  Array3D arr(extent, data);
  for(k = 0; k < extent[2]; ++k) {
    for(j = 0; j < extent[1]; ++j) {
      for(i = 0; i < extent[0]; ++i) {
        EXPECT_EQ(arr(i, j, k), i + 10 * j + 100 * k);
      }
    }
  }
}

template<typename Padding>
void testConsistency2D(std::array<unsigned, 2> extent) {
  constexpr unsigned RANK = 2;
  // Build up array
  using Array2D = brick::Array<int, RANK, Padding>;
  std::array<unsigned, RANK> extentWithPadding{};
  for(unsigned i = 0; i < RANK; ++i) {
    extentWithPadding[i] = extent[i] + 2 * Array2D::PADDING(i);
  }
  unsigned numElements = std::accumulate(extentWithPadding.begin(),
                                         extentWithPadding.end(),
                                         1,
                                         std::multiplies<unsigned>());
  std::shared_ptr<int> data((int*)malloc(numElements * sizeof(int)), free);
  unsigned i = 0, j = 0;
  int paddingIndex = -1;
  for(unsigned index = 0; index < numElements; ++index) {
    if(i < Array2D::PADDING(0) || j < Array2D::PADDING(1)
        || i >= extent[0] + Array2D::PADDING(0)
        || j >= extent[1] + Array2D::PADDING(1)) {
      data.get()[index] = paddingIndex--;
    } else {
      data.get()[index] = (i - Array2D::PADDING(0))
                    + 10 * (j - Array2D::PADDING(1));
    }
    i += 1;
    j += i / extentWithPadding[0];
    i %= extentWithPadding[0];
    ASSERT_TRUE(j < extentWithPadding[1] || index >= numElements - 1);
  }
  Array2D arr(extent, data);
  for(j = 0; j < extent[1]; ++j) {
    for(i = 0; i < extent[0]; ++i) {
      EXPECT_EQ(arr(i, j), i + 10 * j);
    }
  }
}

template<typename Padding>
void testConsistency1D(unsigned extent) {
  // Build up array
  using Array1D = brick::Array<int, 1, Padding>;
  unsigned extentWithPadding = extent + 2 * Array1D::PADDING(0);
  std::shared_ptr<int> data((int*)malloc(extentWithPadding * sizeof(int)), free);
  int paddingIndex = -1;
  for(unsigned index = 0; index < extentWithPadding; ++index) {
    if(index < Array1D::PADDING(0) || index >= extent + Array1D::PADDING(0)) {
      data.get()[index] = paddingIndex--;
    } else {
      data.get()[index] = (index - Array1D::PADDING(0));
    }
  }
  Array1D arr({extent}, data);
  for(int i = 0; i < extent; ++i) {
    EXPECT_EQ(arr(i), i);
  }
}

/**
 * Check data ordering
 */
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

/**
 * Check iterators
 */
TYPED_TEST(BasicArrayConsistencyTests, IteratorTest) {
  typedef brick::Padding<TypeParam::PADDING[2], TypeParam::PADDING[1], TypeParam::PADDING[0]> Padding;
  constexpr unsigned RANK = 3;
  // Build up array
  using Array3D = brick::Array<int, RANK, Padding>;
  std::array<unsigned, RANK> extent = {3, 3, 3};
  unsigned numElements = std::accumulate(extent.begin(),
                                         extent.end(),
                                         1,
                                         std::multiplies<unsigned>());

  // Check forwards
  Array3D arr(extent);
  int index = 0;
  typename Array3D::iterator_type it = arr.begin();
  while(it != arr.end()) {
    assert(index < numElements);
    *it = index++;
    it++;
  }

  for(unsigned k = 0; k < extent[2]; ++k) {
    for(unsigned j = 0; j < extent[1]; ++j) {
      for(unsigned i = 0; i < extent[0]; ++i) {
        EXPECT_EQ(arr(i, j, k), i + extent[0] * (j + extent[1] * k));
      }
    }
  }

  // Now check reverse
  index = 0;
  it = arr.end();
  while(it != arr.begin()) {
    assert(index < numElements);
    *(--it) = index++;
  }

  for(unsigned k = 0; k < extent[2]; ++k) {
    for(unsigned j = 0; j < extent[1]; ++j) {
      for(unsigned i = 0; i < extent[0]; ++i) {
        EXPECT_EQ(arr(i, j, k), (extent[0] - 1 - i)
                                + extent[0] * ((extent[1] - 1 - j)
                                + extent[1] *  (extent[2] - 1 - k)));
      }
    }
  }
}

/**
 * Check array load/stores
 */
TYPED_TEST(BasicArrayConsistencyTests, loadStoreTest) {
  typedef brick::Padding<TypeParam::PADDING[2], TypeParam::PADDING[1], TypeParam::PADDING[0]> Padding;
  constexpr unsigned RANK = 3;
  // Build up array
  using Array3D = brick::Array<int, RANK, Padding>;
  std::array<unsigned, RANK> extent = {3, 3, 3};
  Array3D arr(extent), arrCopy(extent);
  for(unsigned k = 0; k < extent[2]; ++k) {
    for (unsigned j = 0; j < extent[1]; ++j) {
      for (unsigned i = 0; i < extent[0]; ++i) {
        arr(i, j, k) = i + extent[0] * (j + extent[1] * k);
        arrCopy(i, j, k) = 0;
      }
    }
  }

  // Test copy
  arrCopy.loadFrom(arr);

  for(unsigned k = 0; k < extent[2]; ++k) {
    for (unsigned j = 0; j < extent[1]; ++j) {
      for (unsigned i = 0; i < extent[0]; ++i) {
        EXPECT_EQ(arrCopy(i, j, k), i + extent[0] * (j + extent[1] * k));
        arr(i, j, k) = 0;
      }
    }
  }

  // Test store
  arrCopy.storeTo(arr);

  for(unsigned k = 0; k < extent[2]; ++k) {
    for (unsigned j = 0; j < extent[1]; ++j) {
      for (unsigned i = 0; i < extent[0]; ++i) {
        EXPECT_EQ(arr(i, j, k), i + extent[0] * (j + extent[1] * k));
      }
    }
  }
}
