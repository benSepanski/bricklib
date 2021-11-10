//
// Created by Ben_Sepanski on 11/9/2021.
//

#include "MPILayout.h"
#include <gtest/gtest.h>

class MPIEnvironment : public ::testing::Environment {
public:
  void SetUp() override {
    int provided;
    int argc = 0;
    char **argv;
    // setup MPI environment
    check_MPI(MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided));
    if (provided != MPI_THREAD_SERIALIZED) {
      check_MPI(MPI_Finalize());
      ASSERT_EQ(provided, MPI_THREAD_SERIALIZED);
    }
  }

  ~MPIEnvironment() override {}

  void TearDown() override {}
};

template <unsigned NumDims> class MPI_CartesianTest : public ::testing::Test {
protected:
  MPI_Comm cartesianComm;

  void SetUp() override {
    // get number of MPI processes and my rank
    int size, rank;
    check_MPI(MPI_Comm_size(MPI_COMM_WORLD, &size));
    check_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

    // have MPI decide how many processes to put per dim
    std::array<int, NumDims> numProcsPerDim{};
    check_MPI(MPI_Dims_create(size, NumDims, numProcsPerDim.data()));

    // set up processes on a cartesian communication grid
    std::array<int, NumDims> periodic{};
    for (int i = 0; i < NumDims; ++i) {
      periodic[i] = true;
    }
    bool allowRankReordering = true;
    check_MPI(MPI_Cart_create(MPI_COMM_WORLD, NumDims, numProcsPerDim.data(),
                              periodic.data(), allowRankReordering,
                              &cartesianComm));
    if (cartesianComm == MPI_COMM_NULL) {
      throw std::runtime_error("Failure in cartesian comm setup");
    }
  }

  MPI_Comm getCartesianComm() { return cartesianComm; }
};

struct Region {
  bool isGhost;
  int regionID;
};

/**
 * Base case for getRegion
 * @see getRegion
 */
template <size_t CurrentDim = 0, size_t NumDims, typename... IndexType>
typename std::enable_if<CurrentDim == NumDims, Region>::type
getRegion(std::array<unsigned, NumDims> extent,
          std::array<unsigned, NumDims> ghostDepth) {
  return Region{false, 0};
}

/**
 *
 * @tparam CurrentDim current dimension
 * @tparam NumDims total number of dimensions
 * @tparam IndexType index type (convertible to unsigned)
 * @param extent extent (excluding ghost values)
 * @param ghostDepth depth of ghost zone
 * @param firstIndex current index
 * @param remainingIndices remaining indices
 * @return A unique identifier for each region, or the negative
 *         identifier of the region represented by a ghost zone
 */
template <size_t CurrentDim = 0, size_t NumDims, typename... IndexType>
    typename std::enable_if <
    CurrentDim<NumDims, Region>::type
    getRegion(std::array<unsigned, NumDims> extent,
              std::array<unsigned, NumDims> ghostDepth, unsigned firstIndex,
              IndexType... remainingIndices) {
  static_assert(sizeof...(IndexType) + 1 == NumDims - CurrentDim,
                "Mismatch in number of indices");
  Region r = getRegion<CurrentDim + 1>(extent, ghostDepth, remainingIndices...);
  r.regionID *= 3;
  if (firstIndex < ghostDepth[CurrentDim]) {
    r.isGhost = true;
    r.regionID += 2;
  } else if (firstIndex < 2 * ghostDepth[CurrentDim]) {
    r.regionID += 0;
  } else if (firstIndex < extent[CurrentDim]) {
    r.regionID += 1;
  } else if (firstIndex < extent[CurrentDim] + ghostDepth[CurrentDim]) {
    r.regionID += 2;
  } else {
    r.isGhost = true;
    r.regionID += 0;
  }
  return r;
}

typedef MPI_CartesianTest<3> MPI_CartesianTest3D;
TEST_F(MPI_CartesianTest3D, BasicLayoutTest) {
  // setup MPI layout
  typedef Dim<4, 2, 1> BrickDims;
  std::array<unsigned, 3> extent = {4, 8, 16};
  std::array<unsigned, 3> ghostDepth = {1, 2, 4};
  MPI_Comm comm = getCartesianComm();
  brick::MPILayout<BrickDims> mpiLayout(comm, extent, ghostDepth, skin3d_good);

  // build a brick from the layout
  brick::BrickedArray<bElem, BrickDims> bArr(mpiLayout.getBrickLayout());

  for (unsigned k = 0; k < extent[2] + 2 * ghostDepth[2]; ++k) {
    for (unsigned j = 0; j < extent[1] + 2 * ghostDepth[1]; ++j) {
      for (unsigned i = 0; i < extent[0] + 2 * ghostDepth[0]; ++i) {
        Region r = getRegion(extent, ghostDepth, i, j, k);
        bArr(i, j, k) = r.isGhost ? -r.regionID : r.regionID;
      }
    }
  }
  // exchange
  mpiLayout.exchangeWithoutMMAP(bArr);
  // Now make sure that ghosts received the appropriate regionTag
  // (NOTE: THIS RELIES ON THE CARTESIAN COMM BEING PERIODIC)
  for (unsigned k = 0; k < extent[2] + 2 * ghostDepth[2]; ++k) {
    for (unsigned j = 0; j < extent[1] + 2 * ghostDepth[1]; ++j) {
      for (unsigned i = 0; i < extent[0] + 2 * ghostDepth[0]; ++i) {
        Region r = getRegion(extent, ghostDepth, i, j, k);
        EXPECT_EQ(bArr(i, j, k), r.regionID);
      }
    }
  }
}

TEST_F(MPI_CartesianTest3D, BasicLayoutMmapTest) {
  // setup MPI layout
  typedef Dim<4, 2, 1> BrickDims;
  std::array<unsigned, 3> extent = {4, 8, 16};
  std::array<unsigned, 3> ghostDepth = {1, 2, 4};
  MPI_Comm comm = getCartesianComm();
  brick::MPILayout<BrickDims> mpiLayout(comm, extent, ghostDepth, skin3d_good);

  // build a brick from the layout using mmap
  brick::BrickedArray<bElem, BrickDims> bArr(mpiLayout.getBrickLayout(), nullptr);

  for (unsigned k = 0; k < extent[2] + 2 * ghostDepth[2]; ++k) {
    for (unsigned j = 0; j < extent[1] + 2 * ghostDepth[1]; ++j) {
      for (unsigned i = 0; i < extent[0] + 2 * ghostDepth[0]; ++i) {
        Region r = getRegion(extent, ghostDepth, i, j, k);
        bArr(i, j, k) = r.isGhost ? -r.regionID : r.regionID;
      }
    }
  }
  // exchange
  ExchangeView ev = mpiLayout.buildExchangeView(bArr);
  ev.exchange();
  // Now make sure that ghosts received the appropriate regionTag
  // (NOTE: THIS RELIES ON THE CARTESIAN COMM BEING PERIODIC)
  for (unsigned k = 0; k < extent[2] + 2 * ghostDepth[2]; ++k) {
    for (unsigned j = 0; j < extent[1] + 2 * ghostDepth[1]; ++j) {
      for (unsigned i = 0; i < extent[0] + 2 * ghostDepth[0]; ++i) {
        Region r = getRegion(extent, ghostDepth, i, j, k);
        EXPECT_EQ(bArr(i, j, k), r.regionID);
      }
    }
  }
}

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
  return RUN_ALL_TESTS();
}