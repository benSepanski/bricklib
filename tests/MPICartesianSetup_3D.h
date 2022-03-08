//
// Created by Ben_Sepanski on 11/11/2021.
//

#ifndef BRICK_MPICARTESIANSETUP_3D_H
#define BRICK_MPICARTESIANSETUP_3D_H

/**
 * Used to set up a cartesian communicator/MPILayout
 *
 * @tparam CommunicatingDims the dimensions that bricks should communicate in
 */
template<typename CommunicatingDims>
class MPI_CartesianTest3D : public ::testing::Test {
protected:
  MPI_Comm cartesianComm;
  // setup MPI layout
  typedef Dim<4, 2, 1> BrickDims;
  std::array<unsigned, 3> extent = {4, 8, 16};
  std::array<unsigned, 3> ghostDepth =
      { CommunicatingDims::communicatesInDim(0) ? 1 : 0,
       CommunicatingDims::communicatesInDim(1) ? 2 : 0,
       CommunicatingDims::communicatesInDim(2) ? 4 : 0 };
  std::array<unsigned, 3> extentWithGZ = {
      extent[0] + 2 * ghostDepth[0],
      extent[1] + 2 * ghostDepth[1],
      extent[2] + 2 * ghostDepth[2]
  };

  void SetUp() override {
    // get number of MPI processes and my rank
    int size, rank;
    check_MPI(MPI_Comm_size(MPI_COMM_WORLD, &size));
    check_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

    // have MPI decide how many processes to put per dim
    std::array<int, 3> numProcsPerDim{};
    check_MPI(MPI_Dims_create(size, 3, numProcsPerDim.data()));

    // set up processes on a cartesian communication grid
    std::array<int, 3> periodic{};
    for (int i = 0; i < 3; ++i) {
      periodic[i] = true;
    }
    bool allowRankReordering = true;
    check_MPI(MPI_Cart_create(MPI_COMM_WORLD, 3, numProcsPerDim.data(),
                              periodic.data(), allowRankReordering,
                              &cartesianComm));
    if (cartesianComm == MPI_COMM_NULL) {
      throw std::runtime_error("Failure in cartesian comm setup");
    }
  }

  MPI_Comm getCartesianComm() { return cartesianComm; }

  brick::MPIHandle<3, CommunicatingDims> buildMPIHandle() {
    MPI_Comm comm = getCartesianComm();
    return brick::MPIHandle<3, CommunicatingDims>(comm);
  }

  brick::MPILayout<BrickDims, CommunicatingDims> buildMPILayout(brick::MPIHandle<3, CommunicatingDims> &mpiHandle) {
    std::vector<BitSet> skin = skin3d_good;
    for(long d = 0; d < 3; ++d) {
      if(!CommunicatingDims::communicatesInDim(d)) {
        auto containsD = [d](const BitSet &b) -> bool {
          return b.get(d+1) || b.get(-d-1);
        };
        skin.erase(
            std::remove_if(skin.begin(), skin.end(), containsD),
            skin.end()
        );
      }
    }
    return brick::MPILayout<BrickDims, CommunicatingDims>(mpiHandle, extent, ghostDepth, skin);
  }


  struct Region {
    bool isGhost;
    int regionID;
  };

  /**
 * Base case for getRegion
 * @see getRegion
   */
  template <size_t CurrentDim = 0>
  typename std::enable_if<CurrentDim == 3, Region>::type
  getRegion() const {
    return Region{false, 0};
  }

  /**
 *
 * @tparam CurrentDim current dimension
 * @tparam IndexType index type (convertible to unsigned)
 * @param firstIndex current index
 * @param remainingIndices remaining indices
 * @return A unique identifier for each region, or the negative
 *         identifier of the region represented by a ghost zone
   */
  template <size_t CurrentDim = 0, typename... IndexType>
      typename std::enable_if<CurrentDim <3 , Region>::type
                                getRegion(unsigned firstIndex, IndexType... remainingIndices) const {
    static_assert(sizeof...(IndexType) + 1 == 3 - CurrentDim,
                  "Mismatch in number of indices");
    Region r = getRegion<CurrentDim + 1>(remainingIndices...);
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

  /**
   * Fill each ghost element of arr with -1 and each non-ghost
   * element with the value from getIndex(i, j, k)
   *
   * @tparam T 3d array type
   * @param arr the array
   */
  template<typename T>
  void fill3DArray(T &arr) {
    for (unsigned k = 0; k < this->extentWithGZ[2]; ++k) {
      for (unsigned j = 0; j < this->extentWithGZ[1]; ++j) {
        for (unsigned i = 0; i < this->extentWithGZ[0]; ++i) {
          auto r = this->template getRegion(i, j, k);
          if(r.isGhost) {
            arr(i, j, k) = -1;
          } else {
            arr(i, j, k) = this->getIndex(i, j, k);
          }
        }
      }
    }
  }

  /**
   *
   * @param i index along axis 0
   * @param j index along axis 1
   * @param k index along axis 2
   * @return A unique index into the array if non-ghost, or the expected
   *         ghost element otherwise
   */
  unsigned getIndex(unsigned i, unsigned j, unsigned k) {
    unsigned flatIndex = 0;
    std::array<unsigned, 3> actualIdx = {i, j, k};
    for(unsigned d = 0; d < actualIdx.size(); ++d) {
      if(actualIdx[d] < ghostDepth[d]) {
        actualIdx[d] += extent[d] - ghostDepth[d];
      } else if(actualIdx[d] < ghostDepth[d] + extent[d]) {
        actualIdx[d] -= ghostDepth[d];
      } else {
        actualIdx[d] -= extent[d] + ghostDepth[d];
      }
      if(d > 0) {
        flatIndex *= extent[d - 1];
      }
      flatIndex += actualIdx[d];
    }
    return flatIndex;
  }
};

#endif // BRICK_MPICARTESIANSETUP_3D_H
