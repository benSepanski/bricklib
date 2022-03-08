//
// Created by Ben_Sepanski on 11/8/2021.
//

#ifndef BRICK_MPILAYOUT_H
#define BRICK_MPILAYOUT_H

#include "BrickLayout.h"
#include "BrickedArray.h"
#include "MPIHandle.h"
#include "array-mpi.h"
#include "brick-mpi.h"

#include <memory>

namespace brick {
/**
 * Generic template for partial specialization.
 * @tparam BrickDims the dimensions of the brick
 * @tparam CommunicatingDims the dimensions communication occurs in
 */
template <typename BrickDims, typename CommunicatingDims = CommDims<>>
class MPILayout;


/**
 * An MPI-layout to handle setup of fast MPI transfers for
 * bricks.
 *
 * @tparam BDims the brick-dimensions
 * @tparam CommInDim true if communicating in the dimension (default is true)
 */
template <unsigned... BDims, bool... CommInDim>
class MPILayout<Dim<BDims...>, CommDims<CommInDim...>> {
  /// constexprs/typedefs
private:
  // private typedefs
  using GridArray =
      Array<unsigned, sizeof...(BDims), Padding<>, unsigned, unsigned>;

public:
  // public constexprs/typedefs
  static constexpr unsigned RANK = sizeof...(BDims);
  typedef Dim<BDims...> BrickDims;
  typedef CommDims<CommInDim...> CommunicatingDims;
  typedef BrickDecomp<BrickDims, CommunicatingDims> BrickDecompType;

  /// static methods
private:
  // private static methods

  /**
   * Builds a brickDecomp
   * @tparam ExtentDataType convertible to unsigned
   * @tparam GZDataType convertible to unsigned
   * @param mpiHandle a handle to the mpi communicator
   * @param arrayExtent the extent of the array (excluding ghost elements)
   * @param ghostDepth the number of ghost elements
   * @param skinlist the skin-list to use for building the BrickDecomp
   * @return a pointer to the brickDecomp
   */
  template <typename ExtentDataType = unsigned, typename GZDataType = unsigned>
  std::shared_ptr<BrickDecomp<BrickDims, CommunicatingDims>>
  static buildBrickDecomp(brick::MPIHandle<RANK, CommunicatingDims> &mpiHandle,
                          const std::array<ExtentDataType, RANK> &arrayExtent,
                          const std::array<GZDataType, RANK> &ghostDepth,
                          const std::vector<BitSet> &skinlist) {
    // check skinlist
    for (const BitSet &b : skinlist) {
      for (long potentialElement = -32; potentialElement < 32;
           ++potentialElement) {
        if (b.get(potentialElement)) {
          long dim =
              potentialElement < 0 ? -potentialElement : potentialElement;
          if (!CommunicatingDims::communicatesInDim(dim - 1)) {
            throw std::runtime_error(
                "skinlist contains neighbor in non-communicating direction");
          }
        }
      }
    }
    // get rank, size
    int rank, size;
    check_MPI(MPI_Comm_rank(mpiHandle.getMPIComm(), &rank));
    check_MPI(MPI_Comm_size(mpiHandle.getMPIComm(), &size));
    std::array<int, RANK> coordsOfProc{};
    check_MPI(MPI_Cart_coords(mpiHandle.getMPIComm(), rank, RANK, coordsOfProc.data()));
    // build brickDecomp
    std::shared_ptr<BrickDecompType> brickDecompPtr(new BrickDecompType(
        std::vector<unsigned>(arrayExtent.cbegin(), arrayExtent.cend()),
        std::vector<unsigned>(ghostDepth.cbegin(), ghostDepth.cend())));
    // populate neighbors and build from skinlist
    brickDecompPtr->comm = mpiHandle.getMPIComm();
    populate(mpiHandle.getMPIComm(), *brickDecompPtr, 0, 1, coordsOfProc.data());
    brickDecompPtr->initialize(skinlist);

    // return the pointer
    return brickDecompPtr;
  }

  /**
   * Base case of initializeGrid
   * @see initializeGrid
   */
  template <unsigned Axis, typename T>
  static inline typename std::enable_if<Axis == 0,
                                        typename GridArray::iterator>::type
  initializeGrid(typename GridArray::iterator arr_it, T &gridAccess,
                 const std::array<unsigned, RANK> extents) {
    for (unsigned i = 0; i < extents[0]; ++i) {
      *(arr_it++) = gridAccess[i];
    }
    return arr_it;
  }

  /**
   * Copy the brick-grid from gridAccess into arr_it
   * @tparam T the grid-accessor type
   * @tparam Axis the axis to loop over
   * @param arr_it the array iterator
   * @param gridAccess the gridAccess object (from a BrickDecomp)
   * @param extents the extent of each axis
   */
  template <unsigned Axis, typename T>
  static inline typename std::enable_if<Axis != 0,
                                        typename GridArray::iterator>::type
  initializeGrid(typename GridArray::iterator arr_it, T &gridAccess,
                 const std::array<unsigned, RANK> extents) {
    static_assert(Axis < RANK, "Axis out of bounds");
    for (unsigned i = 0; i < extents[Axis]; ++i) {
      auto newGridAccess = gridAccess[i];
      arr_it = initializeGrid<Axis - 1>(arr_it, newGridAccess, extents);
    }
    return arr_it;
  }

  /**
   * Build a brick-layout from the provided BrickDecomp
   *
   * @tparam ExtentDataType convertible to unsigned
   * @tparam GhostDepthDataType convertible to unsigned
   * @param bDecompPtr the pointer to a brick decomposition
   * @return the brick layout
   */
  template <typename ExtentDataType = unsigned,
            typename GhostDepthDataType = unsigned>
  static brick::BrickLayout<RANK> buildBrickLayout(
      std::shared_ptr<BrickDecomp<BrickDims, CommunicatingDims>> bDecompPtr,
      const std::array<ExtentDataType, RANK> &arrayExtent,
      const std::array<GhostDepthDataType, RANK> &ghostDepth) {
    constexpr std::array<unsigned, RANK> brickDimsInReverse = { BDims... };
    // return an array holding the grid
    std::array<unsigned, RANK> brickGridExtent{};
    for (unsigned d = 0; d < RANK; ++d) {
      unsigned brickDim = brickDimsInReverse[sizeof...(BDims) - 1 - d];
      if(arrayExtent[d] % brickDim != 0) {
        throw std::runtime_error("arrayExtent not divisible by brick-dimension");
      }
      if(ghostDepth[d] % brickDim != 0) {
        throw std::runtime_error("ghost depth not divisible by brick-dimension");
      }
      brickGridExtent[d] = (arrayExtent[d] + 2 * ghostDepth[d]) / brickDim;
    }
    GridArray indexInStorage(brickGridExtent);
    typename GridArray::iterator b = indexInStorage.begin();
    initializeGrid<RANK - 1>(b, *bDecompPtr, brickGridExtent);

    std::shared_ptr<BrickInfo<RANK, CommunicatingDims> > brickInfoPtr(
        new BrickInfo<RANK, CommunicatingDims>(bDecompPtr->getBrickInfo()),
        // Do nothing on delete since bDecompPtr will free the brickInfo
        // when it is deleted
        [](BrickInfo<RANK, CommunicatingDims> *p) {});
    return brick::BrickLayout<RANK>(indexInStorage, brickInfoPtr);
  }

  /// members
private:
  // private members
  std::shared_ptr<BrickDecomp<BrickDims, CommunicatingDims>> brickDecompPtr;
  brick::BrickLayout<RANK> brickLayout;
  std::vector<long> ghost{}, extent{};

public:
  // public methods
  /**
   * Construct a brick-layout by using a BrickDecomp
   *
   * @tparam ExtentDataType must be convertible to unsigned
   * @tparam GZDataType must be convertible to unsigned
   * @param mpiHandle a handle to the MPI communicator
   * @param arrayExtent the extent of the array (in elements)
   * @param ghostDepth the number of ghostExtent elements on each axis
   * @param skinlist the order in which neighbor-data is stored
   */
  template <typename ExtentDataType = unsigned, typename GZDataType = unsigned>
  MPILayout(brick::MPIHandle<RANK, CommunicatingDims> &mpiHandle,
            const std::array<ExtentDataType, RANK> &arrayExtent,
            const std::array<GZDataType, RANK> &ghostDepth,
            const std::vector<BitSet> &skinlist)
      : brickDecompPtr{buildBrickDecomp(mpiHandle, arrayExtent, ghostDepth, skinlist)},
        brickLayout{buildBrickLayout(brickDecompPtr, arrayExtent, ghostDepth)} {
    static_assert(std::is_convertible<ExtentDataType, unsigned>::value,
                  "ExtentDataType not convertible to unsigned");
    static_assert(std::is_convertible<GZDataType, unsigned>::value,
                  "GZDataType not convertible to unsigned");
    static_assert(!templateutils::All<!CommInDim...>::value || sizeof...(CommInDim) < RANK,
                  "Communication must occur in at least one direction");

    // record number of ghost elements and extent
    for (unsigned d = 0; d < RANK; ++d) {
      this->extent.push_back(arrayExtent[d]);
      this->ghost.push_back(ghostDepth[d]);
    }
  }

  /**
   * @return the brick layout
   */
  BrickLayout<RANK> getBrickLayout() const { return brickLayout; }

  /**
   * @note We return a pointer here to avoid firing off the BrickDecomp
   *       destructor, which may free up some of its data.
   * @return a pointer to the brick decomposition
   */
  std::shared_ptr<BrickDecomp<BrickDims, CommunicatingDims>>
  getBrickDecompPtr() const {
    return brickDecompPtr;
  }

  /**
   * @tparam T type-parameters to BrickedArray
   * @param brickedArray the bricked array to get an exchange view for.
   *                     Must have been setup using mmap.
   * @return an exchange view for the host-data of the bricked array
   */
  template <typename... T>
  ExchangeView buildBrickedArrayMMAPExchangeView(const brick::BrickedArray<T...> &brickedArray) {
    if(brickedArray.getStorage().mmap_info == nullptr) {
      throw std::runtime_error("Storage has no mmap info");
    }
    return brickDecompPtr->exchangeView(brickedArray.getStorage());
  }

  /**
   * Exchange ghost-zones of the brickedArray without using any MMAP info
   *
   * @tparam T type parameters to BrickedArray
   * @param brickedArray the bricked array to exchange ghost-zones.
   *                     Must have been built on this object's BrickLayout
   */
  template<typename ... T>
  void exchangeBrickedArray(brick::BrickedArray<T...> &brickedArray) {
    BrickStorage storage = brickedArray.getStorage();
    brickDecompPtr->exchange(storage);
  }

#ifdef __CUDACC__
   /**
   * Exchange ghost-zones of the brickedArray on device
   * without using any MMAP info
   *
   * Requires cuda-aware
   *
   * @tparam T type parameters to BrickedArray
   * @param brickedArray the bricked array to exchange ghost-zones on the
   *                     device for.
   *                     Must have been built on this object's BrickLayout.
   */
  template<typename ... T>
  void exchangeCudaBrickedArray(brick::BrickedArray<T...> &brickedArray) {
    BrickStorage storage = brickedArray.getCudaStorage();
    brickDecompPtr->exchange(storage);
  }

  /**
   * Copy the boundary regions of the bricked array from the CUDA
   * device to host
   * (Note: this will copy for all interleaved fields)
   *
   * @tparam T type-parameters for the bricked array
   * @param brickedArray a BrickedArray built on this object's BrickLayout
   */
  template <typename... T>
  void copyBoundaryFromCuda(BrickedArray<T...> brickedArray) {
    BrickStorage bStorage = brickedArray.getStorage(),
                 bStorage_dev = brickedArray.getCudaStorage();
    auto bdyStart = brickDecompPtr->sep_pos[0];
    auto bdyEnd = brickDecompPtr->sep_pos[1];
    auto dst = bStorage.dat.get() + bStorage.step * bdyStart;
    auto src = bStorage_dev.dat.get() + bStorage_dev.step * bdyStart;
    size_t size = bStorage.step * (bdyEnd - bdyStart) * sizeof(bElem);
    gpuCheck(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
  }

  /**
   * Copy the ghost regions of the bricked array to the CUDA
   * device from host
   * (Note: this will copy for all interleaved fields)
   *
   * @tparam T type-parameters for the bricked array
   * @param brickedArray a BrickedArray built on this object's BrickLayout
   */
  template <typename... T>
  void copyGhostToCuda(BrickedArray<T...> brickedArray) {
    BrickStorage bStorage = brickedArray.getStorage(),
                 bStorage_dev = brickedArray.getCudaStorage();
    auto ghostStart = brickDecompPtr->sep_pos[1];
    auto ghostEnd = brickDecompPtr->sep_pos[2];
    auto dst = bStorage_dev.dat.get() + bStorage_dev.step * ghostStart;
    auto src = bStorage.dat.get() + bStorage.step * ghostStart;
    size_t size = bStorage.step * (ghostEnd - ghostStart) * sizeof(bElem);
    gpuCheck(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
  }
#endif
};

} // end namespace brick

#endif // BRICK_MPILAYOUT_H
