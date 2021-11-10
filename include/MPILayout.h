//
// Created by Ben_Sepanski on 11/8/2021.
//

#ifndef BRICK_MPILAYOUT_H
#define BRICK_MPILAYOUT_H

#include "BrickLayout.h"
#include "BrickedArray.h"
#include "array-mpi.h"
#include "brick-mpi.h"

#include <memory>

/**
 * @brief check for MPI failure
 *
 * @param return_value the value returned from an MPI call
 * @param func name of the MPI function being invoked
 * @param filename filename of call site
 * @param line line number of call site
 */
void _check_MPI(int return_value, const char *func, const char *filename,
                const int line) {
  if (return_value != MPI_SUCCESS) {
    char error_msg[MPI_MAX_ERROR_STRING + 1];
    int error_length;
    std::ostringstream error_stream;
    error_stream << "MPI Error during call " << func << " " << filename << ":"
                 << line << std::endl;
    if (MPI_Error_string(return_value, error_msg, &error_length) !=
        MPI_SUCCESS) {
      error_stream << "Invalid argument passed to MPI_Error_string"
                   << std::endl;
    } else {
      error_stream << error_msg << std::endl;
    }
    throw std::runtime_error(error_stream.str());
  }
}

#define check_MPI(x) _check_MPI(x, #x, __FILE__, __LINE__)

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
 * bricks and arrays.
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
   * @param cartesianComm a cartesian communicator
   * @param arrayExtent the extent of the array (excluding ghost elements)
   * @param ghostDepth the number of ghost elements
   * @param skinlist the skin-list to use for building the BrickDecomp
   * @return a pointer to the brickDecomp
   */
  template <typename ExtentDataType = unsigned, typename GZDataType = unsigned>
  std::shared_ptr<BrickDecomp<BrickDims, CommunicatingDims>>
  static buildBrickDecomp(MPI_Comm &cartesianComm,
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
    check_MPI(MPI_Comm_rank(cartesianComm, &rank));
    check_MPI(MPI_Comm_size(cartesianComm, &size));
    std::array<int, RANK> coordsOfProc{};
    check_MPI(MPI_Cart_coords(cartesianComm, rank, RANK, coordsOfProc.data()));
    // build brickdecomp
    std::shared_ptr<BrickDecompType> brickDecompPtr(new BrickDecompType(
        std::vector<unsigned>(arrayExtent.cbegin(), arrayExtent.cend()),
        std::vector<unsigned>(ghostDepth.cbegin(), ghostDepth.cend())));
    // populate neighbors and build from skinlist
    brickDecompPtr->comm = cartesianComm;
    populate(cartesianComm, *brickDecompPtr, 0, 1, coordsOfProc.data());
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
    typedef templateutils::ParameterPackManipulator<unsigned> UnsignedPackManip;
    typedef UnsignedPackManip::Pack<BDims...> BrickDimsPack;
    // return an array holding the grid
    std::array<unsigned, RANK> brickGridExtent{};
    for (unsigned d = 0; d < RANK; ++d) {
      unsigned brickDim = UnsignedPackManip::get<BrickDimsPack>(RANK - 1 - d);
      brickGridExtent[d] = (arrayExtent[d] + 2 * ghostDepth[d]) / brickDim;
    }
    Array<unsigned, RANK, Padding<>, unsigned, unsigned> indexInStorage(
        brickGridExtent);
    initializeGrid<RANK - 1>(indexInStorage.begin(), *bDecompPtr,
                             brickGridExtent);

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

  /// methods
public:
  // public methods
  /**
   * Construct a brick-layout by using a BrickDecomp
   *
   * @tparam ExtentDataType must be convertible to unsigned
   * @tparam GZDataType must be convertible to unsigned
   * @param cartesianComm a cartesian MPI communicator
   * @param arrayExtent the extent of the array (in elements)
   * @param ghostDepth the number of ghost elements on each axis
   * @param skinlist the order in which neighbor-data is stored
   */
  template <typename ExtentDataType = unsigned, typename GZDataType = unsigned>
  MPILayout(MPI_Comm &cartesianComm,
            const std::array<ExtentDataType, RANK> &arrayExtent,
            const std::array<GZDataType, RANK> &ghostDepth,
            const std::vector<BitSet> &skinlist)
      : brickDecompPtr{buildBrickDecomp(cartesianComm, arrayExtent, ghostDepth,
                                        skinlist)},
        brickLayout{buildBrickLayout(brickDecompPtr, arrayExtent, ghostDepth)} {
    static_assert(std::is_convertible<ExtentDataType, unsigned>::value,
                  "ExtentDataType not convertible to unsigned");
    static_assert(std::is_convertible<GZDataType, unsigned>::value,
                  "GZDataType not convertible to unsigned");

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
  ExchangeView
  buildExchangeView(const brick::BrickedArray<T...> &brickedArray) {
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
  void exchangeWithoutMMAP(brick::BrickedArray<T...> &brickedArray) {
    BrickStorage storage = brickedArray.getStorage();
    brickDecompPtr->exchange(storage);
  }

#ifdef __CUDACC__
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
    auto dst = bStorage.dat.get() + bStorage.step + brickDecompPtr->sep_pos[0];
    auto src =
        bStorage_dev.dat.get() + bStorage_dev.step + brickDecompPtr->sep_pos[0];
    size_t size = bStorage.step *
                  (brickDecompPtr->sep_pos[1] - brickDecompPtr->sep_pos[0]) *
                  sizeof(bElem);
    cudaCheck(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
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
    auto dst =
        bStorage_dev.dat.get() + bStorage_dev.step + brickDecompPtr->sep_pos[1];
    auto src = bStorage.dat.get() + bStorage.step + brickDecompPtr->sep_pos[1];
    size_t size = bStorage.step *
                  (brickDecompPtr->sep_pos[2] - brickDecompPtr->sep_pos[1]) *
                  sizeof(bElem);
    cudaCheck(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
  }
#endif

  /**
   * Exchange arrays
   * @tparam DataType data-type of the array
   * @tparam ArrPadding the padding of the array
   * @tparam ArrSizeType the size-type of the array
   * @tparam ArrIndexType the index-type of the array
   * @param arr the array to exchange
   */
  template <typename DataType, typename ArrPadding, typename ArrSizeType,
            typename ArrIndexType>
  void exchangeArray(
      Array<DataType, RANK, ArrPadding, ArrSizeType, ArrIndexType> arr) {
    std::vector<long> padding;
    padding.reserve(RANK);
    for (unsigned d = 0; d < RANK; ++d) {
      if (arr.extent[d] != extent[d]) {
        throw std::runtime_error("Mismatch in extents");
      }
      padding.push_back(arr.PADDING(d));
    }
    exchangeArr<RANK>(arr.getData().get(), brickDecompPtr->comm,
                      brickDecompPtr->rank_map, this->extent, padding,
                      this->ghost);
  }

  // TODO: Implement exchangeArrayWithTypes
};

} // end namespace brick

#endif // BRICK_MPILAYOUT_H
