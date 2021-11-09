//
// Created by Ben_Sepanski on 11/6/2021.
//

#ifndef BRICK_BRICKLAYOUT_H
#define BRICK_BRICKLAYOUT_H

#include "Array.h"
#include "bricksetup.h"
#include <unordered_set>

namespace brick {
namespace { // Anonymous namespace
/**
     * Base class so that we can handle multiple BrickInfo types
     * with differing CommunicatingDims in a single object
     * @tparam Rank the rank of the BrickInfo
 */
template <unsigned Rank> struct BrickInfoWrapperBase {
  /**
     * @return a representation of the communicating dims
   */
  virtual std::vector<bool> getCommunicatingDims() const {
    return {};
  }

  /**
       * Consider two wrappers equal if they wrap the same type
       * @param b the other wrapper
       * @return true if the wrappers wrap the same type
   */
  bool operator==(const BrickInfoWrapperBase &b) const {
    std::vector<bool> otherCommDims = b.getCommunicatingDims(),
                      myCommDims = getCommunicatingDims();
    if(otherCommDims.size() != myCommDims.size()) {
      return false;
    }
    for(unsigned i = 0; i < otherCommDims.size(); ++i) {
      if(myCommDims[i] != otherCommDims[i]) {
        return false;
      }
    }
    return true;
  }

  /**
       * Virtual destructor for proper deletion
   */
  virtual ~BrickInfoWrapperBase() = default;
};
} // End anonymous namespace
}

namespace std { // std namespace
#pragma clang diagnostic push
#pragma ide diagnostic ignored "OCUnusedStructInspection" // Ignore some clang-tidy stuff
/**
   * Implement hashing for pointer to BrickInfoWrapperBase
   * @tparam Rank the rank of the BrickInfoWrapper
 */
template <unsigned Rank>
struct hash<std::shared_ptr<brick::BrickInfoWrapperBase<Rank> > > {
  size_t operator()(const std::shared_ptr<brick::BrickInfoWrapperBase<Rank> > &bPtr) const {
    size_t result = 0;
    size_t base = 1;
    for (bool comm : bPtr->getCommunicatingDims()) {
      result += (comm ? 2 : 1) * base;
      base *= 3;
    }
    return result;
  }
};
#pragma clang diagnostic pop
} // End std namespace

namespace brick {
namespace { // Anonymous namespace
/**
     * Generic template for partial specialization
     * @see BrickInfoWrapper
 */
template <unsigned Rank, typename CommunicatesInDim> struct BrickInfoWrapper;

/**
     * A wrapper around a BrickInfo
     * @tparam Rank the rank of the brick info
     * @tparam CommInDim the dimensions the brick info is communicating in
 */
template <unsigned Rank, bool... CommInDim>
struct BrickInfoWrapper<Rank, CommDims<CommInDim...>>
    : public BrickInfoWrapperBase<Rank> {
  typedef BrickInfo<Rank, CommDims<CommInDim...>> MyBrickInfoType;
  std::shared_ptr<MyBrickInfoType> brickInfo;

  /**
       * Default constructor
   */
  explicit BrickInfoWrapper() : brickInfo{nullptr} {}

  /**
       * Take a copy of bInfo
       * @param bInfo the BrickInfo to copy
   */
  explicit BrickInfoWrapper(std::shared_ptr<MyBrickInfoType> bInfo)
      : brickInfo{bInfo} {}

  /**
       * Destructor
   */
  ~BrickInfoWrapper() override = default;

  /**
       * @return a representation of the communicating dims
   */
  static std::vector<bool> communicatingDims() { return {CommInDim...}; }

  std::vector<bool> getCommunicatingDims() const override {
    return communicatingDims();
  }
};
} // End anonymous namespace

/**
 * A layout of bricks in memory
 *
 * @tparam Rank the rank of the layout
 */
template <unsigned Rank> struct BrickLayout {
  // constexprs and typedefs
public:
  static constexpr unsigned RANK = Rank;
  typedef Array<unsigned, RANK, Padding<>, unsigned, unsigned> ArrayType;

 /// static methods
private:
  // private static methods
  static ArrayType
  buildColumnMajorGrid(const std::array<unsigned, RANK> extent) {
    ArrayType arr(extent);

    unsigned index = 0;
    for (unsigned &element : arr) {
      element = index++;
    }
    return arr;
  }

  /// Members
private:
  // private members
  std::unordered_set<std::shared_ptr<BrickInfoWrapperBase<Rank>>>
      cachedBrickInfo;
#ifdef __CUDACC__
  std::unordered_set<std::shared_ptr<BrickInfoWrapperBase<Rank>>>
      cachedBrickInfoOnDev;
#endif
public:
  // public members
  // Brick-index to index in storage
  const ArrayType indexInStorage;
  const unsigned numBricks; //< number of bricks

  /// Methods
private:
  // private methods
  /**
         * @throw std::runtime_error if the extent of indexInStorage is not valid
         * @return The number of bricks
   */
  unsigned computeNumBricks() const {
    // check for possible overflow in number of elements
    unsigned nBricks = 1;
    for (unsigned d = 0; d < RANK; ++d) {
      unsigned extent = indexInStorage.extent[d];
      assert(extent != 0);
      size_t newNumBricks = extent * nBricks;
      if (newNumBricks / extent != nBricks) {
        throw std::runtime_error(
            "Brick grid does not fit inside of unsigned type");
      }
      nBricks = newNumBricks;
    }
    return nBricks;
  }

  template<typename CommunicatingDims>
  std::pair<typename decltype(cachedBrickInfo)::iterator, bool>
  insertIntoCache(BrickInfo<Rank, CommunicatingDims> brickInfo) {
    typedef BrickInfo<Rank, CommunicatingDims> BrickInfoType;
    typedef BrickInfoWrapper<Rank, CommunicatingDims> BrickInfoWrapperType;
    // Build a pointer to the BrickInfo
    std::shared_ptr<BrickInfoType> bInfoPtr(
        (BrickInfoType *)malloc(sizeof(BrickInfoType)), [](BrickInfoType *p) {
          free(p->adj);
          free(p);
        });
    *bInfoPtr.get() = brickInfo;
    // build and insert the wrapper
    std::shared_ptr<BrickInfoWrapperBase<Rank>> key(new BrickInfoWrapperType);
    *reinterpret_cast<BrickInfoWrapperType *>(key.get()) =
        BrickInfoWrapperType(bInfoPtr);
    auto insertHandle = cachedBrickInfo.insert(key);
    return insertHandle;
  }

public:
  // public methods
  /**
   * Build a layout from the provided array
   * @param indexInStorage Each entry holds the index of the brick
   *                       at the given logical index
   */
  explicit BrickLayout(const ArrayType indexInStorage)
      : indexInStorage{indexInStorage}, numBricks{computeNumBricks()} {}

  /**
   * Build a column-major brick layout
   * @param extent the extent (in units of bricks)
   */
  explicit BrickLayout(const std::array<unsigned, RANK> extent)
      : indexInStorage{buildColumnMajorGrid(extent)}
      , numBricks{computeNumBricks()}
  {}

  /**
       * @return the number of bricks in this layout
   */
  FORCUDA INLINE unsigned size() const { return numBricks; }

  /**
   * Build, cache, and return
   * an adjacency list with communication in the given dimensions
   *
   * @tparam CommunicatingDims the dimensions communication must occur in
   * @return the brick info
   * @see BrickInfo
   * @see CommDims
   */
  template <typename CommunicatingDims = CommDims<>>
  std::shared_ptr<BrickInfo<RANK, CommunicatingDims>> getBrickInfoPtr() {
    typedef BrickInfo<RANK, CommunicatingDims> BrickInfoType;
    typedef BrickInfoWrapper<Rank, CommunicatingDims> BrickInfoWrapperType;

    // look for BrickInfo in cache
    std::shared_ptr<BrickInfoWrapperBase<Rank>> key(new BrickInfoWrapperType);
    auto iterator = cachedBrickInfo.find(key);
    // compute brick info if not  already cached
    if (iterator == cachedBrickInfo.end()) {
      BrickInfoType bInfo(numBricks);
      std::vector<long> extentAsVector, strideAsVector;
      extentAsVector.reserve(RANK);
      strideAsVector.reserve(RANK);
      long stride = 1;
      for (unsigned d = 0; d < RANK; ++d) {
        extentAsVector.push_back(indexInStorage.extent[d]);
        strideAsVector.push_back(stride);
        stride *= indexInStorage.extent[d];
      }
      assert(stride == numBricks);
      init_iter<RANK, RANK>(
          extentAsVector, strideAsVector, bInfo, indexInStorage.getData().get(),
          indexInStorage.getData().get(),
          indexInStorage.getData().get() + numBricks, RunningTag());
      auto insertHandle = insertIntoCache(bInfo);;
      assert(insertHandle.second);
      iterator = insertHandle.first;
      assert(iterator != cachedBrickInfo.end());
    }
    std::shared_ptr<BrickInfoWrapperBase<Rank>> wrapperPtr = *iterator;
    return reinterpret_cast<BrickInfoWrapperType *>(wrapperPtr.get())
        ->brickInfo;
  }

#ifdef __CUDACC__
  /**
   * Build, cache, and return an adjacency list stored on the
   * device with communication in the given dimensions
   *
   * @tparam CommunicatingDims the dimensions communication must occur in
   * @return the brick info
   * @see BrickInfo
   * @see CommDims
   */
  template <typename CommunicatingDims = CommDims<>>
  std::shared_ptr<BrickInfo<RANK, CommunicatingDims>> getBrickInfoDevicePtr() {
    typedef BrickInfo<RANK, CommunicatingDims> BrickInfoType;
    typedef BrickInfoWrapper<Rank, CommunicatingDims> BrickInfoWrapperType;

    // look for BrickInfo in cache
    std::shared_ptr<BrickInfoWrapperBase<Rank>> key(new BrickInfoWrapperType);
    auto iterator = cachedBrickInfoOnDev.find(key);
    // compute brick info if not  already cached
    if (iterator == cachedBrickInfoOnDev.end()) {
      std::shared_ptr<BrickInfo<RANK, CommunicatingDims>> brickInfoPtr =
          getBrickInfoPtr<CommunicatingDims>();
      BrickInfoType _brickInfo_dev =
                        movBrickInfo(*brickInfoPtr, cudaMemcpyHostToDevice),
                    *brickInfo_dev;

      size_t brickInfoSize = sizeof(BrickInfoType);
      cudaCheck(cudaMalloc(&brickInfo_dev, brickInfoSize));
      cudaCheck(cudaMemcpy(brickInfo_dev, &_brickInfo_dev, brickInfoSize,
                           cudaMemcpyHostToDevice));
      auto adj_dev = _brickInfo_dev.adj; // Need this for lambda capture
      std::shared_ptr<BrickInfoType> bInfoSharedPtr(
          brickInfo_dev, [adj_dev](BrickInfoType *p) {
            cudaCheck(cudaFree(adj_dev));
            cudaCheck(cudaFree(p));
          });
      // insert into the cache
      *reinterpret_cast<BrickInfoWrapperType *>(key.get()) =
          BrickInfoWrapperType(bInfoSharedPtr);
      auto insertHandle = cachedBrickInfoOnDev.insert(key);
      assert(insertHandle.second);
      iterator = insertHandle.first;
      assert(iterator != cachedBrickInfoOnDev.end());
    }
    std::shared_ptr<BrickInfoWrapperBase<Rank>> wrapperPtr = *iterator;
    return reinterpret_cast<BrickInfoWrapperType *>(wrapperPtr.get())
        ->brickInfo;
  }
#endif
};
}  // end brick namespace

#endif // BRICK_BRICKLAYOUT_H
