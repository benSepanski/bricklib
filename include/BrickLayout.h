//
// Created by Ben_Sepanski on 11/6/2021.
//

#ifndef BRICK_BRICKLAYOUT_H
#define BRICK_BRICKLAYOUT_H

#include "Array.h"
#include "bricksetup.h"
#include <unordered_map>

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

/**
 * Generic template for partial specialization
 * @tparam CommunicatingDims the dimensions communication occurs in
 * @see CommunicatingDimsKeyGenerator
 */
template<typename CommunicatingDims>
struct CommunicatingDimsKeyGenerator;

/**
 * Used to hash BrickInfo's based on their CommunicatingDims
 * @tparam CommInDim false if no communication in that dimension
 */
template<bool ... CommInDim>
struct CommunicatingDimsKeyGenerator<CommDims<CommInDim...>> {
  const std::vector<bool> key = {CommInDim...};
};
} // End anonymous namespace

/**
 * A layout of bricks in memory
 *
 * @tparam Rank the rank of the layout
 */
template <unsigned Rank> struct BrickLayout {
  // constexprs and typedefs
private:
  // private typedefs
  typedef std::unordered_map<
      std::vector<bool>,
      std::shared_ptr<BrickInfoWrapperBase<Rank>>
      > mapCommDimsToBrickInfoWrapper;
public:
  // public constexprs and typedefs
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

  /**
   * Insert brickInfo into the provided cache
   * @tparam CommunicatingDims the communicating dimensions
   * @param cache the cache to insert into
   * @param brickInfoPtr pointer to the adjacency list to insert
   * @return the iterator returned from the insertion
   */
  template<typename CommunicatingDims>
  std::pair<typename mapCommDimsToBrickInfoWrapper::iterator, bool>
  static insertIntoCache(mapCommDimsToBrickInfoWrapper &cache,
                         std::shared_ptr<BrickInfo<Rank, CommunicatingDims> > brickInfoPtr) {
    typedef BrickInfo<Rank, CommunicatingDims> BrickInfoType;
    typedef BrickInfoWrapper<Rank, CommunicatingDims> BrickInfoWrapperType;
    // build and insert the wrapper
    std::vector<bool> key = CommunicatingDimsKeyGenerator<CommunicatingDims>().key;
    std::shared_ptr<BrickInfoWrapperBase<Rank>> value(new BrickInfoWrapperType(brickInfoPtr));
    auto insertHandle = cache.insert(std::make_pair(key, value));
    return insertHandle;
  }

  /**
   *
   * @tparam CommunicatingDims the communicating dimensions
   * @param cache the cache to retrieve from
   * @return the result of find()
   */
  template<typename CommunicatingDims>
  typename mapCommDimsToBrickInfoWrapper::iterator
  static getFromCache(mapCommDimsToBrickInfoWrapper &cache) {
    std::vector<bool> key = CommunicatingDimsKeyGenerator<CommunicatingDims>().key;
    return cache.find(key);
  }

  /// Members
private:
  // private members

  // This is a shared pointer so that copies of BrickLayout share a cache.
  // The entries are pointers so that we can store derived classes in the
  // cache
  std::shared_ptr<mapCommDimsToBrickInfoWrapper> brickInfoCachePtr{new mapCommDimsToBrickInfoWrapper};
#ifdef __CUDACC__
  std::shared_ptr<mapCommDimsToBrickInfoWrapper> brickInfo_devCachePtr{new mapCommDimsToBrickInfoWrapper};
#endif
public:
  // public members
  // Brick-index to index in storage
  const ArrayType indexInStorage;
  const unsigned numBricks; //< number of bricks

  /// Methods
private:
  /**
   * Compute number of bricks
   * @return The number of bricks
   */
  unsigned computeNumBricks() const {
    // number of bricks is maxIndex + 1
    unsigned maxIndex = 0;
    for(const auto &index : indexInStorage) {
      maxIndex = std::max(index, maxIndex);
    }
    return maxIndex+1;
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
   * Build a layout from the provided array
   * @tparam CommunicatingDims as used by BrickInfo
   * @param indexInStorage Each entry holds the index of the brick
   *                       at the given logical index
   * @param brickInfoPtr an adjacency list that has already been computed to be
   *                  cached on this object
   * @see BrickInfo
   */
  template<typename CommunicatingDims>
  explicit BrickLayout(const ArrayType indexInStorage,
                       std::shared_ptr<BrickInfo<RANK, CommunicatingDims> > brickInfoPtr)
      : indexInStorage{indexInStorage}, numBricks{brickInfoPtr->nbricks} {
    if(computeNumBricks() > numBricks) {
      throw std::runtime_error("indexInStorage accesses bricks further than brickInfoPtr->nbricks");
    }
    insertIntoCache(*brickInfoCachePtr, brickInfoPtr);
  }

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
   * @tparam CommunicatingDims the dimensions communication occurs in
   * @return the brick info
   * @see BrickInfo
   * @see CommDims
   */
  template <typename CommunicatingDims>
  std::shared_ptr<BrickInfo<RANK, CommunicatingDims >> getBrickInfoPtr() {
    typedef BrickInfo<RANK, CommunicatingDims> BrickInfoType;
    typedef BrickInfoWrapper<Rank, CommunicatingDims> BrickInfoWrapperType;

    // look for BrickInfo in cache
    auto iterator = getFromCache<CommunicatingDims>(*brickInfoCachePtr);
    // compute brick info if not  already cached
    if (iterator == brickInfoCachePtr->end()) {
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
      init_iter<RANK, RANK>(
          extentAsVector, strideAsVector, bInfo, indexInStorage.getData().get(),
          indexInStorage.getData().get(),
          indexInStorage.getData().get() + indexInStorage.numElements, RunningTag());
      // Build a pointer to the BrickInfo
      std::shared_ptr<BrickInfoType> bInfoPtr(
          new BrickInfoType(bInfo), [](BrickInfoType *p) {
            free(p->adj);
            delete p;
          });
      auto insertHandle = insertIntoCache(*brickInfoCachePtr, bInfoPtr);
      assert(insertHandle.second);
      iterator = insertHandle.first;
      assert(iterator != brickInfoCachePtr->end());
    }
    std::shared_ptr<BrickInfoWrapperBase<Rank>> wrapperPtr = iterator->second;
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
    auto iterator = getFromCache<CommunicatingDims>(*brickInfo_devCachePtr);
    // compute brick info if not  already cached
    if (iterator == brickInfo_devCachePtr->end()) {
      std::shared_ptr<BrickInfo<RANK, CommunicatingDims>> brickInfoPtr =
          getBrickInfoPtr<CommunicatingDims>();
      BrickInfoType brickInfoWithDataOnDev =
                        movBrickInfo(*brickInfoPtr, cudaMemcpyHostToDevice),
                    *brickInfo_devPtr;
      size_t brickInfoSize = sizeof(BrickInfoType);
      cudaCheck(cudaMalloc(&brickInfo_devPtr, brickInfoSize));
      cudaCheck(cudaMemcpy(brickInfo_devPtr, &brickInfoWithDataOnDev, brickInfoSize,
                           cudaMemcpyHostToDevice));
      // Build a pointer to the BrickInfo
      auto adj_dev = brickInfoWithDataOnDev.adj; // Need this for lambda capture
      std::shared_ptr<BrickInfoType> bInfoSharedPtr_dev(brickInfo_devPtr,
          [adj_dev](BrickInfoType *p) {
            cudaCheck(cudaFree(adj_dev));
            cudaCheck(cudaFree(p));
          });
      auto insertHandle = insertIntoCache(*brickInfo_devCachePtr, bInfoSharedPtr_dev);
      assert(insertHandle.second);
      iterator = insertHandle.first;
      assert(iterator != brickInfo_devCachePtr->end());
    }
    std::shared_ptr<BrickInfoWrapperBase<Rank>> wrapperPtr = iterator->second;
    return reinterpret_cast<BrickInfoWrapperType *>(wrapperPtr.get())
        ->brickInfo;
  }
#endif
};
}  // end brick namespace

#endif // BRICK_BRICKLAYOUT_H
