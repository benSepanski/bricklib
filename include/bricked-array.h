//
// Created by Ben_Sepanski on 10/27/2021.
//

#ifndef BRICK_BRICKED_ARRAY_H
#define BRICK_BRICKED_ARRAY_H

#include "array.h"
#include "brick.h"
#include "bricksetup.h"
#include "template-utils.h"

#include <unordered_set>

namespace brick {
  namespace { // Anonymous namespace
    /**
     * Base class so that we can handle multiple BrickInfo types
     * with differing CommunicatingDims in a single object
     * @tparam Rank the rank of the BrickInfo
     */
    template <unsigned Rank> struct BrickInfoWrapperBase {
      typedef BrickInfo<Rank> MyBrickInfoType;
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
      virtual ~BrickInfoWrapperBase() { }
    };
  } // End anonymous namespace
}

namespace std { // std namespace
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
} // End std namespace


namespace brick {
  namespace {  // Anonymous namespace
    /**
     * Generic template for partial specialization
     * @see BrickInfoWrapper
     */
    template<unsigned Rank, typename CommunicatesInDim>
    struct BrickInfoWrapper;

    /**
     * A wrapper around a BrickInfo
     * @tparam Rank the rank of the brick info
     * @tparam CommInDim the dimensions the brick info is communicating in
     */
    template<unsigned Rank, bool ... CommInDim>
    struct BrickInfoWrapper<Rank, CommDims<CommInDim...> > : public BrickInfoWrapperBase<Rank> {
      typedef BrickInfo<Rank, CommDims<CommInDim...> > MyBrickInfoType;
      std::shared_ptr<MyBrickInfoType> brickInfo;

      /**
       * Default constructor
       */
      explicit BrickInfoWrapper()
      : brickInfo{nullptr}
      { }

      /**
       * Take a copy of bInfo
       * @param bInfo the BrickInfo to copy
       */
      explicit BrickInfoWrapper(std::shared_ptr<MyBrickInfoType> bInfo)
      : brickInfo{bInfo}
      { }

      /**
       * Destructor
       */
      virtual ~BrickInfoWrapper() override { }

      /**
       * @return a representation of the communicating dims
       */
      static std::vector<bool> communicatingDims() {
        return {CommInDim...};
      }

      std::vector<bool> getCommunicatingDims() const override {
        return communicatingDims();
      }
    };
  } // End anonymous namespace

  template<unsigned Rank>
  struct BrickLayout {
    static constexpr unsigned RANK = Rank;
    private:
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
            throw std::runtime_error("Brick grid does not fit inside of unsigned type");
          }
          nBricks = newNumBricks;
        }
        return nBricks;
      }

      std::unordered_set<std::shared_ptr<BrickInfoWrapperBase<Rank> > > cachedBrickInfo;

    public:
      // Public typedefs
      typedef Array<unsigned, RANK, Padding<>, unsigned, unsigned> ArrayType;
      // Brick-index to index in storage
      const ArrayType indexInStorage;
      const unsigned numBricks; //< number of bricks

      /**
       * Build a layout from the provided array
       * @param indexInStorage Each entry holds the index of the brick
       *                       at the given logical index
       */
      explicit BrickLayout(const ArrayType indexInStorage)
      : indexInStorage{indexInStorage}
      , numBricks{computeNumBricks()}
      { }

      explicit BrickLayout(const std::array<unsigned, RANK> extent)
      : indexInStorage{buildColumnMajorGrid(extent)}
      , numBricks{computeNumBricks()}
      { }

      /**
       * @return the number of bricks in this layout
       */
      FORCUDA INLINE
      unsigned size() const {
        return numBricks;
      }

      /**
       * Build, cache, and return
       * an adjacency list with communication in the given dimensions
       *
       * @tparam CommunicatingDims the dimensions communication must occur in
       * @return the brick info
       * @see BrickInfo
       * @see CommDims
       */
      template<typename CommunicatingDims = CommDims<>>
      std::shared_ptr<BrickInfo<RANK, CommunicatingDims> > getBrickInfoPtr() {
        typedef BrickInfo<RANK, CommunicatingDims> BrickInfoType;
        typedef BrickInfoWrapper<Rank, CommunicatingDims> BrickInfoWrapperType;

        // look for BrickInfo in cache
        std::shared_ptr<BrickInfoWrapperBase<Rank> > key(new BrickInfoWrapperType);
        auto iterator = cachedBrickInfo.find(key);
        // compute brick info if not  already cached
        if(iterator == cachedBrickInfo.end()) {
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
              extentAsVector, strideAsVector, bInfo, indexInStorage.getData(),
              indexInStorage.getData(), indexInStorage.getData() + numBricks,
              RunningTag());
          // Build a pointer to the BrickInfo
          std::shared_ptr<BrickInfoType>
              bInfoPtr((BrickInfoType*)malloc(sizeof(BrickInfoType)),
                       [](BrickInfoType * p) {free(p->adj); free(p);});
          *bInfoPtr.get() = bInfo;
          // insert into the cache
          *reinterpret_cast<BrickInfoWrapperType *>(key.get()) = BrickInfoWrapperType(bInfoPtr);
          auto insertHandle = cachedBrickInfo.insert(key);
          assert(insertHandle.second);
          iterator = insertHandle.first;
          assert(iterator != cachedBrickInfo.end());
        }
        std::shared_ptr<BrickInfoWrapperBase<Rank> > wrapperPtr = *iterator;
        return reinterpret_cast<BrickInfoWrapperType*>(wrapperPtr.get())->brickInfo;
      }

    private:
      static ArrayType buildColumnMajorGrid(const std::array<unsigned, RANK> extent) {
        ArrayType arr(extent);

        unsigned index = 0;
        for(unsigned &element : arr) {
          element = index++;
        }
        return arr;
      }
  };

  /**
   * Generic template for specialization
   * @tparam DataType the data-type of the brick (currently must be bElem or bComplexElem)
   * @tparam BrickDims the brick-dimensions (e.g. Dim<K,J,I>)
   * @tparam VectorFolds the vector-folds (e.g. Dim<vj, vi>).
   *                     Automatically left-padded with 1s
   *                     until its length is the same as that of BrickDims.
   */
  template<typename DataType, typename BrickDims, typename VectorFolds = Dim<1>,
           typename = typename brick::templateutils::UnsignedIndexSequence<BrickDims::NUM_DIMS>::type >
  class BrickedArray;

  /**
   * An array stored in bricks
   * @tparam BDim the brick dimensions
   * @tparam VFold the vector folds
   * @tparam _Range0ToRank AUTO-GENERATED TO BE 0,1,2,...,RANK-1: USER SHOULD NOT SUPPLY THESE
   */
  template<typename DataType, unsigned ... BDim, unsigned ... VFold, unsigned ... Range0ToRank>
  class BrickedArray<DataType, Dim<BDim...>, Dim<VFold...>,
                     brick::templateutils::ParameterPackManipulator<unsigned>::Pack<Range0ToRank...> > {

    /// Static constants and typedefs
    private:
      // used to manipulate unsigned template parameter packs
      using UnsignedPackManip = brick::templateutils::ParameterPackManipulator<unsigned>;
      // std::multiply<>() is not constexpr until C++14
      static constexpr unsigned multiply(const unsigned x, const unsigned y) {
        return x * y;
      }
      static constexpr std::array<bool, sizeof...(BDim)> allFalse{}; //< all false

    public:
      // public static constants
      static constexpr unsigned RANK = sizeof...(BDim);
      static constexpr ARRAY_TYPE(unsigned, sizeof...(BDim))
          BRICK_DIMS = UnsignedPackManip::reverse<BDim...>() IF_CUDA_ELSE(.get(), );
      static constexpr ARRAY_TYPE(unsigned, sizeof...(BDim))
          VECTOR_FOLDS = UnsignedPackManip::padAndReverse<RANK, 1, VFold...>() IF_CUDA_ELSE(.get(), );
      static constexpr unsigned NUM_ELEMENTS_PER_BRICK = brick::templateutils::reduce(multiply, 1U, BDim...);
      static constexpr bool isComplex = std::is_same<DataType, bComplexElem>::value;
      
      // public typedefs
      typedef CommDims<allFalse[Range0ToRank]...> NoCommunication;
      typedef BrickedArray<DataType, Dim<BDim...>, Dim<VFold...> > MyType;
      template<typename CommunicatingDims = CommDims<>>
      using BrickType = Brick<Dim<BDim...>, Dim<VFold...>, isComplex, CommunicatingDims>;

    /// Members
    private:
      BrickLayout<RANK> layout;  //< layout of bricks in storage
      // extent of the array being bricked
      const ARRAY_TYPE(unsigned, RANK) extent;
      BrickStorage brickStorage; //< Where the data is actually stored
      // location of first brick in storage (in units of real elements)
      const unsigned offsetIntoBrickStorage;
      // bricks without any adjacency list
      BrickType<NoCommunication> bricks;

    /// Methods
    private:
      /**
       * Builds BrickStorage
       *
       * @param arrayExtent the extent to validate/compute number of bricks
       * @param use_mmap if true, use mmap for allocation
       * @param mmap_fd the file descriptor to use if using mmap, or nullptr
       *                to create one.
       * @param offset the offset into the file descriptor to use, if provided
       * @throw std::runtime_error if the arrayExtent is not valid
       * @return the storage
       */
      BrickStorage buildStorage(std::array<unsigned, RANK> arrayExtent,
                                bool use_mmap = false,
                                void *mmap_fd = nullptr,
                                size_t offset = 0) const {
        // ensure array extent is non-zero and divisible by brick extent
        for (unsigned d = 0; d < RANK; ++d) {
          if (arrayExtent[d] == 0) {
            throw std::runtime_error("Array extent on axis is zero");
          }
          if (arrayExtent[d] % BRICK_DIMS[d] != 0) {
            throw std::runtime_error("Extent in axis is not divisible by brick-dimension");
          }
        }
        // check for possible overflow in number of elements
        size_t numElements = 1;
        for (unsigned d = 0; d < RANK; ++d) {
          assert(arrayExtent[d] != 0);
          size_t newNumElements = arrayExtent[d] * numElements;
          if (newNumElements / arrayExtent[d] != numElements) {
            throw std::runtime_error("Array does not fit inside of size_t");
          }
          numElements = newNumElements;
        }
        // Build BrickStorage
        long step = NUM_ELEMENTS_PER_BRICK * (isComplex ? 2 : 1);
        BrickStorage storage;
        if(use_mmap) {
          if (mmap_fd == nullptr) {
            storage = BrickStorage::mmap_alloc(layout.size(), step);
          } else {
            storage =
                BrickStorage::mmap_alloc(layout.size(), step, mmap_fd, offset);
          }
        } else {
          storage = BrickStorage::allocate(layout.size(), step);
        }
        return storage;
      }

      /**
       * Used to distinguish an unsigned value at the ind
       */
      struct UnsignedWrapper {
        unsigned value;
      };

      /**
       * Base case for brick access
       */
      static inline DataType& accessBrick(DataType &value) {
        return value;
      }

      /**
       * Used to translate (idx0,idx1,...,idxD) to [idx0][idx1][...][idxD-1]
       */
      template<typename BrickAccessorType, typename I, typename ... IndexType>
      static inline DataType& accessBrick(BrickAccessorType b, I lastIndex, IndexType ... earlierIndices) {
        return accessBrick(b[lastIndex], earlierIndices...);
      }

    public:
      /// Constructor methods

      /**
       * Allocate a BrickedArray
       *
       * @param layout the desired layout of bricks in memory
       */
      explicit BrickedArray(BrickLayout<RANK> layout)
      : extent {layout.indexInStorage.extent[Range0ToRank] * BRICK_DIMS[Range0ToRank]...}
      , layout{layout}
      , offsetIntoBrickStorage {0}
      , brickStorage{buildStorage(extent)}
      , bricks{getBricks<NoCommunication>()}
      { }

      /**
       * Allocate a BrickedArray using mmap.
       *
       * @param layout the desired layout of bricks in memory
       * @param mmap_fd the file to mmap into, or nullptr if a new one should
       *                be created
       * @param offset Ignored if mmap_fd is null. Otherwise, the offset
       *               into mmap_fd that is used for allocation.
       */
      explicit BrickedArray(BrickLayout<RANK> layout,
                            void *mmap_fd,
                            size_t offset)
      : extent {layout.indexInStorage.extent[Range0ToRank] * BRICK_DIMS[Range0ToRank]...}
      , layout{layout}
      , offsetIntoBrickStorage {0}
      , brickStorage{buildStorage(extent, true, mmap_fd, offset)}
      , bricks{getBricks<NoCommunication>()}
      { }
      
      /**
       * @tparam CommunicatingDims the dimensions in which communication is
       *                           required
       * @return the array represented as a list of bricks 
       */
      template<typename CommunicatingDims = CommDims<>>
      BrickType<CommunicatingDims> getBricks() {
        std::shared_ptr<BrickInfo<RANK, CommunicatingDims> > brickInfoPtr =
            layout.template getBrickInfoPtr<CommunicatingDims>();
        return BrickType<CommunicatingDims>(brickInfoPtr.get(), brickStorage, offsetIntoBrickStorage);
      }

      /**
       * Deep copy an array into this bricked array
       * @tparam SizeType the size type of the array
       * @tparam IndexType the index type of the array
       * @tparam ArrayPadding any array padding
       * @param arr the array to copy. Extent must match this object
       */
      template<typename SizeType, typename IndexType, unsigned ... ArrayPadding>
      void loadFrom(const Array<DataType, RANK, Padding<ArrayPadding...>, SizeType, IndexType> &arr) {
        // Ensure extents match
        for(unsigned d = 0; d < RANK; ++d) {
          if(arr.extent[d] != this->extent[d]) {
            throw std::runtime_error("Array extent does not match this object");
          }
        }
        constexpr std::array<long, RANK> allZero{};  //< all zero
        // Get bricks
        auto brick = getBricks<NoCommunication>();
        // get arrays stored into vectors
        std::vector<long> extentAsVector = {extent[Range0ToRank]...},
                          paddingAsVector = {arr.PADDING[Range0ToRank]...},
                          ghostAsVector = {allZero[Range0ToRank]...};
        // Load into bricks
        copyToBrick<RANK>(extentAsVector,
                          paddingAsVector,
                          ghostAsVector,
                          arr.getData(),
                          layout.indexInStorage.getData(),
                          brick) ;
      }

      /**
       * Deep copy into an array from this bricked array
       * @tparam SizeType the size type of the array
       * @tparam IndexType the index type of the array
       * @tparam ArrayPadding any array padding
       * @param arr the array to copy into. Extent must match this object
       */
      template<typename SizeType, typename IndexType, unsigned ... ArrayPadding>
      void storeTo(Array<DataType, RANK, Padding<ArrayPadding...>, SizeType, IndexType> &arr) {
        // Ensure extents match
        for(unsigned d = 0; d < RANK; ++d) {
          if(arr.extent[d] != this->extent[d]) {
            throw std::runtime_error("Array extent does not match this object");
          }
        }
        constexpr std::array<long, RANK> allZero{};  //< all zero
        // Get bricks
        auto brick = getBricks<NoCommunication>();
        // get arrays stored into vectors
        std::vector<long> extentAsVector = {extent[Range0ToRank]...},
                          paddingAsVector = {arr.PADDING[Range0ToRank]...},
                          ghostAsVector = {allZero[Range0ToRank]...};
        // Copy from bricks
        copyFromBrick<RANK>(extentAsVector,
                            paddingAsVector,
                            ghostAsVector,
                            arr.getData(),
                            layout.indexInStorage.getData(),
                            brick) ;
      }

      /**
       * Inefficient access into the bricks array
       *
       * @tparam IndexType type of variadic parameter pack
       * @param indices the indices into the array
       * @return a reference to the element at the provided index
       */
      template<typename ... IndexType>
      DataType& operator()(IndexType ... indices) {
        static_assert(sizeof...(IndexType) == RANK, "Mismatch in number of indices");
        unsigned brickIndex = layout.indexInStorage.get(indices / BRICK_DIMS[Range0ToRank] ...);
        auto b = bricks[brickIndex];
        return templateutils::callOnReversedArgs<DataType&>(
            accessBrick<decltype(b), IndexType...>, indices % BRICK_DIMS[Range0ToRank]..., b
        );
      }

      // Some validity checks
      static_assert(std::is_same<DataType, bElem>::value
                        || std::is_same<DataType, bComplexElem>::value,
                    "bElem and bComplexElem are the only data types supported");
      static_assert(sizeof...(VFold) <= sizeof...(BDim),
                    "Number of vector-folds must be at most the number of brick-dimensions");
      static_assert(sizeof...(VFold) >= 1,
                    "Must provide at least one vector-fold");
      // Verify that brick-dims are divisible by vector folds
      template<unsigned d>
      static constexpr typename std::enable_if<0 != d, bool>::type vfoldDividesBDim() {
        static_assert(d < RANK, "d out of bounds");
        return BRICK_DIMS[d] % VECTOR_FOLDS[d] == 0 && vfoldDividesBDim<d-1>();
      }
      template<unsigned d>
      static constexpr typename std::enable_if<0 == d, bool>::type vfoldDividesBDim() {
        return BRICK_DIMS[d] % VECTOR_FOLDS[d] == 0;
      }
      static_assert(vfoldDividesBDim<RANK-1>(), "Vector folds must divide brick dimensions");
      // Make sure _Range0ToRank is correct
      static_assert(std::is_same<brick::templateutils::ParameterPackManipulator<unsigned>::Pack<Range0ToRank...>,
                                 typename brick::templateutils::UnsignedIndexSequence<sizeof...(BDim)>::type
                                 >::value,
                    "User should not supply _Range0ToRank template argument");
  };

  // Define public static constants
  template<typename DataType, unsigned ... BDim, unsigned ... VFold, unsigned ... Range0ToRank>
  constexpr ARRAY_TYPE(unsigned, sizeof...(BDim))
  BrickedArray<DataType, Dim<BDim...>, Dim<VFold...>,
               brick::templateutils::ParameterPackManipulator<unsigned>::Pack<Range0ToRank...>
               >::BRICK_DIMS;
  template<typename DataType, unsigned ... BDim, unsigned ... VFold, unsigned ... Range0ToRank>
  constexpr ARRAY_TYPE(unsigned, sizeof...(BDim))
      BrickedArray<DataType, Dim<BDim...>, Dim<VFold...>,
                   brick::templateutils::ParameterPackManipulator<unsigned>::Pack<Range0ToRank...>
                   >::VECTOR_FOLDS;
}

#endif // BRICK_BRICKED_ARRAY_H
