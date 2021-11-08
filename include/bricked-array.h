//
// Created by Ben_Sepanski on 10/27/2021.
//

#ifndef BRICK_BRICKED_ARRAY_H
#define BRICK_BRICKED_ARRAY_H

#include "array.h"
#include "brick.h"
#include "bricklayout.h"
#include "bricksetup.h"
#include "ManagedBrickStorage.h"
#include "template-utils.h"

#include <tuple>
#include <utility>

#ifdef __CUDACC__
#include "brick-cuda.h"
#endif

namespace brick {
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
      template<unsigned ... PackValues>
      using UnsignedPack = UnsignedPackManip::Pack<PackValues...>;
      static constexpr std::array<bool, sizeof...(BDim)> allFalse{}; //< all false

    public:
      // public static constants
      static constexpr unsigned RANK = sizeof...(BDim);
      static constexpr unsigned BRICK_DIMS[sizeof...(BDim)] = {
        UnsignedPackManip::get<UnsignedPack<BDim...> >(RANK - 1 - Range0ToRank)...
      };
      static constexpr unsigned VECTOR_FOLDS[sizeof...(BDim)] = {
        UnsignedPackManip::getOrDefault<UnsignedPack<VFold...>, 1>(
              sizeof...(VFold) - 1 - Range0ToRank
        )...
      };
      static constexpr unsigned NUM_ELEMENTS_PER_BRICK = brick::templateutils::reduce(templateutils::multiply<unsigned>, 1U, BDim...);
      static constexpr bool isComplex = std::is_same<DataType, bComplexElem>::value;
      
      // public typedefs
      typedef CommDims<allFalse[Range0ToRank]...> NoCommunication;
      template<typename CommunicatingDims = CommDims<>>
      using BrickType = Brick<Dim<BDim...>, Dim<VFold...>, isComplex, CommunicatingDims>;

    /// Static functions
    private:
      // private static functions
#pragma clang diagnostic push
#pragma ide diagnostic ignored "OCUnusedGlobalDeclarationInspection" // ignore some clang-tidy stuff
      /**
       * Base case for brick access
       */
      static inline DataType& accessBrick(DataType &value) {
        return value;
      }
#pragma clang diagnostic pop

      /**
       * Used to translate (idx0,idx1,...,idxD) to [idx0][idx1][...][idxD-1]
       */
      template<typename BrickAccessorType, typename I, typename ... IndexType>
      static inline DataType& accessBrick(BrickAccessorType b, I lastIndex, IndexType ... earlierIndices) {
        return accessBrick(b[lastIndex], earlierIndices...);
      }
    public:
      // public static functions

    /// Members
    public:
      // public members
      // extent of the array being bricked
      const unsigned extent[RANK];
    private:
      // private members
      BrickLayout<RANK> layout;  //< layout of bricks in storage
      ManagedBrickStorage brickStorage; //< Where the data is actually stored
      // location of first brick in storage (in units of real elements)
      const unsigned offsetIntoBrickStorage;
      // bricks without any adjacency list
      BrickType<NoCommunication> bricks;

    /// Methods
    private:
      // private methods
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
      brick::ManagedBrickStorage buildStorage(std::array<unsigned, RANK> arrayExtent,
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
          return brick::ManagedBrickStorage(layout.size(), step, mmap_fd, offset);
        } else {
          return brick::ManagedBrickStorage(layout.size(), step);
        }
      }

    public:
      /**
       * @tparam CommunicatingDims the dimensions in which communication is
       *                           required
       * @return the array represented as a list of bricks
       */
      template<typename CommunicatingDims = CommDims<>>
      BrickType<CommunicatingDims> viewBricks() {
        std::shared_ptr<BrickInfo<RANK, CommunicatingDims> > brickInfoPtr =
            layout.template getBrickInfoPtr<CommunicatingDims>();
        return BrickType<CommunicatingDims>(brickInfoPtr.get(),
                                            brickStorage.getHostStorage(),
                                            offsetIntoBrickStorage);
      }

      /// Constructor methods

      /**
       * Allocate a BrickedArray
       *
       * @param layout the desired layout of bricks in memory
       */
      explicit BrickedArray(BrickLayout<RANK> layout)
      : extent{layout.indexInStorage.extent[Range0ToRank] * BRICK_DIMS[Range0ToRank]...}
      , layout{layout}
      , offsetIntoBrickStorage {0}
      , brickStorage{buildStorage({extent[Range0ToRank]...})}
      , bricks{viewBricks<NoCommunication>()}
      { }

      /**
       * Build a BrickedArray from an existing BrickStorage.
       *
       * Users should avoid calling this directly unless they
       * are familiar with the bricks library.
       *
       * @param layout the desired layout of bricks in memory
       * @see InterleavedBrickArrays
       */
      explicit BrickedArray(BrickLayout<RANK> layout,
                            ManagedBrickStorage storage,
                            unsigned offsetIntoStorage)
          : extent{layout.indexInStorage.extent[Range0ToRank] * BRICK_DIMS[Range0ToRank]...}
          , layout{layout}
          , offsetIntoBrickStorage {offsetIntoStorage}
          , brickStorage{std::move(storage)}
          , bricks{viewBricks<NoCommunication>()}
      { }

#pragma clang diagnostic push
#pragma ide diagnostic ignored "OCUnusedGlobalDeclarationInspection" // Ignore some clang-tidy stuff
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
      , brickStorage{buildStorage({extent[Range0ToRank]...}, true, mmap_fd, offset)}
      , bricks{viewBricks<NoCommunication>()}
      { }
#pragma clang diagnostic pop

#ifdef __CUDACC__
      /**
       * @note This does not copy data, just malloc it (if not already done)
       *
       * @tparam CommunicatingDims the dimensions in which communication is
       *                           required
       * @return the array represented as a list of bricks with the data stored
       *         on the device
       */
      template<typename CommunicatingDims = CommDims<>>
      BrickType<CommunicatingDims> viewBricksOnDevice() {
        std::shared_ptr<BrickInfo<RANK, CommunicatingDims>> brickInfo_dev =
            layout.template getBrickInfoDevicePtr<CommunicatingDims>();
        return BrickType<CommunicatingDims>(brickInfo_dev.get(),
                                            brickStorage.getCudaStorage(),
                                            offsetIntoBrickStorage);
      }

      /**
       * IMPORTANT: As a side effect, this also copies any interleaved
       *            fields to the device
       *
       * Copy data to the device
       */
      void copyToDevice() {
        size_t dataSize = brickStorage.step * brickStorage.chunks * sizeof(bElem);
        cudaCheck(cudaMemcpy(brickStorage.getCudaStorage().dat.get(),
                                brickStorage.getHostStorage().dat.get(),
                                dataSize,
                                cudaMemcpyHostToDevice));
      }

      /**
       * IMPORTANT: As a side effect, this also copies any interleaved
       *            fields from the device
       *
       * @tparam CommunicatingDims the dimensions in which communication is
       *                           required
       * @return the array represented as a list of bricks with data stored
       *         on the device
       */
      void copyFromDevice() {
        size_t dataSize = brickStorage.step * brickStorage.chunks * sizeof(bElem);
        cudaCheck(cudaMemcpy(brickStorage.getHostStorage().dat.get(),
                                brickStorage.getCudaStorage().dat.get(),
                                dataSize,
                                cudaMemcpyDeviceToHost));
      }
#endif

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
        auto brick = viewBricks<NoCommunication>();
        // get arrays stored into vectors
        std::vector<long> extentAsVector = {extent[Range0ToRank]...},
                          paddingAsVector = {arr.PADDING(Range0ToRank)...},
                          ghostAsVector = {allZero[Range0ToRank]...};
        // Load into bricks
        copyToBrick<RANK>(extentAsVector,
                          paddingAsVector,
                          ghostAsVector,
                          arr.getData().get(),
                          layout.indexInStorage.getData().get(),
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
        auto brick = viewBricks<NoCommunication>();
        // get arrays stored into vectors
        std::vector<long> extentAsVector = {extent[Range0ToRank]...},
                          paddingAsVector = {arr.PADDING(Range0ToRank)...},
                          ghostAsVector = {allZero[Range0ToRank]...};
        // Copy from bricks
        copyFromBrick<RANK>(extentAsVector,
                            paddingAsVector,
                            ghostAsVector,
                            arr.getData().get(),
                            layout.indexInStorage.getData().get(),
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

  // Define static constants
  template<typename DataType, unsigned ... BDim, unsigned ... VFold, unsigned ... Range0ToRank>
  constexpr unsigned BrickedArray<DataType,
                                  Dim<BDim...>,
                                  Dim<VFold...>,
                                  brick::templateutils::ParameterPackManipulator<unsigned>::Pack<Range0ToRank...>
                                  >::BRICK_DIMS[sizeof...(BDim)];
  template<typename DataType, unsigned ... BDim, unsigned ... VFold, unsigned ... Range0ToRank>
  constexpr unsigned BrickedArray<DataType,
                                  Dim<BDim...>,
                                  Dim<VFold...>,
                                  brick::templateutils::ParameterPackManipulator<unsigned>::Pack<Range0ToRank...>
                                  >::VECTOR_FOLDS[sizeof...(BDim)];


  template<typename BrickDims,
           typename ... >
  struct InterleavedBrickedArrays;

  template<typename DataType, typename VectorFold = Dim<1> >
  struct DataTypeVectorFoldPair {
    typedef DataType dataType;
    typedef VectorFold vectorFold;
  };

  template<unsigned ... BDims,
           typename ... VectorFolds,
           typename ... DataTypes>
  struct InterleavedBrickedArrays<Dim<BDims...>,
                                  DataTypeVectorFoldPair<DataTypes, VectorFolds>...>
  {
    /// constexpr/typedefs
    private:
      // private constexprs
      static constexpr unsigned NUM_ELEMENTS_PER_BRICK =
          templateutils::reduce(templateutils::multiply<unsigned>, 1, BDims...);
    public:
      // public constexpr
      static constexpr unsigned Rank = sizeof...(BDims);
    /// Members
    public:
      std::tuple<
          std::vector<BrickedArray<DataTypes, Dim<BDims...>, VectorFolds> >...
          > fields;
    /// static methods
    private:
      // private static methods
      /**
       * Base case for get offsets
       */
      template<typename ...>
      static void computeOffsets(std::vector<size_t> &offsets) {  }

      /**
       * Compute the offsets into brick storage for the provided number
       * of interleaved fields of each data type.
       */
      template<typename HeadDataType, typename ... TailDataTypes, typename ... T>
      static void computeOffsets(std::vector<size_t> &offsets, unsigned headCount, T ... tailCounts) {
        static_assert(sizeof...(T) == sizeof...(TailDataTypes),
                      "Mismatch in number of arguments");
        static_assert(sizeof(HeadDataType) % sizeof(bElem) == 0,
                      "sizeof(bElem) must divide sizeof(DataType)");
        assert(!offsets.empty());
        size_t lastOffset = offsets.back();
        for(unsigned i = 0; i < headCount; ++i) {
          lastOffset += NUM_ELEMENTS_PER_BRICK * sizeof(HeadDataType) / sizeof(bElem);
          offsets.push_back(lastOffset);
        }
        computeOffsets<TailDataTypes...>(offsets, tailCounts...);
      }

    /// Methods
    private:
      // private methods
      /**
       * Base case
       * @see initializeBrickedArrays
       */
      template<typename ...>
      void initializeBrickedArrays(const BrickLayout<Rank> &,
                                   const brick::ManagedBrickStorage &,
                                   std::vector<size_t>::const_iterator ) { }

      /**
       * initialize the bricked arrays
       * @tparam HeadDataTypeVectorFoldPair data type/vector fold pair of current array type
       * @tparam Tail remaining data type/vector fold pairs
       * @tparam T parameter pack of counts
       * @param layout the brick layout
       * @param storage the storage to use
       * @param offset the iterator over offsets
       * @param headCount number of fields of the head type to make
       * @param tailCounts number of fields of the tail types to make
       */
      template<typename HeadDataTypeVectorFoldPair, typename ... Tail, typename ... T>
      void initializeBrickedArrays(const BrickLayout<Rank> &layout,
                                   const brick::ManagedBrickStorage &storage,
                                   std::vector<size_t>::const_iterator offset,
                                   unsigned headCount, T ... tailCounts) {
        typedef typename HeadDataTypeVectorFoldPair::dataType d;
        typedef typename HeadDataTypeVectorFoldPair::vectorFold vf;
        typedef BrickedArray<d, Dim<BDims...>, vf> BrickedArrayType;
        for(unsigned i = 0; i < headCount; ++i) {
          std::get<sizeof...(DataTypes) - 1 - sizeof...(Tail)>(fields).push_back(BrickedArrayType(layout, storage, *(offset++)));
        }
        initializeBrickedArrays<Tail...>(layout, storage, offset, tailCounts...);
      }

    public:
      // public methods
      /**
       *
       * @tparam T the types of the count arguments
       * @param layout the layout to use
       * @param fieldCount the number of fields of each data type/vector fold
       *                   pair to build
       */
      template<typename ... T>
      explicit InterleavedBrickedArrays(BrickLayout<Rank> layout, T ... fieldCount) {
        static_assert(sizeof...(T) == sizeof...(DataTypes),
                      "Must provide field counts for each DataType");
        std::vector<size_t> offsets = {0};
        computeOffsets<DataTypes...>(offsets, fieldCount...);
        size_t step = offsets.back();
        offsets.pop_back();
        brick::ManagedBrickStorage storage(layout.size(), step);
        initializeBrickedArrays<DataTypeVectorFoldPair<DataTypes, VectorFolds>...>(layout, storage, offsets.cbegin(), fieldCount...);
      }

      // TODO: Write MMAP version of constructor
  };
}

#endif // BRICK_BRICKED_ARRAY_H
