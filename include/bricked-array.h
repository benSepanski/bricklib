//
// Created by Ben_Sepanski on 10/27/2021.
//

#ifndef BRICK_BRICKED_ARRAY_H
#define BRICK_BRICKED_ARRAY_H

#include "array.h"
#include "brick.h"
#include "template-utils.h"

namespace brick {
  template<unsigned Rank, typename Padding = Padding<>>
  struct BrickLayout {
    public:
      // Public typedefs
      typedef Array<unsigned, Rank, Padding, unsigned, unsigned> ArrayType;
      // Brick-index to index in storage
      const ArrayType indexInStorage;
      /**
       * Build a layout from the provided array
       * @param indexInStorage Each entry holds the index of the brick
       *                       at the given logical index
       */
      explicit BrickLayout(const ArrayType indexInStorage)
      : indexInStorage{indexInStorage}
      , numBricks{computeNumBricks()}
      { }

      explicit BrickLayout(const std::array<unsigned, Rank> extent)
      : indexInStorage{buildColumnMajorGrid(extent)}
      , numBricks{computeNumBricks()}
      { }

      /**
       * @return the number of bricks in this layout
       */
      FORCUDA INLINE
      unsigned size() {
        return numBricks;
      }

    private:
      const unsigned numBricks;

      static ArrayType buildColumnMajorGrid(const std::array<unsigned, Rank> extent) {
        ArrayType arr(extent);

        unsigned index = 0;
        for(unsigned &element : arr) {
          element = index++;
        }
        return arr;
      }

      /**
       * @throw std::runtime_error if the extent of indexInStorage is not valid
       * @return The number of bricks
       */
      unsigned computeNumBricks() const {
        // check for possible overflow in number of elements
        unsigned nBricks = 1;
        for (unsigned d = 0; d < Rank; ++d) {
          unsigned extent = indexInStorage.getExtent(d);
          assert(extent != 0);
          size_t newNumBricks = extent * nBricks;
          if (newNumBricks / extent != nBricks) {
            throw std::runtime_error("Brick grid does not fit inside of unsigned type");
          }
          nBricks = newNumBricks;
        }
        return nBricks;
      }
  };

  /**
   * Generic template for specialization
   * @tparam DataType the data-type of the brick (currently must be bElem or bComplexElem)
   * @tparam BrickDims the brick-dimensions (e.g. Dim<K,J,I>)
   * @tparam LayoutPadding padding in the Array holding the bricks layout
   * @tparam VectorFolds the vector-folds (e.g. Dim<vj, vi>).
   *                     Automatically left-padded with 1s
   *                     until its length is the same as that of BrickDims.
   */
  template<typename DataType, typename BrickDims, typename VectorFolds = Dim<>,
           typename LayoutPadding = Padding<>,
           typename = typename brick::templateutils::UnsignedIndexSequence<BrickDims::NUM_DIMS>::type >
  class BrickedArray;

  /**
   * An array stored in bricks
   * @tparam BDim the brick dimensions
   * @tparam VFold the vector folds
   * @tparam _Range0ToRank AUTO-GENERATED TO BE 0,1,2,...,RANK-1: USER SHOULD NOT SUPPLY THESE
   */
  template<typename DataType, typename LayoutPadding, unsigned ... BDim, unsigned ... VFold, unsigned ... Range0ToRank>
  class BrickedArray<DataType, Dim<BDim...>, Dim<VFold...>,
                     LayoutPadding,
                     brick::templateutils::ParameterPackManipulator<unsigned>::Pack<Range0ToRank...> > {
    // used to manipulate unsigned template parameter packs
    using UnsignedPackManip = brick::templateutils::ParameterPackManipulator<unsigned>;
    // std::multiply<>() is not constexpr until C++14
    static constexpr unsigned multiply(const unsigned x, const unsigned y) {
      return x * y;
    }

    public:
      // public typedefs
      typedef BrickedArray<DataType, Dim<BDim...>, Dim<VFold...> > MyType;

      // public static constants
      static constexpr unsigned RANK = sizeof...(BDim);
      static constexpr ARRAY_TYPE(unsigned, sizeof...(BDim))
          BRICK_DIMS = UnsignedPackManip::reverse<BDim...>() IF_CUDA_ELSE(.get(), );
      static constexpr ARRAY_TYPE(unsigned, sizeof...(BDim))
          VECTOR_FOLDS = UnsignedPackManip::padAndReverse<RANK, 1, VFold...>() IF_CUDA_ELSE(.get(), );
      static constexpr unsigned NUM_ELEMENTS_PER_BRICK = brick::templateutils::reduce(multiply, 1U, BDim...);
      static constexpr bool isComplex = std::is_same<DataType, bComplexElem>::value;

      /// Constructor methods

      /**
       * Allocate a BrickedArray
       *
       * @param layout the desired layout of bricks in memory
       */
      explicit BrickedArray(BrickLayout<RANK, LayoutPadding> layout)
      : extent {layout.indexInStorage.getExtent(Range0ToRank) * BRICK_DIMS[Range0ToRank]...}
      , offsetIntoBrickStorage {0}
      {
        validateExtent(extent);
        // Build BrickStorage
        long step = NUM_ELEMENTS_PER_BRICK * (isComplex ? 2 : 1);
        BrickStorage storage = BrickStorage::allocate(layout.size(), step);
        // Store fields
        this->brickStorage = storage;
      }

      /**
       * Allocate a BrickedArray using mmap.
       *
       * @param layout the desired layout of bricks in memory
       * @param mmap_fd the file to mmap into, or nullptr if a new one should
       *                be created
       * @param offset Ignored if mmap_fd is null. Otherwise, the offset
       *               into mmap_fd that is used for allocation.
       */
      explicit BrickedArray(BrickLayout<RANK, LayoutPadding> layout,
                            void *mmap_fd,
                            size_t offset)
      : extent {layout.indexInStorage.getExtent(Range0ToRank) * BRICK_DIMS[Range0ToRank]...}
      , offsetIntoBrickStorage {0}
      {
        validateExtent(extent);
        // Build BrickStorage
        long step = NUM_ELEMENTS_PER_BRICK * (isComplex ? 2 : 1);
        BrickStorage storage;
        if(mmap_fd == nullptr) {
          storage = BrickStorage::mmap_alloc(layout.size(), step);
        } else {
          storage = BrickStorage::mmap_alloc(layout.size(), step, mmap_fd, offset);
        }
        // Store fields
        this->brickStorage = storage;
      }

    private:
      // Some validity checks
      static_assert(std::is_same<DataType, bElem>::value
                        || std::is_same<DataType, bComplexElem>::value,
                    "bElem and bComplexElem are the only data types supported");
      static_assert(sizeof...(VFold) <= sizeof...(BDim),
                    "Number of vector-folds must be at most the number of brick-dimensions");
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

      // extent of the array being bricked
      const ARRAY_TYPE(unsigned, RANK) extent;
      BrickStorage brickStorage; //< Where the data is actually stored
      const unsigned offsetIntoBrickStorage; //< location of first brick in storage (in units of real elements)

      /**
       * @param arrayExtent the extent to validate/compute number of bricks
       * @throw std::runtime_error if the arrayExtent is not valid
       */
      void validateExtent(std::array<unsigned, RANK> arrayExtent) const {
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
      }
  };

  // Define public static constants
  template<typename DataType, typename LayoutPadding, unsigned ... BDim, unsigned ... VFold, unsigned ... Range0ToRank>
  constexpr ARRAY_TYPE(unsigned, sizeof...(BDim))
  BrickedArray<DataType, Dim<BDim...>, Dim<VFold...>, LayoutPadding,
               brick::templateutils::ParameterPackManipulator<unsigned>::Pack<Range0ToRank...>
               >::BRICK_DIMS;
  template<typename DataType, typename LayoutPadding, unsigned ... BDim, unsigned ... VFold, unsigned ... Range0ToRank>
  constexpr ARRAY_TYPE(unsigned, sizeof...(BDim))
      BrickedArray<DataType, Dim<BDim...>, Dim<VFold...>, LayoutPadding,
                   brick::templateutils::ParameterPackManipulator<unsigned>::Pack<Range0ToRank...>
                   >::VECTOR_FOLDS;
}

#endif // BRICK_BRICKED_ARRAY_H
