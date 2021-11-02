//
// Created by Ben_Sepanski on 10/15/2021.
//

#ifndef BRICK_ARRAY_H
#define BRICK_ARRAY_H

#include <algorithm> // std::transform
#include <array> // std::array
#include <numeric> // std::plus

#include "template-utils.h"

/// Overloaded attributes for potentially GPU-usable functions (in place of __host__ __device__ etc.)
#if defined(__CUDACC__) || defined(__HIP__)
#define FORCUDA __host__ __device__
#else
#define FORCUDA
#endif
/// Inline if not in debug mode
#ifndef NDEBUG
#define INLINE
#else
#define INLINE inline
#endif
// Alignment
#ifndef ALIGN
#define ALIGN 2048
#endif

namespace brick {
  /**
   * An empty struct to specify padding
   * @tparam Pad the padding
   */
  template<unsigned ... Pad>
  struct Padding{};

  /**
   * Generic template for specialization
   * @tparam SizeType size type (should be able to represent largest number of elements in array)
   * @tparam IndexType type of indices
   * @see Array
   */
  template<typename DataType, unsigned Rank, typename Padding = Padding<>,
           typename SizeType = size_t, typename IndexType = unsigned,
           typename = typename brick::templateutils::UnsignedIndexSequence<Rank>::type>
  class Array;

  /**
   * A column-major array (index [i, j, k] of I x J x K array
   *                       accesses memory location k * IJ + j * I + i)
   *
   * @tparam DataType the type of data in the array
   * @tparam Rank the rank of the array
   * @tparam Padding padding on each dimension (rightmost is most-contiguous)
   * @tparam Range0ToRank AUTO-GENERATED TO BE 0,1,2,...,RANK-1: USER SHOULD NOT SUPPLY THESE
   * @see Array
   */
  template<typename DataType, unsigned Rank, typename SizeType, typename IndexType,
           unsigned ... PadInEachDim, unsigned ... Range0ToRank>
  class Array<DataType, Rank, Padding<PadInEachDim...>, SizeType, IndexType,
              brick::templateutils::ParameterPackManipulator<unsigned>::Pack<Range0ToRank...>> {
    static_assert(Rank == sizeof...(PadInEachDim) || sizeof...(PadInEachDim) == 0,
                  "Padding must be length 0, or of length Rank");
    // Make sure _Range0ToRank is correct
    static_assert(std::is_same<brick::templateutils::ParameterPackManipulator<unsigned>::Pack<Range0ToRank...>,
                               typename brick::templateutils::UnsignedIndexSequence<Rank>::type
                      >::value,
                  "User should not supply _Range0ToRank template argument");

    // used to manipulate unsigned templates
    using UnsignedPackManip = brick::templateutils::ParameterPackManipulator<unsigned>;

    // private members
    IF_CUDA_ELSE(DataType * const, const std::shared_ptr<DataType>) data;
    const ARRAY_TYPE(SizeType, Rank+1) stride;

    /**
     * Computes index. Use getIndex instead since it does extra checks
     * when in debug mode.
     *
     * @return index into a 0-rank array
     * @see getIndex
     */
    FORCUDA INLINE
    SizeType getIndexImpl() const {
      return 0;
    }

    /**
     * Computes index. Use getIndex instead since it does extra checks
     * when in debug mode.
     *
     * @param headIndex the index
     * @return the true index into the (possibly padded) rank-1 array
     * @see getIndex
     */
    FORCUDA INLINE
    SizeType getIndexImpl(IndexType headIndex) const {
      assert(headIndex < extent[RANK - 1]);
      return PADDING[RANK - 1] + headIndex;
    }

    /**
     * Computes index. Use getIndex instead since it does extra checks
     * when in debug mode.
     *
     * @tparam ConvertibleToIndexType a type convertible to IndexType
     * @param headIndex the first index
     * @param headOfTailIndices the second index
     * @param tailOfTailIndices remaining indices
     * @return the flat index into the array
     * @see getIndex
     */
    template<typename ... ConvertibleToIndexType>
    FORCUDA INLINE
    SizeType getIndexImpl(IndexType headIndex,
                          IndexType headOfTailIndices,
                          ConvertibleToIndexType... tailOfTailIndices) const {
      constexpr unsigned d = RANK - (sizeof...(tailOfTailIndices) + 2);
      assert(headIndex < extent[d]);

      SizeType tailOffset = getIndexImpl(headOfTailIndices, tailOfTailIndices...);
      return tailOffset * (extent[d] + 2 * PADDING[d]) +
             (PADDING[d] + headIndex);
    }

    /**
     * @tparam ConvertibleToIndexType Parameter pack
     *   convertible to IndexType and of size RANK
     * @param indices the indices into the array
     * @return the flat index into the array in memory
     */
    template<typename ... ConvertibleToIndexType>
    FORCUDA INLINE
    SizeType getIndex(ConvertibleToIndexType... indices) const {
      static_assert(sizeof...(ConvertibleToIndexType) == RANK,
                    "Number of indices must match Rank");
      static_assert(brick::templateutils::All<std::is_convertible<
                        ConvertibleToIndexType, IndexType>::value...>::value,
                    "Indices must be convertible to IndexType");
      SizeType index = getIndexImpl(indices...);
      assert(index < numElementsWithPadding);
      return index;
    }

    /**
     * @param arrExtent the extent
     * @return an array holding the stride in each dimension
     */
    static std::array<SizeType, Rank> computeStride(const std::array<IndexType, Rank> arrExtent) {
      SizeType strideInDim = 1;
      std::array<SizeType, Rank> stride;
      for(unsigned d = 0; d < Rank; ++d) {
        stride[d] = strideInDim;
        strideInDim *= arrExtent[d] + 2 * PADDING[d];
      }
      return stride;
    }

    /**
      * @return the number of padding elements on the edges of this array
    */
    FORCUDA INLINE
    SizeType computePaddingOnBoundary() {
      SizeType pad = 0;
      for(unsigned d = 0; d < Rank; ++d) {
        pad += PADDING[d] * stride[d];
      }
      return pad;
    }

  public:
    // public typedefs
    typedef Array<DataType, Rank, Padding<PadInEachDim...>, SizeType, IndexType,
                  brick::templateutils::ParameterPackManipulator<unsigned>::Pack<Range0ToRank...>>
        MyType;
    // public constexprs
    static constexpr unsigned RANK = Rank;
    static constexpr ARRAY_TYPE(unsigned, Rank)
        PADDING = UnsignedPackManip::padAndReverse<RANK, 0, PadInEachDim...>()
                  IF_CUDA_ELSE(.get(), );
    // public constants
    const ARRAY_TYPE(IndexType, Rank) extent;
    const SizeType numElements, numElementsWithPadding;

    /**
     * @param arrExtent the arrExtent of the array
     * @return the number of elements (excluding padding)
     */
    static SizeType computeNumElements(const std::array<IndexType, Rank> arrExtent) {
      return Array<DataType, Rank>::computeNumElementsWithPadding(arrExtent);
    }

    /**
     * @param arrExtent the arrExtent of the array
     * @return the number of elements (including padding)
     */
    static SizeType computeNumElementsWithPadding(const std::array<IndexType, Rank> arrExtent) {
      std::array<IndexType, Rank> extentWithPadding;
      SizeType numElements = 1;
      for(unsigned d = 0; d < RANK; ++d) {
        numElements *= arrExtent[d] + 2 * PADDING[d];
      }
      return numElements;
    }

    /**
     * Create an array with the given extent
     * @param arrExtent the extent in each dimension (0th entry is most contiguous)
     * @return the new Array object
     */
    explicit Array(const std::array<IndexType, RANK> arrExtent)
    : extent {arrExtent[Range0ToRank]...}
    , stride {computeStride(arrExtent)[Range0ToRank]..., computeNumElementsWithPadding(arrExtent)}
    , numElements{computeNumElements(arrExtent)}
    , numElementsWithPadding{computeNumElementsWithPadding(arrExtent)}
    , data{(DataType *)aligned_alloc(ALIGN, computeNumElementsWithPadding(arrExtent) * sizeof(DataType)), free}
    { }

    /**
     * Create an array using the provided data
     * @param arrExtent the extent of the array
     * @param data the data to use
     */
    explicit Array(const std::array<IndexType, RANK> arrExtent, std::shared_ptr<DataType> data)
    : extent {arrExtent[Range0ToRank]...}
    , stride {computeStride(arrExtent)[Range0ToRank]..., computeNumElementsWithPadding(arrExtent)}
    , data{data}
    , numElements{computeNumElements(arrExtent)}
    , numElementsWithPadding{computeNumElementsWithPadding(arrExtent)}
    { }

    /**
     * @param axis The axis (0 <= axis < Rank)
     * @return the extent in the provided axis
     */
    FORCUDA INLINE
    IndexType getExtent(unsigned axis) const {
      assert(axis >= 0);
      assert(axis < Rank);
      return extent[axis];
    }

    /**
     * @param indices the indices into the array
     * @param ...
     * @return a reference to the value at that index
     */
    template<typename ... I>
    FORCUDA INLINE
    DataType& operator()(I ... indices) {
      SizeType flatIndex = this->getIndex(indices...);
      return this->data.get()[flatIndex];
    }

    /**
     * @param indices the indices into the array
     * @param ...
     * @return a copy of the values at that index
     */
    template<typename ... I>
    FORCUDA INLINE
    DataType get(I ... indices) const {
      SizeType flatIndex = this->getIndex(indices...);
      return this->data.get()[flatIndex];
    }

    /**
     * Iterator
     * https://internalpointers.com/post/writing-custom-iterators-modern-cpp
     */
    class Iterator{
      private:
        SizeType flatIndex;
        MyType *array;

      public:
        using iterator_category = std::bidirectional_iterator_tag;
        using difference_type = void;
        using value_type = DataType;
        using pointer = DataType*;
        using reference = DataType&;

        FORCUDA INLINE
        explicit Iterator(MyType *arr)
        : array{arr}
        , flatIndex{arr->computePaddingOnBoundary()}
        { }

        FORCUDA INLINE
        explicit Iterator(MyType *arr, SizeType flatIndex)
        : array{arr}
        , flatIndex{flatIndex}
        { }

        FORCUDA INLINE
        Iterator(const Iterator &that)
        {
          *this = that;
        }

        FORCUDA INLINE
        Iterator& operator=(const Iterator &that) {
          if(this != &that) {
            this->array = that.array;
            this->flatIndex = that.flatIndex;
          }
          return *this;
        }

        FORCUDA INLINE
        reference operator*() const {
          assert(flatIndex < array->numElementsWithPadding);
          return array->data IF_CUDA_ELSE( , .get())[flatIndex];
        }

        FORCUDA INLINE
        pointer operator->() {
          assert(flatIndex < array->numElementsWithPadding);
          return array->data IF_CUDA_ELSE( , .get()) + flatIndex;
        }

        // Prefix increment
        FORCUDA INLINE
        Iterator& operator++() {
          flatIndex++;
          if(!brick::templateutils::All<(PADDING[Range0ToRank] == 0)...>::value) {
            #pragma unroll // So that can eliminate dead code
            for(unsigned d = 0; d < Rank; ++d) {
              if(PADDING[d] != 0) {
                IndexType idxInDim = (flatIndex / array->stride[d])
                                   % (array->getExtent(d) + 2 * PADDING[d]);
                assert(PADDING[d] <= idxInDim && idxInDim <= PADDING[d] + array->getExtent(d));
                if(idxInDim - PADDING[d] == array->getExtent(d)) {
                  flatIndex += 2 * PADDING[d] * array->stride[d];
                }
              }
            }
          }
          return *this;
        }

        // Postfix Increment
        FORCUDA INLINE
        Iterator operator++(int) {
          Iterator tmp = *this;
          ++(*this);
          return tmp;
        }

        // Prefix decrement
        FORCUDA INLINE
        Iterator& operator--() {
          flatIndex--;
          if(!brick::templateutils::All<(PADDING[Range0ToRank] == 0)...>::value) {
            #pragma unroll // So that can eliminate dead code
            for(unsigned d = 0; d < Rank; ++d) {
              if(PADDING[d] != 0) {
                IndexType idxInDim = (flatIndex / array->stride[d])
                                     % (array->getExtent(d) + 2 * PADDING[d]);
                assert(PADDING[d] - 1 <= idxInDim && idxInDim < PADDING[d] + array->getExtent(d));
                if(idxInDim == PADDING[d] - 1) {
                  flatIndex -= 2 * PADDING[d] * array->stride[d];
                }
              }
            }
          }
          return *this;
        }

        // Postfix Decrement
        FORCUDA INLINE
        Iterator operator--(int) {
          Iterator tmp = *this;
          --(*this);
          return tmp;
        }

        FORCUDA INLINE
        friend bool operator==(const Iterator& a, const Iterator &b) {
          return a.array == b.array && a.flatIndex == b.flatIndex;
        }

        FORCUDA INLINE
        friend bool operator!=(const Iterator& a, const Iterator &b) {
          return !operator==(a, b);
        }
    };

    /**
     * @return iterator to first element
     */
    FORCUDA INLINE
    Iterator begin() {return Iterator(this); }

    /**
     * @return iterator from element after-last
     */
    FORCUDA INLINE
    Iterator end() {
      // B/c the iterator++ skips all padding, the element "after" the last
      // element is the one after the padding after the last element
      return Iterator(this, numElementsWithPadding + computePaddingOnBoundary());
    }
  };

  // Define static members of array class
  template<typename DataType, unsigned Rank, typename SizeType, typename IndexType, unsigned ... PadInEachDim, unsigned ... Range0ToRank> constexpr
#ifdef __CUDACC__
  unsigned[Rank]
#else
  std::array<unsigned, Rank>
#endif
  Array<DataType, Rank, Padding<PadInEachDim...>, SizeType, IndexType,
        typename brick::templateutils::ParameterPackManipulator<unsigned>::Pack<Range0ToRank...> >::PADDING;
}

#endif // BRICK_ARRAY_H
