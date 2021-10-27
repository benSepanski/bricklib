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
   * @see Array
   */
  template<typename DataType, unsigned Rank, typename Padding = Padding<>>
  class Array;

  /**
   * A column-major array (index [i, j, k] of I x J x K array
   *                       accesses memory location k * IJ + j * I + i)
   *
   * @tparam DataType the type of data in the array
   * @tparam Rank the rank of the array
   * @tparam Padding padding on each dimension (rightmost is most-contiguous)
   * @see Array
   */
  template<typename DataType, unsigned Rank, unsigned ... PadInEachDim>
  class Array<DataType, Rank, Padding<PadInEachDim...> > {
    static_assert(Rank == sizeof...(PadInEachDim) || sizeof...(PadInEachDim) == 0,
                  "Padding must be length 0, or of length Rank");

    // used to manipulate unsigned templates
    using UnsignedPackManip = brick::templateutils::ParameterPackManipulator<unsigned>;

    static constexpr unsigned RANK = Rank;
    DataType *data = nullptr;
    bool freeDataOnDelete;
#ifdef __CUDACC__
    unsigned extent[RANK];
#else
    std::array<unsigned, RANK> extent;
#endif

    /**
     * Computes index. Use getIndex instead since it does extra checks
     * when in debug mode.
     *
     * @return index into a 0-rank array
     * @see getIndex
     */
    FORCUDA INLINE
    size_t getIndexImpl() const {
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
    size_t getIndexImpl(unsigned headIndex) const {
      assert(headIndex < extent[RANK - 1]);
      return PADDING[RANK - 1] + headIndex;
    }

    /**
     * Computes index. Use getIndex instead since it does extra checks
     * when in debug mode.
     *
     * @tparam ConvertibleToUnsigned a type convertible to unsigned
     * @param headIndex the first index
     * @param headOfTailIndices the second index
     * @param tailOfTailIndices remaining indices
     * @return the flat index into the array
     * @see getIndex
     */
    template<typename ... ConvertibleToUnsigned>
    FORCUDA INLINE
    size_t getIndexImpl(unsigned headIndex,
                        unsigned headOfTailIndices,
                        ConvertibleToUnsigned ... tailOfTailIndices) const {
      constexpr unsigned d = RANK - (sizeof...(tailOfTailIndices) + 2);
      assert(headIndex < extent[d]);

      size_t tailOffset = getIndexImpl(headOfTailIndices, tailOfTailIndices...);
      return tailOffset * (extent[d] + 2 * PADDING[d]) +
             (PADDING[d] + headIndex);
    }

    /**
     * @tparam ConvertibleToUnsigned Parameter pack
     *   convertible to unsigned and of size RANK
     * @param indices the indices into the array
     * @return the flat index into the array in memory
     */
    template<typename ... ConvertibleToUnsigned>
    FORCUDA INLINE
    size_t getIndex(ConvertibleToUnsigned ... indices) const {
      static_assert(sizeof...(ConvertibleToUnsigned) == RANK,
                    "Number of indices must match Rank");
      static_assert(brick::templateutils::All<std::is_convertible<ConvertibleToUnsigned, unsigned>::value...>::value,
                    "Indices must be convertible to unsigned");
      size_t index = getIndexImpl(indices...);
      size_t n = numElementsWithPadding();
      assert(index < n);
      return index;
    }

  public:
    static constexpr
#ifdef __CUDACC__
        unsigned PADDING[RANK] =
#else
        std::array<unsigned, RANK> PADDING =
#endif
            UnsignedPackManip::PackToArray<
                typename UnsignedPackManip::PackReverser<
                    typename UnsignedPackManip::PackPadder<RANK, 0, UnsignedPackManip::Pack<PadInEachDim...> >::type
                    >::type
                >::value
#ifdef __CUDACC__
                .get()
#endif
        ;

    /**
     * @param arrExtent the arrExtent of the array
     * @return the number of elements (excluding padding)
     */
    FORCUDA INLINE
    static size_t numElements(const std::array<unsigned, Rank> arrExtent) {
      return Array<DataType, Rank>::numElementsWithPadding(arrExtent);
    }

    /**
     * @param arrExtent the arrExtent of the array
     * @return the number of elements (including padding)
     */
    FORCUDA INLINE
    static size_t numElementsWithPadding(const std::array<unsigned, Rank> arrExtent) {
#ifdef __CUDACC__
      unsigned[RANK]
#else
      std::array<unsigned, RANK>
#endif
          extentWithPadding;
      size_t numElements = 1;
      for(unsigned d = 0; d < RANK; ++d) {
        numElements *= arrExtent[d] + 2 * PADDING[d];
      }
      return numElements;
    }

    /**
     * @return the number of elements in the array
     */
    FORCUDA INLINE
    size_t numElements() const {
      return numElements(extent);
    }

    /**
     * @return the number of elements (including padding) in the array
     */
    FORCUDA INLINE
    size_t numElementsWithPadding() const {
      return numElementsWithPadding(extent);
    }

    /**
     * Create an array with the given extent
     * @param arrExtent the extent in each dimension (0th entry is most contiguous)
     * @return the new Array object
     */
    explicit Array(const std::array<unsigned, RANK> arrExtent) {
      size_t n = numElementsWithPadding(arrExtent);
      this->data = (DataType*) aligned_alloc(ALIGN, n * sizeof(DataType));
      this->freeDataOnDelete = true;
      std::copy(arrExtent.begin(), arrExtent.end(), this->extent.begin());
    }

    /**
     * Create an array using the provided data
     * @param arrExtent the extent of the array
     * @param data the data to use
     */
    explicit Array(const std::array<unsigned, RANK> arrExtent, DataType* data) {
      this->data = data;
      this->freeDataOnDelete = false;
      std::copy(arrExtent.begin(), arrExtent.end(), this->extent.begin());
    }

    /**
     * Destructor, frees data if we're the one that allocated it
     */
    ~Array() {
      if(freeDataOnDelete) {
        free(this->data);
      }
    }

    /**
     * @param indices the indices into the array
     * @param ...
     * @return a reference to the value at that index
     */
    template<typename ... IndexType>
    FORCUDA INLINE
    DataType& operator()(IndexType ... indices) {
      size_t flatIndex = this->getIndex(indices...);
      return this->data[flatIndex];
    }

    /**
     * @param indices the indices into the array
     * @param ...
     * @return a copy of the values at that index
     */
    template<typename ... IndexType>
    FORCUDA INLINE
    DataType get(IndexType ... indices) const {
      size_t flatIndex = this->getIndex(indices...);
      return this->data[flatIndex];
    }
  };

  // Define static members of array class
  template<typename DataType, unsigned Rank, unsigned ... PadInEachDim> constexpr
#ifdef __CUDACC__
  unsigned[Array<DataType, Rank, Padding<PadInEachDim...>>::RANK]
#else
  std::array<unsigned, Array<DataType, Rank, Padding<PadInEachDim...>>::RANK>
#endif
  Array<DataType, Rank, Padding<PadInEachDim...> >::PADDING;
}

#endif // BRICK_ARRAY_H
