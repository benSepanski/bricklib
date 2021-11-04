//
// Created by Ben_Sepanski on 10/15/2021.
//

#ifndef BRICK_ARRAY_H
#define BRICK_ARRAY_H

#include <algorithm> // std::transform
#include <array> // std::array
#include <iterator>
#include <memory> // std::shared_ptr
#include <numeric> // std::plus

#include "template-utils.h"

/// Overloaded attributes for potentially GPU-usable functions (in place of __host__ __device__ etc.)
#if defined(__CUDACC__) || defined(__HIP__)
#define FORCUDA __host__ __device__
#include "brick-cuda.h"
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
              brick::templateutils::ParameterPackManipulator<unsigned>::Pack<Range0ToRank...> > {
    static_assert(Rank == sizeof...(PadInEachDim) || sizeof...(PadInEachDim) == 0,
                  "Padding must be length 0, or of length Rank");
    // Make sure _Range0ToRank is correct
    static_assert(std::is_same<brick::templateutils::ParameterPackManipulator<unsigned>::Pack<Range0ToRank...>,
                               typename brick::templateutils::UnsignedIndexSequence<Rank>::type
                      >::value,
                  "User should not supply _Range0ToRank template argument");

    /// typedefs/static constants
    private:
      // private typedefs
      // used to manipulate unsigned templates
      using UnsignedPackManip = brick::templateutils::ParameterPackManipulator<unsigned>;
      template<unsigned ... PackValues>
      using UnsignedPack = UnsignedPackManip::Pack<PackValues...>;
    public:
      // public constexprs
      static constexpr unsigned RANK = Rank;
      /**
       * Return the padding in the specified axis
       * @param axis
       * @return the padding
       */
      static constexpr unsigned PADDING(unsigned axis) {
          typedef UnsignedPack<PadInEachDim...> PaddingPack;
          return UnsignedPackManip::getOrDefault<PaddingPack, 0>(
                           sizeof...(PadInEachDim) - 1 - axis
                           );
      };

    /// static functions
    private:
      // private static functions
      /**
       * @param arrExtent the extent
       * @return an array holding the stride in each dimension
       */
      static std::array<SizeType, Rank> computeStride(const std::array<IndexType, Rank> arrExtent) {
        SizeType strideInDim = 1;
        std::array<SizeType, Rank> stride;
        #pragma unroll
        for(unsigned d = 0; d < Rank; ++d) {
          stride[d] = strideInDim;
          strideInDim *= arrExtent[d] + 2 * PADDING(d);
        }
        return stride;
      }
    public:
      // public static functions
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
        SizeType numElements = 1;
        for(unsigned d = 0; d < RANK; ++d) {
          numElements *= arrExtent[d] + 2 * PADDING(d);
        }
        return numElements;
      }

    /// Member attributes
    private:
      // private members
      const std::shared_ptr<DataType> sharedDataPtr;
      DataType * const dataPtr = sharedDataPtr.get();
      const SizeType stride[Rank+1];
    public:
      // public members
      const IndexType extent[Rank];
      const SizeType numElements = computeNumElements({extent[Range0ToRank]...}),
                     numElementsWithPadding = computeNumElementsWithPadding({extent[Range0ToRank]...});

    /// Member methods
    private:
      // private member methods
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
        return PADDING(RANK - 1) + headIndex;
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
        return tailOffset * (extent[d] + 2 * PADDING(d)) +
               (PADDING(d) + headIndex);
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
        * @return the number of padding elements on the edges of this array
      */
      FORCUDA INLINE
      SizeType computePaddingOnBoundary() const {
        SizeType pad = 0;
        #pragma unroll
        for(unsigned d = 0; d < Rank; ++d) {
          pad += PADDING(d) * stride[d];
        }
        return pad;
      }

    public:
      // public member methods
      /**
       * Create an array with the given extent
       * @param arrExtent the extent in each dimension (0th entry is most contiguous)
       * @return the new Array object
       */
      explicit Array(const std::array<IndexType, RANK> arrExtent)
      : sharedDataPtr{(DataType *)aligned_alloc(ALIGN, computeNumElementsWithPadding(arrExtent) * sizeof(DataType)), free}
      , stride {computeStride(arrExtent)[Range0ToRank]..., computeNumElementsWithPadding(arrExtent)}
      , extent {arrExtent[Range0ToRank]...}
      { }

      /**
       * Create an array using the provided data
       * @param arrExtent the extent of the array
       * @param data the data to use
       */
      explicit Array(const std::array<IndexType, RANK> arrExtent, std::shared_ptr<DataType> data)
      : sharedDataPtr{data}
      , stride {computeStride(arrExtent)[Range0ToRank]..., computeNumElementsWithPadding(arrExtent)}
      , extent {arrExtent[Range0ToRank]...}
      { }

      /**
       * Copy constructor
       * @param that the array to make a shallow copy of
       */
      Array(const Array &that)
      : sharedDataPtr{that.sharedDataPtr}
      , dataPtr{that.dataPtr}
      , stride {that.stride[Range0ToRank]...}
      , extent {that.extent[Range0ToRank]...}
      , numElements{that.numElements}
      , numElementsWithPadding{that.numElementsWithPadding}
      { }

      /**
       * @return A pointer to the start of the array
       */
      std::shared_ptr<DataType> getData() const {
        return sharedDataPtr;
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
        return this->dataPtr[flatIndex];
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
        return this->dataPtr[flatIndex];
      }

#ifdef __CUDACC__
      /**
       * Build an array object with data on the device.
       * This array and its members are stored on the host, but
       * the data pointer is a device pointer.
       *
       * @return The array object
       */
      // TODO: IMPLEMENT ZERO-COPY ALLOCATION
      Array copyToDevice() const {
        size_t dataSize = numElementsWithPadding * sizeof(DataType);
        DataType *dataPtr_dev;
        cudaCheck(cudaMalloc(&dataPtr_dev, dataSize));
        cudaCheck(cudaMemcpy(dataPtr_dev, dataPtr, dataSize, cudaMemcpyHostToDevice));
        std::shared_ptr<DataType> sharedDataPtr_dev(
            dataPtr_dev,
            [](DataType *p) { cudaCheck(cudaFree(p)); }
        );
        return Array({extent[Range0ToRank]...}, sharedDataPtr_dev);
      }

      /**
       * Copy data from that array into this one
       *
       * @param that_dev the array on the device
       */
      void copyFromDevice(const Array &that_dev) {
        size_t dataSize = numElementsWithPadding * sizeof(DataType);
        cudaCheck(cudaMemcpy(dataPtr, that_dev.dataPtr, dataSize, cudaMemcpyDeviceToHost));
      }
#endif

      /**
       * Iterator
       * https://internalpointers.com/post/writing-custom-iterators-modern-cpp
       */
      template<typename T>
      class Iterator {
        /// public typedefs
        public:
          typedef typename std::conditional<std::is_const<T>::value,
                                            const Array,
                                            Array>::type ArrayType;
        /// Attributes and methods
        private:
          SizeType flatIndex;
          ArrayType *array;

        public:
          using iterator_category IF_CUDA_ELSE( ,__attribute__((unused))) = std::bidirectional_iterator_tag;
          using difference_type IF_CUDA_ELSE( , __attribute__((unused))) = void;
          using value_type = T;
          using pointer = T*;
          using reference = T&;

          FORCUDA INLINE
          explicit Iterator(ArrayType *arr)
          : array{arr}
          , flatIndex{arr->computePaddingOnBoundary()}
          { }

          FORCUDA INLINE
          explicit Iterator(ArrayType *arr, SizeType flatIndex)
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
            return array->dataPtr[flatIndex];
          }

          FORCUDA INLINE
          pointer operator->() {
            assert(flatIndex < array->numElementsWithPadding);
            return array->dataPtr + flatIndex;
          }

          // Prefix increment
          FORCUDA INLINE
          Iterator& operator++() {
            flatIndex++;
            if(!brick::templateutils::All<(PADDING(Range0ToRank) == 0)...>::value) {
              #pragma unroll // So that can eliminate dead code
              for(unsigned d = 0; d < Rank; ++d) {
                if(PADDING(d) != 0) {
                  IndexType idxInDim = (flatIndex / array->stride[d])
                                     % (array->extent[d] + 2 * PADDING(d));
                  assert(PADDING(d) <= idxInDim && idxInDim <= PADDING(d) + array->extent[d]);
                  if(idxInDim - PADDING(d) == array->extent[d]) {
                    flatIndex += 2 * PADDING(d) * array->stride[d];
                  }
                }
              }
            }
            return *this;
          }

          // Postfix Increment
          FORCUDA INLINE
          Iterator operator++(int) { // NOLINT(cert-dcl21-cpp)
            const Iterator tmp = *this;
            ++(*this);
            return tmp;
          }

          // Prefix decrement
          FORCUDA INLINE
          Iterator& operator--() {
            flatIndex--;
            if(!brick::templateutils::All<(PADDING(Range0ToRank) == 0)...>::value) {
              #pragma unroll // So that can eliminate dead code
              for(unsigned d = 0; d < Rank; ++d) {
                if(PADDING(d) != 0) {
                  IndexType idxInDim = (flatIndex / array->stride[d])
                                       % (array->extent[d] + 2 * PADDING(d));
                  assert(PADDING(d) - 1 <= idxInDim && idxInDim < PADDING(d) + array->extent[d]);
                  if(idxInDim == PADDING(d) - 1) {
                    flatIndex -= 2 * PADDING(d) * array->stride[d];
                  }
                }
              }
            }
            return *this;
          }

          // Postfix Decrement
          FORCUDA INLINE
          Iterator operator--(int) { // NOLINT(cert-dcl21-cpp)
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

      typedef Iterator<DataType> iterator_type;
      typedef Iterator<const DataType> const_iterator_type;

      /**
       * @return iterator to first element
       */
      FORCUDA INLINE
      iterator_type begin() {return iterator_type(this); }

      /**
       * @return iterator from element after-last
       */
      FORCUDA INLINE
      iterator_type end() {
        // B/c the iterator++ skips all padding, the element "after" the last
        // element is the one after the padding after the last element
        return iterator_type(this, numElementsWithPadding + computePaddingOnBoundary());
      }

      /**
       * @return const iterator to first element
       */
      FORCUDA INLINE
      const_iterator_type cbegin() {return const_iterator_type(this); }

      /**
       * @return const iterator from element after-last
       */
      FORCUDA INLINE
      const_iterator_type cend() {
        // B/c the iterator++ skips all padding, the element "after" the last
        // element is the one after the padding after the last element
        return const_iterator_type(this, numElementsWithPadding + computePaddingOnBoundary());
      }

      /**
       * Deep copy that array into this one
       * @tparam ThatSizeType size-type of that array
       * @tparam ThatIndexType index-type of that array
       * @tparam ThatArrayPadding padding of that array
       * @param that the array to load from
       */
      template<typename ThatSizeType, typename ThatIndexType, unsigned ... ThatArrayPadding>
      void loadFrom(const Array<DataType, Rank, Padding<ThatArrayPadding...>, ThatSizeType, ThatIndexType> &that) {
        // Ensure extents match
        for(unsigned d = 0; d < RANK; ++d) {
          if(that.extent[d] != this->extent[d]) {
            throw std::runtime_error("Array extent does not match this object");
          }
        }
        auto *thatPtr = &that; // Avoid omp synchronization
        const unsigned boundaryPadding = computePaddingOnBoundary();
        // Perform load
        #pragma omp parallel for firstprivate(thatPtr, boundaryPadding) default(none)
        for(unsigned outerIndex = 0; outerIndex < this->extent[RANK - 1]; ++outerIndex) {
          unsigned thisFlatIndex = boundaryPadding + outerIndex * stride[RANK - 1];
          unsigned thatFlatIndex = boundaryPadding + outerIndex * thatPtr->stride[RANK - 1];
          iterator_type this_iterator(this, thisFlatIndex);
          const_iterator_type that_iterator(thatPtr, thatFlatIndex);
          for(unsigned i = 0; i < numElements / this->extent[RANK - 1]; ++i) {
            *this_iterator = *that_iterator;
            this_iterator++;
            that_iterator++;
          }
        }
      }

      /**
       * Deep copy this array into that one
       * @tparam ThatSizeType size-type of that array
       * @tparam ThatIndexType index-type of that array
       * @tparam ThatArrayPadding padding of that array
       * @param that the array to load from
       */
      template<typename ThatSizeType, typename ThatIndexType, unsigned ... ThatArrayPadding>
      void storeTo(Array<DataType, Rank, Padding<ThatArrayPadding...>, ThatSizeType, ThatIndexType> &that) const {
        // Ensure extents match
        for(unsigned d = 0; d < RANK; ++d) {
          if(that.extent[d] != this->extent[d]) {
            throw std::runtime_error("Array extent does not match this object");
          }
        }
        auto *thatPtr = &that; // Avoid omp synchronization
        const unsigned boundaryPadding = computePaddingOnBoundary();
    // Perform load
  #pragma omp parallel for firstprivate(thatPtr, boundaryPadding) default(none)
        for(unsigned outerIndex = 0; outerIndex < this->extent[RANK - 1]; ++outerIndex) {
          unsigned thisFlatIndex = boundaryPadding + outerIndex * stride[RANK - 1];
          unsigned thatFlatIndex = boundaryPadding + outerIndex * thatPtr->stride[RANK - 1];
          const_iterator_type this_iterator(this, thisFlatIndex);
          iterator_type that_iterator(thatPtr, thatFlatIndex);
          for(unsigned i = 0; i < numElements / this->extent[RANK - 1]; ++i) {
            *that_iterator = *this_iterator;
            this_iterator++;
            that_iterator++;
          }
        }
      }
  };
}

#endif // BRICK_ARRAY_H
