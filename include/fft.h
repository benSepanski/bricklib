/**
 * @file 
 * @brief hook for FFT computations on bricks
 */

#ifndef BRICK_FFT_H
#define BRICK_FFT_H

#include "brick.h"
#ifdef __CUDACC__
#include "cufft.h"
#endif

// useful template helpers
namespace // anonymous namespace
{
  /**
   * @brief value is true iff list contains query
   *  Generic struct for template specialization.
   * @tparam query the element checking for membership in the list
   * @tparam list
   */
  template<unsigned query, unsigned ... list> struct Contains;

  /**
   * @brief inductive case of contains check
   * 
   * @tparam head first element of the list
   * @tparam tail remainder of the list
   * @see Contains
   */
  template<unsigned query, unsigned head, unsigned ... tail> 
  struct Contains<query, head, tail...>
  {
    constexpr static bool value = (head == query) || Contains<query, tail...>::value;
  };
  /**
   * @brief base case of contains check--empty list
   * @see Contains
   */
  template<unsigned query>
  struct Contains<query> {constexpr static bool value = false;};

  /**
   * @brief Empty template representing a collection of unsigned values
   * @tparam vals the unsigned values
   */
  template<unsigned ... vals> struct UnsignedList;

  /**
   * @brief value is true iff all elements of the first list are contained in the second
   * Empty generic template for specialization
   * @tparam list1 the elements of the first set
   * @tparam list2 the elements of the second set
   */
  template<typename list1, typename list2> struct Subset;
  /**
   * @brief inductive case of subset check
   * @tparam head the first element of the first list
   * @tparam tail the remainder of the first list
   * @tparam superSet the elements of the second list
   * @see Subset
   */
  template<unsigned head, unsigned ... tail, unsigned ... superSet>
  struct Subset<UnsignedList<head, tail...>, UnsignedList<superSet...> >
  {
    constexpr static bool value = Contains<head, superSet...>::value
                                  && Subset<UnsignedList<tail...>, UnsignedList<superSet...> >::value;
  };
  /**
   * @brief base case of subset check--empty set is subset of any other set
   * @tparam superSet the elements in the second set
   * @see Subset
   */
  template<unsigned ... superSet>
  struct Subset<UnsignedList<>, UnsignedList<superSet...> >
  {
    constexpr static bool value = true;
  };

  /**
   * @brief Build an array at compile-time from an UnsignedList
   * 
   * @tparam list the unsigned list to build an array type
   * @see UnsignedList
   */
  template<typename list> struct StaticArrayBuilder;
  /**
   * @brief implementation of StaticArrayBuilder
   * @tparam vals the values to build an array from
   * @see StaticArrayBuilder
   */
  template<unsigned ... vals>
  struct StaticArrayBuilder<UnsignedList<vals...> >
  {
    typedef std::array<unsigned, sizeof...(vals)> type; ///< the type of the array
    constexpr static type value = { vals... };  ///< the array
  };

  /**
   * @brief Merge two lists of unsigned values
   * 
   *  Empty template for specialization
   * 
   * @tparam list1 the first list
   * @tparam list2 the second list
   */
  template<typename list1, typename list2> struct UnsignedListMerger;
  /**
   * @tparam vals1 the values of the first list
   * @tparam vals2 the values of the second list
   * @see UnsignedListMerger
   */
  template<unsigned ... vals1, unsigned ... vals2>
  struct UnsignedListMerger<UnsignedList<vals1...>, UnsignedList<vals2...> >
  {
    typedef UnsignedList<vals1..., vals2...> type;
  };

  /**
   * @brief Generates a list of unsigneds from start to end
   * 
   * @tparam Start the first element (inclusive)
   * @tparam End the last element (esclusive) (must have Start <= End)
   * @see UnsignedList
   */
  template<unsigned Start, unsigned End>
  struct Range
  {
    static_assert(Start <= End, "Start must come before End");
    typedef typename UnsignedListMerger<UnsignedList<Start>, typename Range<Start+1, End>::type>::type type; ///< type of the merged list
  };
  /**
   * @brief base case of Range
   * @tparam Start any unsigned value
   * @see Range
   */
  template<unsigned Start>
  struct Range<Start, Start>
  {
    typedef UnsignedList<> type; ///< empty list
  };

  /**
   * @brief remove any members of Elements which are in ToRemove
   * 
   * Generic template for specialization. public typedef type
   * is the list with designated elements removed.
   * 
   * @tparam Elements the original list
   * @tparam ToRemove elements to remove from Elements
   * @tparam Enable used for SFINAE
   */
  template<typename Elements, typename toRemove, typename Enable = void> struct UnsignedListRemover;
  /**
   * @brief Base case: Elements list is empty
   * @tparam vals2 the elements to remove from the empty list
   * @see UnsignedListRemover
   */
  template<unsigned ... vals2>
  struct UnsignedListRemover<UnsignedList<>, UnsignedList<vals2...>, void>
  {
    typedef UnsignedList<> type;
  };
  /**
   * @brief Inductive case: Elements list is non-empty, and head should be removed
   * @tparam vals1head the first member of the list
   * @tparam vals1tail the remainder of the list
   * @tparam vals2 the elements to remove from the list
   * @see UnsignedListRemover
   */
  template<unsigned vals1head, unsigned ... vals1tail, unsigned ... vals2>
  struct UnsignedListRemover<UnsignedList<vals1head, vals1tail...>, UnsignedList<vals2...>,
                             typename std::enable_if<Contains<vals1head, vals2...>::value>::type>
  {
    typedef typename UnsignedListRemover<UnsignedList<vals1tail...>, UnsignedList<vals2...> >::type type;
  };
  /**
   * @brief Inductive case: Elements list is non-empty, and head should not be removed
   * @tparam vals1head the first member of the list
   * @tparam vals1tail the remainder of the list
   * @tparam vals2 the elements to remove from the list
   * @see UnsignedListRemover
   */
  template<unsigned vals1head, unsigned ... vals1tail, unsigned ... vals2>
  struct UnsignedListRemover<UnsignedList<vals1head, vals1tail...>, UnsignedList<vals2...>,
                             typename std::enable_if<!Contains<vals1head, vals2...>::value>::type>
  {
    typedef typename UnsignedListRemover<UnsignedList<vals1tail...>, UnsignedList<vals2...> >::type tailType;
    typedef typename UnsignedListMerger<UnsignedList<vals1head>, tailType>::type type;
  };

  /**
   * @brief A range, with certain elements removed
   * Empty template for specialization
   * @tparam range  the range
   * @tparam intsToRemove  the values to remove from that range
   */
  template<typename range, unsigned ... intsToRemove> struct RangeSetMinus;
  /**
   * @tparam Start the start of the range (inclusive)
   * @tparam End the end of the range (exclusive)
   * @see RangeSetMinus
   */
  template<unsigned Start, unsigned End, unsigned ... intsToRemove>
  struct RangeSetMinus<Range<Start, End>, intsToRemove...>
  {
    typedef typename UnsignedListRemover<typename Range<Start, End>::type, UnsignedList<intsToRemove...> >::type listType;
    typedef typename StaticArrayBuilder<listType>::type arrayType;
    static constexpr arrayType value = StaticArrayBuilder<listType>::value;
  };

  /**
   * @brief base case of reduction, return init
   * @see reduction_impl
   */
  template<unsigned idx, typename F, typename T, unsigned N>
  constexpr std::enable_if<N <= idx, T> reduction_impl(F f, const T &init, const std::array<T, N> &arr)
  {
    return init;
  } 

  /**
   * @brief inductive case of reduction
   * 
   * @tparam idx current idx
   * @tparam F reduction function
   * @tparam T the type we are reducing over
   * @tparam N the size of the array
   */
  template<unsigned idx, typename F, typename T, unsigned N>
  constexpr std::enable_if<idx < N, T> reduction_impl(F f, const T &init, const std::array<T, N> &arr)
  {
    return reduction_impl<idx+1>(f, f(init, arr[idx]), arr);
  } 

  /**
   * @brief Perform a reduction over an array
   *
   * @tparam F the function type of the reduction
   * @tparam T the data type
   * @tparam N number of elements
   * @param f reduction function
   * @param init initializer of reduction
   * @param arr array reducting over
   * @return the result of the reduction
   */
  template<typename F, typename T, unsigned N>
  constexpr T reduction(F f, const T &init, const std::array<T, N> &arr)
  {
    return reduction_impl<0>(f, init, arr);
  } 
}  // end anonymous space

// cuFFT interface
#ifdef __CUDACC__

// https://developer.nvidia.com/blog/cuda-pro-tip-use-cufft-callbacks-custom-data-processing/
#ifndef __CUDACC_RDC__
#error THIS CODE REQUIRES CUDA RELOCATABLE DEVICE CODE COMPILATION
#endif

/**
 * @brief Interface from bricks into cuFFT
 * 
 * Empty template for specialization
 * 
 * @tparam BrickType the type of the bricks
 * @tparam FFTDims the collection of dimensions to take a fourier transfor over
 */
template<typename BrickType, typename FFTDims> class BricksCuFFTPlan;

/**
 * @tparam FFTDims the dimensions to take a fourier transfor over (must be at least 1, at most 3)
 * @see BricksCuFFTPlan
 * @see Brick
 */
template<bool isComplex,
         typename CommunicatingDims,
         unsigned ... BDims,
         unsigned ... Fold,
         unsigned ... FFTDims
         >
class BricksCuFFTPlan<Brick<Dim<BDims...>, Dim<Fold...>, isComplex, CommunicatingDims>, Dim<FFTDims...> >
{
  static constexpr unsigned FFTRank = sizeof...(FFTDims);

  static_assert(FFTRank > 0, "Expected positive FFTRank");
  static_assert(FFTRank <= sizeof...(BDims), "FFTRank cannot be bigger than number of dimensions");
  static_assert(FFTRank <= 3, "FFT Dim > 3 not implemented in cuFFT");
  static_assert(Subset<UnsignedList<FFTDims...>, typename Range<0, sizeof...(BDims)>::type>::value,
                "FFTDims must be in range 0...num dims");
  
  private:
    // typedefs
    typedef Brick<Dim<BDims...>, Dim<Fold...>, isComplex, CommunicatingDims> BrickType;
    typedef typename BrickType::elemType elemType;
    // constexprs
    static constexpr std::array<unsigned int, sizeof...(BDims) - FFTRank> 
      nonFourierDims = RangeSetMinus<Range<0, sizeof...(BDims)>, FFTDims...>::value;
    static constexpr std::array<unsigned int, FFTRank> fourierDims = { FFTDims... };
    static constexpr unsigned NON_FOURIER_BRICK_SIZE = reduction(&std::multiplies<unsigned>::operator(), 1, nonFourierDims);
    static constexpr unsigned FOURIER_BRICK_SIZE = reduction(&std::multiplies<unsigned>::operator(), 1, fourierDims);
    // attributes
    cufftHandle plan;

    // // TODO: Docs
    struct BricksCuFFTInfo
    {
      size_t batchSize; ///< number of fourier plans
      size_t nonFourierGridExtent[sizeof...(BDims) - FFTRank];
      size_t nonFourierGridStride[sizeof...(BDims) - FFTRank];
      size_t fourierGridExtent[FFTRank];
      size_t fourierGridStride[FFTRank];
      unsigned *grid_ptr; ///< pointer to brick-grid on device
      BrickType *inBrick; ///< pointer to input brick
    };

    // TODO: Docs
    template<unsigned idx, unsigned nonFourierIdx, unsigned fourierIdx>
    __device__ static
    std::enable_if<idx >= sizeof...(BDims), elemType>
    accessFromFlatIndexes(const elemType &brickElt, unsigned flatNonFourierIndex, unsigned flatFourierIndex) { return brickElt;}

    // TODO: Docs
    template<unsigned idx, unsigned nonFourierIdx, unsigned fourierIdx, typename T>
    __device__ static
    std::enable_if<idx < sizeof...(BDims) && idx == nonFourierDims[nonFourierIdx], elemType>
    accessFromFlatIndexes(const T &brickAccessor, unsigned flatNonFourierIndex, unsigned flatFourierIndex)
    {
      constexpr unsigned NEXT_NON_FOURIER_DIM = std::min((unsigned) sizeof...(BDims) - FFTRank - 1, fourierIdx + 1); ///< to avoid out-of-bounds during compilation
      constexpr unsigned BRICK_STRIDE = Dim<BDims...>::template product<idx>();
      unsigned indexInDim = flatNonFourierIndex / BRICK_STRIDE;
      unsigned newFlatNonFourierIndex = flatNonFourierIndex % BRICK_STRIDE;
      return accessFromFlatIndexes<idx+1, NEXT_NON_FOURIER_DIM, fourierIdx>(brickAccessor[indexInDim], flatNonFourierIndex, flatFourierIndex);
    }

    // TODO: Docs
    template<unsigned idx, unsigned nonFourierIdx, unsigned fourierIdx, typename T>
    __device__ static
    std::enable_if<idx < sizeof...(BDims) && idx == fourierDims[fourierIdx], elemType>
    accessFromFlatIndexes(const T &brickAccessor, unsigned flatNonFourierIndex, unsigned flatFourierIndex)
    {
      constexpr unsigned NEXT_FOURIER_DIM = std::min((unsigned) FFTRank - 1, fourierIdx + 1); ///< to avoid out-of-bounds during compilation
      constexpr unsigned BRICK_STRIDE = Dim<BDims...>::template product<idx>();
      unsigned indexInDim = flatNonFourierIndex / BRICK_STRIDE;
      unsigned newFlatNonFourierIndex = flatNonFourierIndex % BRICK_STRIDE;
      return accessFromFlatIndexes<idx+1, nonFourierIdx, NEXT_FOURIER_DIM>(brickAccessor[indexInDim], flatNonFourierIndex, flatFourierIndex);
    }

  public:
    using cuElemType = std::conditional<BrickType::complex, bCuComplexElem, bElem>;

    BricksCuFFTPlan(std::array<size_t, sizeof...(BDims)> grid_size, cufftType type)
    {
      // get logical strides of data
      int stride[FFTRank];
      stride[0] = 1;
      for(unsigned i = 0; i + 1 < FFTRank; ++i)
      {
        stride[i + 1] = grid_size[fourierDims[i]] * stride[i];
      }
      // figure out logical distance between batches
      size_t distBetweenBatches = stride[FFTRank - 1] * grid_size[fourierDims[FFTRank - 1]];
      // compute number of batches
      size_t numBatches = 1;
      for(unsigned i = 0; i < sizeof...(BDims) - FFTRank; ++i)
      {
        numBatches *= nonFourierDims[i];
      }
      // set up cuda plan
      cufftPlanMany(&this->plan, FFTRank, stride,
                    nullptr, -1, distBetweenBatches,
                    nullptr, -1, distBetweenBatches,
                    type, numBatches);
    }

    ~BricksCuFFTPlan()
    {
      cufftDestroy(plan);
    }

    // https://docs.nvidia.com/cuda/cufft/index.html#callback-routines
    __device__ cuElemType bricksLoadCallback(void *dataIn,
                                             size_t offset,
                                             void *callerInfo,
                                             void *sharedPtr)
    {
      // get info
      BricksCuFFTInfo *cuFFTInfo = reinterpret_cast<BricksCuFFTInfo*>(callerInfo);

      // Figure out flat indices (separated by fourier and non-fourier dimensions)
      int batchIndex = offset / cuFFTInfo->batchSize;
      int batchIndexInGrid = batchIndex / NON_FOURIER_BRICK_SIZE;
      int batchIndexInBrick = batchIndex % NON_FOURIER_BRICK_SIZE;

      int fourierIndex = offset % cuFFTInfo->batchSize;
      int fourierIndexInGrid = fourierIndex / FOURIER_BRICK_SIZE;
      int fourierIndexInBrick = fourierIndex % FOURIER_BRICK_SIZE;

      // get index into grid
      unsigned gridIndex = 0;
      for(unsigned idx = 0; idx < sizeof...(BDims) - FFTRank; ++idx)
      {
        gridIndex += (batchIndexInGrid % cuFFTInfo->nonFourierGridExtent[idx]) * cuFFTInfo->nonFourierGridStride[idx];
        batchIndexInGrid /= (cuFFTInfo->nonFourierGridExtent[idx]);
      }
      for(unsigned idx = 0; idx < FFTRank; ++idx)
      {
        gridIndex += (fourierIndexInGrid % cuFFTInfo->fourierGridExtent[idx]) * cuFFTInfo->fourierGridStride[idx];
        fourierIndexInGrid /= (cuFFTInfo->fourierGridExtent[idx]);
      }

      // get element 
      unsigned indexOfBrick = cuFFTInfo->grid_ptr[gridIndex];
      BrickType brick = cuFFTInfo->inBrick;
      return (cuElemType) accessFromFlatIndexes<0,0,0>(brick[indexOfBrick], batchIndexInBrick, fourierIndexInBrick);
    }
};

#endif // defined(__CUDACC__)

#endif // BRICK_FFT_H