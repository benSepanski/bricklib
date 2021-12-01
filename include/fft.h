/**
 * @file 
 * @brief hook for FFT computations on bricks
 */

#ifndef BRICK_FFT_H
#define BRICK_FFT_H

#include <array>
#include <functional>
#include "brick.h"
#ifdef __CUDACC__
#include "brick-cuda.h"
#include "cufftXt.h"
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
   * @tparam indices void if entire list should be used, otherwise an UnsignedList of indices
   * @see UnsignedList
   */
  template<typename list, typename indices = void> struct StaticArrayBuilder;
  /**
   * @brief implementation of StaticArrayBuilder
   * @tparam vals the values to build an array from
   * @see StaticArrayBuilder
   */
  template<unsigned ... vals>
  struct StaticArrayBuilder<UnsignedList<vals...>, void>
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
   * @brief used to reverse an unsigned list
   * @tparam list the list to reverse
   * @see UnsignedList
   */
  template<typename list>
  struct UnsignedListReverser;
  /**
   * @brief inductive case
   * @tparam head first entry of un-reversed list
   * @tparam tail remainder of un-reversed list
   * @see UnsignedListReverser
   */
  template<unsigned head, unsigned ... tail> struct UnsignedListReverser<UnsignedList<head, tail...> >
  {
    typedef typename UnsignedListReverser<UnsignedList<tail...> >::type reversedTailType;
    typedef typename UnsignedListMerger<reversedTailType, UnsignedList<head> >::type type;
  };
  /**
   * @brief base case
   * @see UnsignedListReverser
   */
  template<> struct UnsignedListReverser<UnsignedList<> > {typedef UnsignedList<> type;};

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
  template<unsigned idx, typename F, typename L, typename R, size_t N>
  constexpr typename std::enable_if<N <= idx, L>::type
  reduction_impl(F f, const L &init, const std::array<R, N> &arr)
  {
    return init;
  } 

  /**
   * @brief inductive case of reduction
   * 
   * @tparam idx current idx
   * @tparam F the function type
   * @tparam L type of the left argument
   * @tparam R type of the right argument
   * @tparam N the size of the array
   */
  template<unsigned idx, typename F, typename L, typename R, size_t N>
  constexpr typename std::enable_if<idx < N, L>::type 
  reduction_impl(F f, const L &init, const std::array<R, N> &arr)
  {
    const L newInit = f(init, arr[idx]);
    return reduction_impl<idx+1>(f, newInit, arr);
  } 

  /**
   * @brief Perform a reduction over an array
   *
   * @tparam F the function type
   * @tparam L the left argument type
   * @tparam R the right argument type 
   * @tparam N number of elements
   * @param f reduction function
   * @param init initializer of reduction
   * @param arr array reducting over
   * @return the result of the reduction
   */
  template<typename F, typename L, typename R, size_t N>
  constexpr L reduction(F f, const L &init, const std::array<R, N> &arr)
  {
    return reduction_impl<0>(f, init, arr);
  } 

  /**
   * @brief statically check if arr is sorted (with no duplicates)
   * copied from https://stackoverflow.com/questions/59613339/how-to-can-a-stdarray-class-member-be-statically-asserted-to-be-sorted-in-c1
   */
  template <typename T, std::size_t N>
  constexpr bool is_sorted(std::array<T, N> const& arr, std::size_t from) {
      return N - from == 0 or (arr[from - 1] < arr[from] and is_sorted(arr, from + 1));
  }

  /**
   * @brief statically check if arr is sorted (with no duplicates)
   * copied from https://stackoverflow.com/questions/59613339/how-to-can-a-stdarray-class-member-be-statically-asserted-to-be-sorted-in-c1
   */
  template <typename T, std::size_t N>
  constexpr bool is_sorted(std::array<T, N> const& arr) {
      return N == 0 or is_sorted(arr, 1);
  }
}  // end anonymous space

/**
 * @brief the input/output types of the fourier transform
 */
enum FourierDataType { ComplexToComplex, RealToComplex, SymmetricComplexToReal };

/**
 * @brief empty template to specify fourier dimensions
 * @tparam DataType input/output types of the fourier transform
 * @tparam dims the dimensions taking a fourier transform. Must be in descending order
 */
template<FourierDataType DataType, unsigned ... dims>
struct FourierType;

// cufft interface
#ifdef __CUDACC__

// https://developer.nvidia.com/blog/cuda-pro-tip-use-cufft-callbacks-custom-data-processing/
#ifndef __CUDACC_RDC__
#error THIS CODE REQUIRES CUDA RELOCATABLE DEVICE CODE COMPILATION
#endif

/**
 * @brief Check the return of CUDA FFT calls, do nothing during release build
 */
#ifdef NDEBUG
#define cufftCheck(x) x
#else

#include <cstdio>

#define cufftCheck(x) _cufftCheck(x, #x ,__FILE__, __LINE__)
#endif

// always check, even if in debug mode
#define cufftAlwaysCheck(x) _cufftCheck(x, #x ,__FILE__, __LINE__)

/// Internal for #cufftCheck(x)
template<typename T>
void _cufftCheck(T e, const char *func, const char *call, const int line) {
  if (e != CUFFT_SUCCESS) {
    const char *errorMsg;
    switch (e)  // https://docs.nvidia.com/cuda/cufft/index.html#cufftresult 
    {
      case CUFFT_INVALID_PLAN: errorMsg = "CUFFT_INVALID_PLAN"; break;
      case CUFFT_ALLOC_FAILED: errorMsg = "CUFFT_ALLOC_FAILED"; break;
      case CUFFT_INVALID_TYPE: errorMsg = "CUFFT_INVALID_TYPE"; break;
      case CUFFT_INVALID_VALUE: errorMsg = "CUFFT_INVALID_VALUE"; break;
      case CUFFT_INTERNAL_ERROR: errorMsg = "CUFFT_INTERNAL_ERROR"; break;
      case CUFFT_EXEC_FAILED: errorMsg = "CUFFT_EXEC_FAILED"; break;
      case CUFFT_SETUP_FAILED: errorMsg = "CUFFT_SETUP_FAILED"; break;
      case CUFFT_INVALID_SIZE: errorMsg = "CUFFT_INVALID_SIZE"; break;
      case CUFFT_UNALIGNED_DATA: errorMsg = "CUFFT_UNALIGNED_DATA"; break;
      case CUFFT_INCOMPLETE_PARAMETER_LIST: errorMsg = "CUFFT_INCOMPLETE_PARAMETER_LIST"; break;
      case CUFFT_INVALID_DEVICE: errorMsg = "CUFFT_INVALID_DEVICE"; break;
      case CUFFT_PARSE_ERROR: errorMsg = "CUFFT_PARSE_ERROR"; break;
      case CUFFT_NO_WORKSPACE: errorMsg = "CUFFT_NO_WORKSPACE"; break;
      case CUFFT_NOT_IMPLEMENTED: errorMsg = "CUFFT_NOT_IMPLEMENTED"; break;
      case CUFFT_LICENSE_ERROR: errorMsg = "CUFFT_LICENSE_ERROR"; break;
      case CUFFT_NOT_SUPPORTED: errorMsg = "CUFFT_NOT_SUPPORTED"; break;
      default: errorMsg = "UNRECOGNIZED CUFFT ERROR CODE";
    }
    printf("\"%s\" at %d in %s\n\treturned %d\n-> %s\n", func, line, call, (int) e, errorMsg);
    exit(EXIT_FAILURE);
  }
}

// Need a __device__ global to read callbacks from symbol memory
// https://docs.nvidia.com/cuda/cufft/index.html#callback-creation
template<typename BricksCufftPlanType, bool fourierBrickDimsFillGridDims>
__device__ typename BricksCufftPlanType::myCufftCallbackLoadType 
globalBricksCufftLoadCallBack = BricksCufftPlanType::template bricksLoadCallback<fourierBrickDimsFillGridDims>;
// Need a __device__ global to read callbacks from symbol memory
// https://docs.nvidia.com/cuda/cufft/index.html#callback-creation
template<typename BricksCufftPlanType, bool fourierBrickDimsFillGridDims>
__device__ typename BricksCufftPlanType::myCufftCallbackStoreType
globalBricksCufftStoreCallBack = BricksCufftPlanType::template bricksStoreCallback<fourierBrickDimsFillGridDims>;
// __constant__ memory for cufft-info
template<typename BricksCufftInfo>
__device__ __constant__ BricksCufftInfo globalInBricksCufftInfo_dev_const;
template<typename BricksCufftInfo>
__device__ __constant__ BricksCufftInfo globalOutBricksCufftInfo_dev_const;

/**
 * @brief Interface from bricks into cfftu
 * 
 * Empty template for specialization
 * 
 * @tparam BrickType the type of the bricks
 * @tparam FFTDims the collection of dimensions to take a fourier transfor over
 */
template<typename BrickType, typename FFTDims> class BricksCufftPlan;

/**
 * ONLY ONE OF THESE CAN EXIST AT A TIME DUE TO USE OF CONSTANT MEMORY
 * 
 * TODO: Allow people to pass in pointers to constant memory
 * 
 * @tparam FFTDims the dimensions to take a fourier transfor over (must be at least 1, at most 3)
 * @tparam isComplex is ignored
 * @see BricksCufftPlan
 * @see Brick
 */
template<bool isComplex,
         typename CommunicatingDims,
         FourierDataType DataType,
         unsigned ... BDims,
         unsigned ... Fold,
         unsigned ... FFTDims
         >
class BricksCufftPlan<Brick<Dim<BDims...>, Dim<Fold...>, isComplex, CommunicatingDims>, FourierType<DataType, FFTDims...> >
{
  public:
    // validate the FFTDims
    static_assert(sizeof...(FFTDims) > 0, "Expected positive FFTRank");
    static_assert(sizeof...(FFTDims) <= sizeof...(BDims), "FFTRank cannot be bigger than number of dimensions");
    static_assert(sizeof...(FFTDims) <= 3, "FFT Dim > 3 not implemented in cufft");
    static_assert(Subset<UnsignedList<FFTDims...>, typename Range<0, sizeof...(BDims)>::type>::value,
                  "FFTDims must be in range 0...num dims");
    //// public typedefs
    typedef Brick<Dim<BDims...>, Dim<Fold...>, DataType != RealToComplex, CommunicatingDims> InBrickType;  ///< type of the input-brick
    typedef Brick<Dim<BDims...>, Dim<Fold...>, DataType != SymmetricComplexToReal, CommunicatingDims> OutBrickType;  ///< type of the ouput-brick
    typedef typename InBrickType::elemType inElemType;  ///< type of input brick elements
    typedef typename OutBrickType::elemType outElemType;  ///< type of input brick elements
    using inCuElemType = typename std::conditional<InBrickType::complex, bCuComplexElem, bElem>::type; ///< cuda equivalent type of input brick elements
    using outCuElemType = typename std::conditional<OutBrickType::complex, bCuComplexElem, bElem>::type; ///< cuda equivalent type of output brick elements
    using myType = BricksCufftPlan<Brick<Dim<BDims...>, Dim<Fold...>, isComplex, CommunicatingDims>, FourierType<DataType, FFTDims...> >;

    // verify in/out types
    static constexpr bool inDoublePrecision = std::is_same<inCuElemType, double>::value || std::is_same<inCuElemType, cuDoubleComplex>::value;
    static constexpr bool outDoublePrecision = std::is_same<outCuElemType, double>::value || std::is_same<outCuElemType, cuDoubleComplex>::value;
    static_assert(!(inDoublePrecision ^ outDoublePrecision), "input and output data types must have same precision");
    static constexpr bool inIsReal = std::is_same<inCuElemType, double>::value || std::is_same<inCuElemType, float>::value;
    static constexpr bool outIsReal = std::is_same<outCuElemType, double>::value || std::is_same<outCuElemType, float>::value;
    static_assert(!(inIsReal && outIsReal), "input and output data types must not both be real");

    /// need to define these for linking definitions
    using nonFourierDimsType = std::array<unsigned, sizeof...(BDims) - sizeof...(FFTDims)>;
    using fourierDimsType = std::array<unsigned, sizeof...(FFTDims)>;
    using brickDimsType = std::array<unsigned, sizeof...(BDims)>;

    //// constants
    static constexpr unsigned FFTRank = sizeof...(FFTDims);  ///< rank of the FFT

    // dimensions not performing an FFT
    static constexpr nonFourierDimsType nonFourierDims = RangeSetMinus<Range<0, sizeof...(BDims)>, FFTDims...>::value;  
    static constexpr fourierDimsType fourierDims = { FFTDims... }; ///< dimensions in which performing an FFT
    static_assert(is_sorted(fourierDims), "FFT dims must be specified in sorted order and contain no duplicates");
    /// brick dimensions (reversed so that in order of most contiguous, ..., least contiguous)
    static constexpr brickDimsType brickDims = StaticArrayBuilder<typename UnsignedListReverser<UnsignedList<BDims...> >::type>::value;
    /// brick extent in each fourier dimension (in order most contiguous, ...., least contiguous)
    static constexpr fourierDimsType fourierBrickDims = { brickDims[FFTDims]... };

    //// product of the fourier brick dimensions
    static constexpr unsigned FOURIER_BRICK_SIZE = reduction(std::multiplies<unsigned>(), 1U, fourierBrickDims);
    //// product of the non-fourier brick-dimensions 
    static constexpr unsigned NON_FOURIER_BRICK_SIZE = reduction(std::multiplies<unsigned>(), 1U, brickDims) / FOURIER_BRICK_SIZE;
    static constexpr bool doublePrecision = inDoublePrecision; ///< are data types double precision
    /// users can specify forward/inverse in a more readable way
    static constexpr bool BRICKS_FFT_FORWARD = false;
    static constexpr bool BRICKS_FFT_INVERSE = true;

    /// my callback load/store function types
    typedef typename std::conditional<doublePrecision, 
                                      typename std::conditional<inIsReal, cufftCallbackLoadD, cufftCallbackLoadZ>::type,
                                      typename std::conditional<inIsReal, cufftCallbackLoadR, cufftCallbackLoadC>::type
                                      >::type myCufftCallbackLoadType;
    typedef typename std::conditional<doublePrecision, 
                                      typename std::conditional<outIsReal, cufftCallbackStoreD, cufftCallbackStoreZ>::type,
                                      typename std::conditional<outIsReal, cufftCallbackStoreR, cufftCallbackStoreC>::type
                                      >::type myCufftCallbackStoreType;

    /// pointers needed by callbacks, stored in constant memory
    template<typename elemType>
    struct BricksCufftInfo
    {
      size_t batchSize; ///< size of each batch
      size_t brick_step; ///< step-size of brick
      const unsigned *grid_ptr; ///< pointer to brick-grid on device
      size_t nonFourierGridExtent[sizeof...(BDims) - FFTRank];
      size_t nonFourierGridStride[sizeof...(BDims) - FFTRank];
      size_t fourierGridExtent[FFTRank];
      size_t fourierGridStride[FFTRank];
    };


    /**
     * @brief Construct a new Bricks CufftPlan object
     * 
     * Allocates the cufft plan
     * 
     * @param grid_size the number of bricks in each dimension ({most contiguous,...,least contiguous})
     * @param type the type of cufft transform
     */
    BricksCufftPlan(std::array<size_t, sizeof...(BDims)> grid_size)
    {
      // get logical embeding of data
      std::array<int, FFTRank> embed;
      for(unsigned i = 0; i < FFTRank; ++i)
      {
        unsigned d = fourierDims[i];
        embed[FFTRank - 1 - i] = grid_size[d] * brickDims[d];
      }
      // figure out size of each batch
      size_t batchSize = 1;
      for(unsigned i = 0; i < FFTRank; ++i) 
      {
        batchSize *= embed[i];
      }
      // compute number of batches
      size_t numBatches = 1;
      for(unsigned i = 0; i < sizeof...(BDims) - FFTRank; ++i)
      {
        unsigned d = nonFourierDims[i];
        numBatches *= grid_size[d] * brickDims[d];
      }
      // set up cuda plan
      cufftCheck(cufftPlanMany(&this->plan, FFTRank, embed.data(),
                               embed.data(), 1, batchSize,
                               embed.data(), 1, batchSize,
                               myCufftType, numBatches));
      
      //// build host-side BricksCufftInfo
      myInCufftInfo.batchSize = batchSize;
      myOutCufftInfo.batchSize = batchSize;
      myInCufftInfo.grid_ptr = nullptr;
      myOutCufftInfo.grid_ptr = nullptr;
      // store grid extents (separated by non-fourier/fourier dims)
      for(unsigned i = 0; i < sizeof...(BDims) - FFTRank; ++i)
      {
        myInCufftInfo.nonFourierGridExtent[i] = grid_size[nonFourierDims[i]];
        myOutCufftInfo.nonFourierGridExtent[i] = grid_size[nonFourierDims[i]];
      }
      for(unsigned i = 0; i < FFTRank; ++i)
      {
        myInCufftInfo.fourierGridExtent[i] = grid_size[fourierDims[i]];
        myOutCufftInfo.fourierGridExtent[i] = grid_size[fourierDims[i]];
      }
      // store grid strides (separated by non-fourier/fourier dims)
      size_t stride = 1;
      for(unsigned dim = 0, nonFourierDimIdx = 0, fourierDimIdx = 0; dim < sizeof...(BDims); ++dim)
      {
        if(nonFourierDimIdx < sizeof...(BDims) - FFTRank && nonFourierDims[nonFourierDimIdx] == dim)
        {
          myInCufftInfo.nonFourierGridStride[nonFourierDimIdx] = stride;
          myOutCufftInfo.nonFourierGridStride[nonFourierDimIdx++] = stride;
        }
        else
        {
          myInCufftInfo.fourierGridStride[fourierDimIdx] = stride;
          myOutCufftInfo.fourierGridStride[fourierDimIdx++] = stride;
        }
        stride *= grid_size[dim];
      }
      // make sure fft dimensions can be factored into primes that allow __syncthreads()
      // https://docs.nvidia.com/cuda/cufft/index.html#callback-coding
      std::array<int, 4> allowed_factors = {2, 3, 5, 7};
      for(auto fDim : fourierBrickDims) {
        for(auto factor : allowed_factors) {
          if(factor <= 1) continue;
          while(fDim > 1 && fDim % factor == 0) fDim /= factor;
        }
        if(fDim != 1) throw std::runtime_error("Each Fourier dimension must not have prime factors greater than 7");
      }
    }

    /**
     * @brief Destroy the Bricks Cufft Plan object
     * Deallocates the cufft plan
     */
    ~BricksCufftPlan()
    {
      cufftCheck(cufftDestroy(plan));
    }

    /**
     * @brief set up for computation
     * 
     * @param inBrick_dev input brick
     * @param in_grid_ptr_dev device pointer to brick-grid for input brick
     * @param outBrick_dev output brick
     * @param out_grid_ptr_dev device pointer to brick-grid output brick
     */
    void setup(InBrickType inBrick_dev, unsigned *in_grid_ptr_dev,
               OutBrickType outBrick_dev, unsigned *out_grid_ptr_dev)
    {
      // set up host-side myCufftInfo
      myInCufftInfo.grid_ptr = in_grid_ptr_dev;
      myInCufftInfo.brick_step = inBrick_dev.step;
      myOutCufftInfo.grid_ptr = out_grid_ptr_dev;
      myOutCufftInfo.brick_step = outBrick_dev.step;
      // record dats
      in_dat = inBrick_dev.dat;
      out_dat = outBrick_dev.dat;

      // copy myCufftInfo to device
      gpuCheck(cudaMemcpyToSymbol(globalInBricksCufftInfo_dev_const<BricksCufftInfo<const inElemType> >,
                                   &myInCufftInfo,
                                   sizeof(BricksCufftInfo<const inElemType>)
                                   ));
      gpuCheck(cudaMemcpyToSymbol(globalOutBricksCufftInfo_dev_const<BricksCufftInfo<outElemType> >,
                                   &myOutCufftInfo,
                                   sizeof(BricksCufftInfo<outElemType>)
                                   ));

      // setup load and store callback
      myCufftCallbackLoadType loadCallbackPtr;
      myCufftCallbackStoreType storeCallbackPtr;
      // if the brick-dim in the FFT-dimensions fills the grid, use a specialized
      // callback to just traverse the BrickStorage contiguously and avoid having
      // to use the grid-pointer
      bool fourier_brick_dims_fill_grid_dims = true;
      for(unsigned i = 0; i < FFTRank ; ++i)
      {
        // if grid-dimension is 1, then the brick fills that dimension
        fourier_brick_dims_fill_grid_dims &= (myInCufftInfo.fourierGridExtent[i] == 1);
      }
      // get address of function
      if(fourier_brick_dims_fill_grid_dims)
      {
        gpuCheck(cudaMemcpyFromSymbol(
                                &loadCallbackPtr, 
                                globalBricksCufftLoadCallBack<myType, true>,
                                sizeof(loadCallbackPtr)));
        gpuCheck(cudaMemcpyFromSymbol(
                                &storeCallbackPtr, 
                                globalBricksCufftStoreCallBack<myType, true>,
                                sizeof(storeCallbackPtr)));
      }
      else
      {
        gpuCheck(cudaMemcpyFromSymbol(
                                &loadCallbackPtr, 
                                globalBricksCufftLoadCallBack<myType, false>,
                                sizeof(loadCallbackPtr)));
        gpuCheck(cudaMemcpyFromSymbol(
                                &storeCallbackPtr, 
                                globalBricksCufftStoreCallBack<myType, false>,
                                sizeof(storeCallbackPtr)));
      }
      cufftCheck(cufftXtSetCallback(plan, (void **)&loadCallbackPtr,  myCufft_CB_LD, 0));
      cufftCheck(cufftXtSetCallback(plan, (void **)&storeCallbackPtr, myCufft_CB_ST, 0));
    }

    /**
     * @brief start the (asynchronous) fast fourier transform
     * @param inverse BRICKS_FFT_FORWARD (false) if forward transform, BRICKS_FFT_INVERSE (true) otherwise
     */
    void launch(bool inverse = BRICKS_FFT_FORWARD)
    {
      // make sure we've setup
      if(myInCufftInfo.grid_ptr == nullptr || myOutCufftInfo.grid_ptr == nullptr)
      {
        throw std::runtime_error("Must call setup before calling launch()");
      }
      // disallow nonsensical inverse
      if(inverse && DataType == RealToComplex)
      {
        throw std::runtime_error("Invalid FourierDataType");
      }
      // start execution
      cufftCheck(cufftXtExec(plan, (void *) in_dat, (void *) out_dat,
                             inverse == BRICKS_FFT_FORWARD ? CUFFT_FORWARD : CUFFT_INVERSE));
    }

    /**
     * @brief callback into bricks from cufft for loads
     * @tparam fourierBrickDimsFillGridDims true if the fourier brick-dims fill the grid dimensions
     * https://docs.nvidia.com/cuda/cufft/index.html#callback-routines
     */
    template<bool fourierBrickDimsFillGridDims>
    static __device__ __forceinline__
    inCuElemType bricksLoadCallback(void * __restrict__ dataIn,
                                    size_t offset,
                                    void * __restrict__ callerInfo,
                                    void * __restrict__ sharedPtr)
    {
      BricksCufftInfo<const inElemType> *cufftInfo = &globalInBricksCufftInfo_dev_const<BricksCufftInfo<const inElemType> >;
      size_t index = getIndex<fourierBrickDimsFillGridDims>(offset, cufftInfo);
      const inCuElemType *in_dat = reinterpret_cast<const inCuElemType*>(dataIn);
      return in_dat[index];
    }

    /**
     * @brief callback into bricks from cufft for stores
     * @tparam fourierBrickDimsFillGridDims true if the fourier brick-dims equal the grid dimensions
     * https://docs.nvidia.com/cuda/cufft/index.html#callback-routines
     */
    template<bool fourierBrickDimsFillGridDims>
    static __device__ __forceinline__
    void bricksStoreCallback(void * __restrict__ dataOut,
                             size_t offset,
                             outCuElemType element,
                             void * __restrict__ callerInfo,
                             void * __restrict__ sharedPtr)
    {
      BricksCufftInfo<outElemType> *cufftInfo = &globalOutBricksCufftInfo_dev_const<BricksCufftInfo<outElemType> >;
      size_t index = getIndex<fourierBrickDimsFillGridDims>(offset, cufftInfo);
      outCuElemType *out_dat = reinterpret_cast<outCuElemType*>(dataOut);
      out_dat[index] = element;
    }

  private:
    // table of cufft fft types and callback types
    static constexpr cufftType cufftTypeTable[2][3] = { {CUFFT_R2C, CUFFT_C2R, CUFFT_C2C}, {CUFFT_D2Z, CUFFT_Z2D, CUFFT_Z2Z} };
    static constexpr cufftXtCallbackType cufftXtCallbackTypeTable[8] =
      { CUFFT_CB_LD_COMPLEX, CUFFT_CB_LD_COMPLEX_DOUBLE, CUFFT_CB_LD_REAL, CUFFT_CB_LD_REAL_DOUBLE,
        CUFFT_CB_ST_COMPLEX, CUFFT_CB_ST_COMPLEX_DOUBLE, CUFFT_CB_ST_REAL, CUFFT_CB_ST_REAL_DOUBLE };
    // my fft type
    static constexpr cufftType myCufftType = cufftTypeTable[doublePrecision][(!inIsReal && !outIsReal) * 2 + (!inIsReal && outIsReal)];
    // my callback load/store types
    static_assert(inIsReal * 2 + inDoublePrecision < 4);
    static constexpr cufftXtCallbackType myCufft_CB_LD = cufftXtCallbackTypeTable[(inIsReal) * 2 + inDoublePrecision];
    static_assert(outIsReal * 2 + outDoublePrecision < 4);
    static constexpr cufftXtCallbackType myCufft_CB_ST = cufftXtCallbackTypeTable[((outIsReal) * 2 + outDoublePrecision) + 4];

    /// the in/out cufft infos
    BricksCufftInfo<const inElemType> myInCufftInfo;
    BricksCufftInfo<outElemType> myOutCufftInfo;
    /// the in/out data ararys for bricks
    const inElemType *in_dat;
    outElemType *out_dat;
    /// cuda plan for FFT execution
    cufftHandle plan;

    /**
     * @brief case where brick-dimension is 1
     * @see shiftInDim
     */
    template<unsigned dim>
    static __device__ __forceinline__
    typename std::enable_if<Dim<BDims...>::template get<dim>() == 1>::type
    shiftInDim(const unsigned * __restrict__ index, const size_t * __restrict__ base) {}

    /**
     * @brief case where brick-dimension is non-1, but vector fold is 1
     * @see shiftInDim
     */
    template<unsigned dim>
    static __device__ __forceinline__
    typename std::enable_if<Dim<BDims...>::template get<dim>() != 1 && Dim<Fold...>::template getOrDefault<dim>(1) == 1>::type
    shiftInDim(unsigned * __restrict__ index, size_t * __restrict__ base)
    {
      constexpr unsigned BRICK_DIM = Dim<BDims...>::template get<dim>();
      constexpr unsigned BRICK_STRIDE = Dim<BDims...>::template product<dim>();
      *base += ((*index) % BRICK_DIM) * BRICK_STRIDE;
      *index /= BRICK_DIM;
    }

    /**
     * @brief shift base by intra-brick index in dim and divide index by the respective brick-dimension
     * 
     * @tparam dim the dimension being indexed into 
     * @param[in,out] index A flattened index into a brick whose most contiguous dimension is dim
     * @param[in,out] base the base value to add to
     */
    template<unsigned dim>
    static __device__ __forceinline__
    typename std::enable_if<Dim<BDims...>::template get<dim>() != 1 && Dim<Fold...>::template getOrDefault<dim>(1) != 1>::type
    shiftInDim(unsigned * __restrict__ index, size_t * __restrict__ base)
    {
      constexpr unsigned BRICK_DIM = Dim<BDims...>::template get<dim>();
      constexpr unsigned BRICK_STRIDE = Dim<BDims...>::template product<dim>();
      constexpr unsigned fold_index = std::min((unsigned) sizeof...(Fold)-1, dim);
      constexpr unsigned VFOLD = Dim<Fold...>::template get<fold_index>();
      constexpr unsigned STRIDE_IN_VEC = Dim<Fold...>::template product<fold_index>();
      *base += (*index % VFOLD) * STRIDE_IN_VEC;
      *index /= VFOLD;
      *base += (*index % (BRICK_DIM / VFOLD)) * (BRICK_STRIDE / STRIDE_IN_VEC);
      *index /= (BRICK_DIM / VFOLD);
    }

    /**
     * @brief Base case for zipping non-fourier and fourier flattened indexes
     * @see zipFlatIndices
     */
    template<unsigned dim, unsigned nonFourierDimIdx, unsigned fourierDimIdx>
    static __device__ __forceinline__
    typename std::enable_if<sizeof...(BDims) <= dim, size_t>::type
    zipFlatIndices(unsigned flatNonFourierIndex, unsigned flatFourierIndex, const size_t base) 
    {
      return base;
    }

    /**
     * @brief compute index of vector and index in vector inside the brick
     * @see zipFlatIndices
     */
    template<unsigned dim, unsigned nonFourierDimIdx, unsigned fourierDimIdx>
    static __device__ __forceinline__
    typename std::enable_if<dim < sizeof...(BDims)
                            && nonFourierDimIdx < sizeof...(BDims) - FFTRank
                            && dim == nonFourierDims[nonFourierDimIdx], size_t>::type
    zipFlatIndices(unsigned flatNonFourierIndex, const unsigned flatFourierIndex, size_t base)
    {
      shiftInDim<dim>(&flatNonFourierIndex, &base);
      return zipFlatIndices<dim+1, nonFourierDimIdx+1, fourierDimIdx>(flatNonFourierIndex, flatFourierIndex, base);
    }

    /**
     * @brief zip the flat indexes (represented as index of vec and index in vec) and add to base
     * 
     * case where current dim is a fourier dimension
     * 
     * @tparam dim current dimension
     * @tparam nonFourierDimIdx current index into non-fourier dimensions
     * @tparam fourierDimIdx current index into fourier dimensions
     * @param flatNonFourierIndex flattened index into remaining non-fourier dimensions
     * @param flatFourierIndex flattened index into remaining fourier dimensions
     * @param base the base value to add the index to
     * @return base + the index into the brick
     */
    template<unsigned dim, unsigned nonFourierDimIdx, unsigned fourierDimIdx>
    static __device__ __forceinline__
    typename std::enable_if<dim < sizeof...(BDims)
                            && fourierDimIdx < FFTRank
                            && dim == fourierDims[fourierDimIdx], size_t>::type
    zipFlatIndices(const unsigned flatNonFourierIndex, unsigned flatFourierIndex, size_t base)
    {
      shiftInDim<dim>(&flatFourierIndex, &base);
      return zipFlatIndices<dim+1, nonFourierDimIdx, fourierDimIdx+1>(flatNonFourierIndex, flatFourierIndex, base);
    }

    /**
     * @brief Get the Index Of the brick object
     * 
     * Case where fourier brick dimenions equal fourier grid dimensions
     * @see getIndexOfBrick
     */
    template<bool fftBrickDimsFillGridDims, typename elemType>
    static __device__ __forceinline__
    typename std::enable_if<fftBrickDimsFillGridDims, unsigned>::type
    getIndexOfBrick(const unsigned batchIndex, const unsigned fourierIndex, const BricksCufftInfo<elemType> * cufftInfo) 
    {
      return batchIndex;
    }

    /**
     * @brief Get the Index Of the brick object
     * 
     * Case where fourier brick dimensions are not equal fourier grid dimensions
     * 
     * @tparam fftBrickDimsFillGridDims true if each fourier brick dimension fills the grid-dimension
     * @param batchIndex index of the batch into the grid
     * @param fourierIndex index of the fourier transform into the grid
     * @param cufftInfo bricks cufft info
     * @return index of the brick
     */
    template<bool fftBrickDimsFillGridDims, typename elemType>
    static __device__ __forceinline__
    typename std::enable_if<!fftBrickDimsFillGridDims, unsigned>::type
    getIndexOfBrick(unsigned batchIndex, unsigned fourierIndex, const BricksCufftInfo<elemType> * cufftInfo) 
    {
      // get index into grid
      unsigned gridIndex = 0;
      for(unsigned idx = 0; idx < sizeof...(BDims) - FFTRank; ++idx)
      {
        gridIndex += (batchIndex % cufftInfo->nonFourierGridExtent[idx]) * cufftInfo->nonFourierGridStride[idx];
        batchIndex /= (cufftInfo->nonFourierGridExtent[idx]);
      }
      for(unsigned idx = 0; idx < FFTRank; ++idx)
      {
        gridIndex += (fourierIndex % cufftInfo->fourierGridExtent[idx]) * cufftInfo->fourierGridStride[idx];
        fourierIndex /= (cufftInfo->fourierGridExtent[idx]);
      }
      // return index of brick
      const unsigned * grid_ptr = cufftInfo->grid_ptr;
      return grid_ptr[gridIndex];
    }

    /**
     * @brief Get index into brick array
     * 
     * @tparam fftBricksDimsFillGridDims true if fourier brick-dims equal grid-dims
     * @param offset lofical offset in the batches of FFTs received from a cufft callback
     * @param cufftInfo pointer to the BricksCufftInfo object
     * @see BricksCufftInfo
     * @return pointer to the element
     */
    template<bool fftBrickDimsFillGridDims, typename elemType>
    static __device__  __forceinline__
    size_t getIndex(const size_t offset, const BricksCufftInfo<elemType> *cufftInfo)
    {
      // Figure out flat indices (separated by fourier and non-fourier dimensions)
      unsigned batchIndex = offset / cufftInfo->batchSize;
      unsigned fourierIndex = offset % cufftInfo->batchSize;

      unsigned batchIndexInGrid = batchIndex / NON_FOURIER_BRICK_SIZE;
      unsigned batchIndexInBrick = batchIndex % NON_FOURIER_BRICK_SIZE;

      unsigned fourierIndexInGrid = fourierIndex / FOURIER_BRICK_SIZE;
      unsigned fourierIndexInBrick = fourierIndex % FOURIER_BRICK_SIZE;

      // get brick index
      unsigned indexOfBrick = getIndexOfBrick<fftBrickDimsFillGridDims>(batchIndexInGrid, fourierIndexInGrid, cufftInfo);

      return zipFlatIndices<0,0,0>(batchIndexInBrick, fourierIndexInBrick, indexOfBrick * cufftInfo->brick_step);
    }
};

// handle commas like in https://stackoverflow.com/questions/13842468/comma-in-c-c-macro
namespace 
{
  template<typename T> struct argument_type;
  template<typename T, typename U> struct argument_type<T(U)> { typedef U type; };
}
// define static members here to avoid linker errors for odr-used static constexprs
#define BricksCufftPlan_STATIC_CONSTEXPR_DEF(t, name) \
template<bool isComplex, \
         typename CommunicatingDims, \
         FourierDataType DataType, \
         unsigned ... BDims, \
         unsigned ... Fold, \
         unsigned ... FFTDims \
         > \
constexpr typename argument_type<void(t)>::type \
BricksCufftPlan<Brick<Dim<BDims...>, Dim<Fold...>, isComplex, CommunicatingDims>, FourierType<DataType, FFTDims...> >::name;
BricksCufftPlan_STATIC_CONSTEXPR_DEF(bool, inDoublePrecision)
BricksCufftPlan_STATIC_CONSTEXPR_DEF(bool, outDoublePrecision)
BricksCufftPlan_STATIC_CONSTEXPR_DEF(bool, inIsReal)
BricksCufftPlan_STATIC_CONSTEXPR_DEF(bool, outIsReal)
BricksCufftPlan_STATIC_CONSTEXPR_DEF(unsigned, FFTRank)
BricksCufftPlan_STATIC_CONSTEXPR_DEF(unsigned, FOURIER_BRICK_SIZE)
BricksCufftPlan_STATIC_CONSTEXPR_DEF(unsigned, NON_FOURIER_BRICK_SIZE)
BricksCufftPlan_STATIC_CONSTEXPR_DEF(bool, doublePrecision)
BricksCufftPlan_STATIC_CONSTEXPR_DEF(bool, BRICKS_FFT_FORWARD)
BricksCufftPlan_STATIC_CONSTEXPR_DEF(bool, BRICKS_FFT_INVERSE)

// define static members here to avoid linker errors for odr-used static constexprs
// (here we use types that are defined inside the scope of BricksCufftPlan, since
// otherwise the declarations don't match the defintions)
#define BricksCufftPlan_STATIC_CONSTEXPR_DEF_SCOPED_TYPE(t, name) \
template<bool isComplex, \
         typename CommunicatingDims, \
         FourierDataType DataType, \
         unsigned ... BDims, \
         unsigned ... Fold, \
         unsigned ... FFTDims \
         > \
constexpr typename \
BricksCufftPlan<Brick<Dim<BDims...>, Dim<Fold...>, isComplex, CommunicatingDims>, FourierType<DataType, FFTDims...> >::t \
BricksCufftPlan<Brick<Dim<BDims...>, Dim<Fold...>, isComplex, CommunicatingDims>, FourierType<DataType, FFTDims...> >::name;
BricksCufftPlan_STATIC_CONSTEXPR_DEF_SCOPED_TYPE(nonFourierDimsType, nonFourierDims)
BricksCufftPlan_STATIC_CONSTEXPR_DEF_SCOPED_TYPE(fourierDimsType, fourierDims)
BricksCufftPlan_STATIC_CONSTEXPR_DEF_SCOPED_TYPE(brickDimsType, brickDims)
BricksCufftPlan_STATIC_CONSTEXPR_DEF_SCOPED_TYPE(fourierDimsType, fourierBrickDims)

#endif // defined(__CUDACC__)

#endif // BRICK_FFT_H