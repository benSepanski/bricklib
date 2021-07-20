/**
 * @file 
 * @brief hook for FFT computations on bricks
 */

#ifndef BRICK_FFT_H
#define BRICK_FFT_H

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

// cuFFT interface
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

/// Internal for #cufftCheck(x)
template<typename T>
void _cufftCheck(T e, const char *func, const char *call, const int line) {
  if (e != CUFFT_SUCCESS) {
    const char *errorMsg;
    switch (e)  // https://docs.nvidia.com/cuda/cufft/index.html#cufftresult 
    {
      CUFFT_INVALID_PLAN: errorMsg = "CUFFT_INVALID_PLAN"; break;
      CUFFT_ALLOC_FAILED: errorMsg = "CUFFT_ALLOC_FAILED"; break;
      CUFFT_INVALID_TYPE: errorMsg = "CUFFT_INVALID_TYPE"; break;
      CUFFT_INVALID_VALUE: errorMsg = "CUFFT_INVALID_VALUE"; break;
      CUFFT_INTERNAL_ERROR: errorMsg = "CUFFT_INTERNAL_ERROR"; break;
      CUFFT_EXEC_FAILED: errorMsg = "CUFFT_EXEC_FAILED"; break;
      CUFFT_SETUP_FAILED: errorMsg = "CUFFT_SETUP_FAILED"; break;
      CUFFT_INVALID_SIZE: errorMsg = "CUFFT_INVALID_SIZE"; break;
      CUFFT_UNALIGNED_DATA: errorMsg = "CUFFT_UNALIGNED_DATA"; break;
      CUFFT_INCOMPLETE_PARAMETER_LIST: errorMsg = "CUFFT_INCOMPLETE_PARAMETER_LIST"; break;
      CUFFT_INVALID_DEVICE: errorMsg = "CUFFT_INVALID_DEVICE"; break;
      CUFFT_PARSE_ERROR: errorMsg = "CUFFT_PARSE_ERROR"; break;
      CUFFT_NO_WORKSPACE: errorMsg = "CUFFT_NO_WORKSPACE"; break;
      CUFFT_NOT_IMPLEMENTED: errorMsg = "CUFFT_NOT_IMPLEMENTED"; break;
      CUFFT_LICENSE_ERROR: errorMsg = "CUFFT_LICENSE_ERROR"; break;
      CUFFT_NOT_SUPPORTED: errorMsg = "CUFFT_NOT_SUPPORTED"; break;
      default: errorMsg = "UNRECOGNIZED CUFFT ERROR CODE";
    }
    printf("\"%s\" at %d in %s\n\treturned %d\n-> %s\n", func, line, call, (int) e, errorMsg);
    exit(EXIT_FAILURE);
  }
}

/**
 * @brief used to select the appropriate cufftExec function
 */
template<cufftType Type> struct CufftExecSelector;
template<> struct CufftExecSelector<CUFFT_R2C> { constexpr static auto value = cufftExecR2C; };
template<> struct CufftExecSelector<CUFFT_C2R> { constexpr static auto value = cufftExecC2R; };
template<> struct CufftExecSelector<CUFFT_C2C> { constexpr static auto value = cufftExecC2C; };
template<> struct CufftExecSelector<CUFFT_D2Z> { constexpr static auto value = cufftExecD2Z; };
template<> struct CufftExecSelector<CUFFT_Z2D> { constexpr static auto value = cufftExecZ2D; };
template<> struct CufftExecSelector<CUFFT_Z2Z> { constexpr static auto value = cufftExecZ2Z; };

/**
 * @brief Interface from bricks into cuFFT
 * 
 * Empty template for specialization
 * 
 * @tparam BrickType the type of the bricks
 * @tparam FFTDims the collection of dimensions to take a fourier transfor over
 */
template<typename BrickType, typename FFTDims> class BricksCufftPlan;

/**
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
    static_assert(sizeof...(FFTDims) <= 3, "FFT Dim > 3 not implemented in cuFFT");
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

    //// constants
    static constexpr unsigned FFTRank = sizeof...(FFTDims);  ///< rank of the FFT
    static constexpr std::array<unsigned int, sizeof...(BDims) - FFTRank> 
      nonFourierDims = RangeSetMinus<Range<0, sizeof...(BDims)>, FFTDims...>::value;  ///< dimensions not performing an FFT
    static constexpr std::array<unsigned int, FFTRank> fourierDims = { FFTDims... }; ///< dimensions in which performing an FFT
    static constexpr std::array<unsigned, sizeof...(BDims)> brickDims = { BDims... };  ///< extent of brick in each dimension
    /// brick extent in each fourier dimension
    static constexpr std::array<unsigned, FFTRank> fourierBrickDims = { brickDims[FFTDims]... };
    //// product of the fourier brick dimensions
    static constexpr unsigned FOURIER_BRICK_SIZE = reduction(std::multiplies<unsigned>(), 1U, fourierBrickDims);
    //// product of the non-fourier brick-dimensions 
    static constexpr unsigned NON_FOURIER_BRICK_SIZE = reduction(std::multiplies<unsigned>(), 1U, brickDims) / FOURIER_BRICK_SIZE;
    // static_assert(FOURIER_BRICK_SIZE == 4);
    // static_assert(NON_FOURIER_BRICK_SIZE == 2);
    static constexpr bool doublePrecision = inDoublePrecision; ///< are data types double precision

    /// pointers passed to callbacks
    struct BricksCufftInfo
    {
      size_t batchSize; ///< size of each batch
      size_t nonFourierGridExtent[sizeof...(BDims) - FFTRank];
      size_t nonFourierGridStride[sizeof...(BDims) - FFTRank];
      size_t fourierGridExtent[FFTRank];
      size_t fourierGridStride[FFTRank];
      const unsigned *in_grid_ptr; ///< pointer to in-brick-grid on device
      const unsigned *out_grid_ptr; ///< pointer to out-brick-grid on device
      InBrickType *inBrick; ///< pointer to input brick
      OutBrickType *outBrick; ///< pointer to output brick
    };

    /**
     * @brief Construct a new Bricks CufftPlan object
     * 
     * Allocates the cuFFT plan
     * 
     * @param grid_size the number of bricks in each dimension
     * @param type the type of cuFFT transform
     */
    BricksCufftPlan(std::array<size_t, sizeof...(BDims)> grid_size)
    {
      // get logical embeding of data
      std::array<int, FFTRank> embed;
      embed[0] = 1;
      for(unsigned i = 0; i + 1 < FFTRank; ++i)
      {
        embed[i + 1] = grid_size[fourierDims[i]];
      }
      // figure out size of each batch
      size_t batchSize = grid_size[fourierDims[FFTRank - 1]];
      for(unsigned i = 0; i < FFTRank; ++i) 
      {
        batchSize *= embed[i];
      }
      // compute number of batches
      size_t numBatches = 1;
      for(unsigned i = 0; i < sizeof...(BDims) - FFTRank; ++i)
      {
        numBatches *= nonFourierDims[i];
      }
      // set up cuda plan
      cufftPlanMany(&this->plan, FFTRank, embed.data(),
                    nullptr, -1, batchSize,
                    nullptr, -1, batchSize,
                    myCufftType, numBatches);
      
      //// build host-side BricksCufftInfo
      myCufftInfo.batchSize = batchSize;
      myCufftInfo.in_grid_ptr = nullptr;
      myCufftInfo.out_grid_ptr = nullptr;
      myCufftInfo.inBrick = nullptr;
      myCufftInfo.outBrick = nullptr;
      // store grid extents (separated by non-fourier/fourier dims)
      for(unsigned i = 0; i < sizeof...(BDims) - FFTRank; ++i)
      {
        myCufftInfo.nonFourierGridExtent[i] = grid_size[nonFourierDims[i]];
      }
      for(unsigned i = 0; i < FFTRank; ++i)
      {
        myCufftInfo.fourierGridExtent[i] = grid_size[fourierDims[i]];
      }
      // store grid strides (separated by non-fourier/fourier dims)
      size_t stride = 1;
      for(unsigned dim = 0, nonFourierDimIdx = 0, fourierDimIdx = 0; dim < sizeof...(BDims); ++dim)
      {
        if(nonFourierDimIdx < sizeof...(BDims) - FFTRank && nonFourierDims[nonFourierDimIdx] == dim)
        {
          myCufftInfo.nonFourierGridStride[nonFourierDimIdx++] = stride;
        }
        else
        {
          myCufftInfo.fourierGridStride[fourierDimIdx++] = stride;
        }
        stride *= grid_size[dim];
      }
    }

    /**
     * @brief Destroy the Bricks Cu F F T Plan object
     * Deallocates the cuFFT plan
     */
    ~BricksCufftPlan()
    {
      cufftDestroy(plan);
    }

    // TODO: doc
    void setup(InBrickType *inBrick_dev, unsigned *in_grid_ptr_dev,
               OutBrickType *outBrick_dev, unsigned *out_grid_ptr_dev,
               BricksCufftInfo *cuFFTInfo_dev)
    {
      // set in host-side myCufftInfo
      myCufftInfo.in_grid_ptr = in_grid_ptr_dev;
      myCufftInfo.inBrick = inBrick_dev;
      myCufftInfo.out_grid_ptr = out_grid_ptr_dev;
      myCufftInfo.outBrick = outBrick_dev;

      // copy myCufftInfo_dev into device memory
      cudaCheck(cudaMemcpyToSymbol(cuFFTInfo_dev, &myCufftInfo, sizeof(BricksCufftInfo)));

      // // setup load and store callback
      myCufftCallbackLoadType loadCallbackPtr;
      myCufftCallbackStoreType storeCallbackPtr;
      // // get address of function
      cudaCheck(cudaMemcpyFromSymbol(
                              &loadCallbackPtr, 
                              myType::bricksLoadCallback,
                              sizeof(loadCallbackPtr)));
      cudaCheck(cudaMemcpyFromSymbol(
                              &storeCallbackPtr, 
                              myType::bricksStoreCallback,
                              sizeof(storeCallbackPtr)));
      // // set the callbacks
      cufftCheck(cufftXtSetCallback(this->plan,
                             (void **)&loadCallbackPtr,
                             this->myCuFFT_CB_LD,
                             (void **)&cuFFTInfo_dev));
      cufftCheck(cufftXtSetCallback(this->plan,
                             (void **)&storeCallbackPtr,
                             this->myCuFFT_CB_ST,
                             (void **)&cuFFTInfo_dev));
    }

    // TODO: Docs
    void launch(bool inverse = false)
    {
      // make sure we've setup
      if(myCufftInfo->inBrick == nullptr
         || myCufftInfo->in_grid_ptr == nullptr
         || myCufftInfo->outBrick == nullptr
         || myCufftInfo->out_grid_ptr == nullptr)
      {
        throw std::runtime_error("Must call setup before calling launch()");
      }
      // disallow nonsensical inverse
      if(inverse && DataType == RealToComplex)
      {
        throw std::runtime_error("Invalid FourierDataType");
      }
    }

  private:
    // info needed for callbacks
    BricksCufftInfo myCufftInfo;
    // table of cufft fft types and callback types
    static constexpr cufftType cufftTypeTable[2][3] = { {CUFFT_R2C, CUFFT_C2R, CUFFT_C2C}, {CUFFT_D2Z, CUFFT_Z2D, CUFFT_Z2Z} };
    static constexpr cufftXtCallbackType cufftXtCallbackTypeTable[8] =
      { CUFFT_CB_LD_COMPLEX, CUFFT_CB_LD_COMPLEX_DOUBLE, CUFFT_CB_LD_REAL, CUFFT_CB_LD_REAL_DOUBLE,
        CUFFT_CB_ST_COMPLEX, CUFFT_CB_ST_COMPLEX_DOUBLE, CUFFT_CB_ST_REAL, CUFFT_CB_ST_REAL_DOUBLE };
    // my fft type
    static constexpr cufftType myCufftType = cufftTypeTable[doublePrecision][(!inIsReal && !outIsReal) << 1 + (!inIsReal && outIsReal)];
    // my callback load/store types
    static constexpr cufftXtCallbackType myCuFFT_CB_LD = cufftXtCallbackTypeTable[(inIsReal) << 1 + inDoublePrecision];
    static constexpr cufftXtCallbackType myCuFFT_CB_ST = cufftXtCallbackTypeTable[((outIsReal) << 1 + outDoublePrecision) + 4];
    /// my callback load/store function types
    typedef typename std::conditional<doublePrecision, 
                                      typename std::conditional<inIsReal, cufftCallbackLoadD, cufftCallbackLoadZ>::type,
                                      typename std::conditional<inIsReal, cufftCallbackLoadR, cufftCallbackLoadC>::type
                                      >::type myCufftCallbackLoadType;
    typedef typename std::conditional<doublePrecision, 
                                      typename std::conditional<outIsReal, cufftCallbackStoreD, cufftCallbackStoreZ>::type,
                                      typename std::conditional<outIsReal, cufftCallbackStoreR, cufftCallbackStoreC>::type
                                      >::type myCufftCallbackStoreType;
    /// my exec function
    static constexpr auto myCufftExec = CufftExecSelector<myCufftType>::value;

    /// cuda plan for FFT execution
    cufftHandle plan;

    // TODO: Docs
    template<unsigned dim, unsigned nonFourierDimIdx, unsigned fourierDimIdx, typename elemType>
    __device__ static
    typename std::enable_if<dim >= sizeof...(BDims), elemType &>::type
    accessFromFlatIndexes(elemType &brickElt, unsigned flatNonFourierIndex, unsigned flatFourierIndex) { return brickElt;}

    // TODO: Docs
    template<unsigned dim, unsigned nonFourierDimIdx, unsigned fourierDimIdx, typename T>
    __device__ static
    typename std::enable_if<dim < sizeof...(BDims)
                            && nonFourierDimIdx < sizeof...(BDims) - FFTRank
                            && dim == nonFourierDims[nonFourierDimIdx], typename T::elemType &>::type
    accessFromFlatIndexes(T brickAccessor, unsigned flatNonFourierIndex, unsigned flatFourierIndex)
    {
      constexpr unsigned BRICK_STRIDE = Dim<BDims...>::template product<dim>();
      unsigned indexInDim = flatNonFourierIndex / BRICK_STRIDE;
      unsigned newFlatNonFourierIndex = flatNonFourierIndex % BRICK_STRIDE;
      constexpr unsigned test[] = { FFTDims... };
      static_assert(dim+1 == test[fourierDimIdx]);
      static_assert(fourierDimIdx < FFTRank);
      static_assert(dim+1 < sizeof...(BDims));
      return accessFromFlatIndexes<dim+1, nonFourierDimIdx+1, fourierDimIdx>(brickAccessor[indexInDim], newFlatNonFourierIndex, flatFourierIndex);
    }

    // TODO: Docs
    template<unsigned dim, unsigned nonFourierDimIdx, unsigned fourierDimIdx, typename T>
    __device__ static
    typename std::enable_if<dim < sizeof...(BDims)
                            && fourierDimIdx < FFTRank
                            && dim == fourierDims[fourierDimIdx], typename T::elemType &>::type
    accessFromFlatIndexes(T brickAccessor, unsigned flatNonFourierIndex, unsigned flatFourierIndex)
    {
      constexpr unsigned BRICK_STRIDE = Dim<BDims...>::template product<dim>();
      unsigned indexInDim = flatFourierIndex / BRICK_STRIDE;
      unsigned newFlatFourierIndex = flatFourierIndex % BRICK_STRIDE;
      return accessFromFlatIndexes<dim+1, nonFourierDimIdx, fourierDimIdx+1>(brickAccessor[indexInDim], flatNonFourierIndex, newFlatFourierIndex);
    }

    // TODO: doc
    template<typename BrickType>
    static __device__ __forceinline__
    typename BrickType::elemType &getElement(BrickType *brick, const unsigned *grid_ptr, size_t offset, BricksCufftInfo *cuFFTInfo)
    {
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
      unsigned indexOfBrick = grid_ptr[gridIndex];
      auto brickAcc = (*brick)[indexOfBrick];
      return accessFromFlatIndexes<0,0,0>(brickAcc, batchIndexInBrick, fourierIndexInBrick);
    }


    // https://docs.nvidia.com/cuda/cufft/index.html#callback-routines
    static __device__
    inCuElemType bricksLoadCallback(void *dataIn,
                                    size_t offset,
                                    void *callerInfo,
                                    void *sharedPtr)
    {
      // get info
      BricksCufftInfo *cuFFTInfo = reinterpret_cast<BricksCufftInfo*>(callerInfo);
      // return the element
      return reinterpret_cast<inCuElemType&>(
          getElement(cuFFTInfo->inBrick, cuFFTInfo->in_grid_ptr, offset, cuFFTInfo)
      );
    }

    // https://docs.nvidia.com/cuda/cufft/index.html#callback-routines
    static __device__
    void bricksStoreCallback(void *dataOut,
                             size_t offset,
                             outCuElemType element,
                             void *callerInfo,
                             void *sharedPtr)
    {
      // get info
      BricksCufftInfo *cuFFTInfo = reinterpret_cast<BricksCufftInfo*>(callerInfo);
      // set the element
      getElement(cuFFTInfo->outBrick, cuFFTInfo->out_grid_ptr, offset, cuFFTInfo) = element;
    }

};

#endif // defined(__CUDACC__)

#endif // BRICK_FFT_H