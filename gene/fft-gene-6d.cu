#include "fft.h"
#include "nvToolsExt.h"
#include "single/transpose-cu.h"
#include "brick-stencils.h"
#include "single-util.h"
#include "util.h"
#include <cmath>
#include <iomanip>
#include <iostream>

// NOTE: none of the arrays use padding or ghost-zones
// TODO: GET RID OF THESE GLOBALLY DEFINED VALUES
#define EXTENT_i 72
#define EXTENT_j 32
#define EXTENT_k 24
#define EXTENT_l 24
#define EXTENT_m 32
#define EXTENT_n 2
#define NUM_ELEMENTS EXTENT_i * EXTENT_j * EXTENT_k * EXTENT_l * EXTENT_m * EXTENT_n
#define EXTENT EXTENT_i,EXTENT_j,EXTENT_k,EXTENT_l,EXTENT_m,EXTENT_n
#define BDIM_i BRICK_DIM[0]
#define BDIM_j BRICK_DIM[1]
#define BDIM_k BRICK_DIM[2]
#define BDIM_l BRICK_DIM[3]
#define BDIM_m BRICK_DIM[4]
#define BDIM_n BRICK_DIM[5]
#define BRICK_EXTENT_i EXTENT_i/BDIM_i
#define BRICK_EXTENT_j EXTENT_j/BDIM_j
#define BRICK_EXTENT_k EXTENT_k/BDIM_k
#define BRICK_EXTENT_l EXTENT_l/BDIM_l
#define BRICK_EXTENT_m EXTENT_m/BDIM_m
#define BRICK_EXTENT_n EXTENT_n/BDIM_n
#define NUM_BRICKS BRICK_EXTENT_i * BRICK_EXTENT_j * BRICK_EXTENT_k * BRICK_EXTENT_l * BRICK_EXTENT_m * BRICK_EXTENT_n
#define BRICK_EXTENT BRICK_EXTENT_i,BRICK_EXTENT_j,BRICK_EXTENT_k,BRICK_EXTENT_l,BRICK_EXTENT_m,BRICK_EXTENT_n

#define CU_WARMUP 5
#define CU_ITER 100
// TODO: REPLACE WITH timeAndPrintStats
template<typename T>
double cutime_func(T func, unsigned cu_warmup = CU_WARMUP, unsigned cu_iter = CU_ITER) {
  for(int i = 0; i < cu_warmup; ++i) func(); // Warm up
  cudaEvent_t start, stop;
  float elapsed = 0.0;
  gpuCheck(cudaDeviceSynchronize());
  gpuCheck(cudaEventCreate(&start));
  gpuCheck(cudaEventCreate(&stop));
  gpuCheck(cudaEventRecord(start));
  for (int i = 0; i < cu_iter; ++i)
    func();
  gpuCheck(cudaEventRecord(stop));
  gpuCheck(cudaEventSynchronize(stop));
  gpuCheck(cudaEventElapsedTime(&elapsed, start, stop));
  return elapsed / cu_iter / 1000;
}

/**
 * @brief exit with error message if any element of arr1 and arr2 are not within tol of each other
 * 
 * @param arr1 first array
 * @param arr2 second array
 * @param N number of elements
 * @param errorMsg error message to include if not close
 * @param arr2scale a number to scale arr2 by
 * @param tol error tolerance
 */
void check_close(bComplexElem *arr1, bComplexElem *arr2, int N, std::string errorMsg = "", double arr2scale = 1.0, float tol=1e-7)
{
  long failureIndex = -1;
  #pragma omp parallel
  for(unsigned i = 0; i < N; ++i)
  {
    std::complex<bElem> diff = (arr1[i] - arr2[i] * arr2scale);
    if(std::abs(diff) >= tol)
    {
      failureIndex = i;
      break;
    }
    if(i % 1000 == 0 && failureIndex != -1) break; ///< periodically check to break early
  }
  if(failureIndex != -1)
  {
    int i = failureIndex;
    std::cerr << errorMsg
              << ": Mismatch at index " << i << ": " << arr1[i] << " != " << arr2[i] * arr2scale
              << " (arr2scale = " << arr2scale
              << ", ratio arr1[i] / (arr2[i]*arr2scale) = "
              << arr1[i] / (arr2[i] * arr2scale) << ")"
              << std::endl;
    exit(1);
  }
}

/**
 * @brief time computation of 1D FFT in j-dimension of in_arr into out_arr
 * 
 * Uses a series of batched call to cufft (no data reordering)
 * 
 * @param in_arr[in] host-side input array, of rank DIM with extents EXTENT_n,...,EXTENT_i
 * @param out_arr[out] host-side output array of same shape as in_arr
 * @param direction[in] either CUFFT_FORWARD or CUFFT_INVERSE
 * @param warmup the number of warmups
 * @param iter the number of iters
 * @return the average number of seconds to compute the FFT
 */
double complex_to_complex_1d_j_fft_array(bComplexElem *in_arr, bComplexElem *out_arr, int warmup, int iter, int direction = CUFFT_FORWARD)
{
  if(direction != CUFFT_FORWARD && direction != CUFFT_INVERSE) throw std::runtime_error("Unrecognized direction");
  // build cufft plan
  cufftHandle plan;
  int fft_rank = 1;
  int array_rank[1] = {EXTENT_j};
  int embed[1] = {EXTENT_j};
  int stride = EXTENT_i;
  int distBetweenBatches = EXTENT_j * EXTENT_i;
  int numBatches = NUM_ELEMENTS / distBetweenBatches;
  static_assert(std::is_same<bElem, float>::value || std::is_same<bElem, double>::value);
  cufftType type = std::is_same<bElem, float>::value ? CUFFT_C2C : CUFFT_Z2Z;
  cufftAlwaysCheck(cufftPlanMany(&plan, fft_rank, array_rank,
                           embed, stride, distBetweenBatches,
                           embed, stride, distBetweenBatches,
                           type, numBatches));

  // copy data to device
  bCuComplexElem *in_arr_dev, *out_arr_dev;
  constexpr size_t ARR_SIZE = sizeof(bComplexElem) * NUM_ELEMENTS;
  gpuCheck(cudaMalloc(&in_arr_dev, ARR_SIZE));
  gpuCheck(cudaMalloc(&out_arr_dev, ARR_SIZE));
  gpuCheck(cudaMemcpy(in_arr_dev, in_arr, ARR_SIZE, cudaMemcpyHostToDevice));

  // lambda function for timing
  auto compute_fft = [&plan](bCuComplexElem *in_arr, bCuComplexElem *out_arr, int direction) -> void 
  {
    // launch cufft execution
    for(unsigned i = 0; i < EXTENT_i; ++i)
    {
      cufftAlwaysCheck(cufftXtExec(plan, in_arr + i, out_arr + i, direction));
    }
  };
  // time function (and compute fft in process)
  double num_seconds = cutime_func([&compute_fft, &in_arr_dev, &out_arr_dev, &direction]() -> void {
    compute_fft(in_arr_dev, out_arr_dev, direction);
  }, warmup, iter);

  // copy data back from device
  gpuCheck(cudaMemcpy(out_arr, out_arr_dev, ARR_SIZE, cudaMemcpyDeviceToHost));

  // sanity check: inverse should match in_arr (up to scaling)
  nvtxRangePushA("inverse_fft_check");
  {
    // malloc space for the inverse computation
    bComplexElem *out_check_arr = zeroComplexArray({EXTENT});
    bCuComplexElem *out_check_arr_dev;
    size_t size = NUM_ELEMENTS * sizeof(bComplexElem);
    gpuCheck(cudaMalloc(&out_check_arr_dev, size));
    // perform the inverse computation
    int inverse_direction = (direction == CUFFT_FORWARD) ? CUFFT_INVERSE : CUFFT_FORWARD;
    compute_fft(out_arr_dev, out_check_arr_dev, inverse_direction);
    gpuCheck(cudaDeviceSynchronize());
    // copy check back to host and make sure it is correct
    gpuCheck(cudaMemcpy(out_check_arr, out_check_arr_dev, size, cudaMemcpyDeviceToHost));
    check_close(in_arr, out_check_arr, NUM_ELEMENTS, "cufft 1d j array inverse check failed", 1.0 / EXTENT_j);
    // free memroy
    gpuCheck(cudaFree(out_check_arr_dev));
    free(out_check_arr);
  }
  nvtxRangePop();

  // free memory
  gpuCheck(cudaFree(out_arr_dev));
  gpuCheck(cudaFree(in_arr_dev));
  cufftAlwaysCheck(cufftDestroy(plan));

  // return timing
  return num_seconds;
}

/**
 * @brief callback into array from cufft for loads
 * Swaps logical i-j to physical j-i
 * https://docs.nvidia.com/cuda/cufft/index.html#callback-routines
 */
__device__ __forceinline__
bCuComplexElem array_load_callback(void * __restrict__ dataIn,
                                   size_t offset,
                                   void * __restrict__ callerInfo,
                                   void * __restrict__ sharedPtr)
{
  assert(offset < NUM_ELEMENTS);
  unsigned j = offset % EXTENT_j;
  unsigned i = (offset / EXTENT_j) % EXTENT_i;
  size_t index = offset + (EXTENT_i - 1) * j + (1 - (size_t) EXTENT_j) * i;
  assert(index < NUM_ELEMENTS);
  return ((bCuComplexElem *) dataIn)[index];
}

/**
 * @brief callback into array from cufft for stores
 * Swaps logical i-j to physical j-i
 * https://docs.nvidia.com/cuda/cufft/index.html#callback-routines
 */
__device__ __forceinline__
void array_store_callback(void * __restrict__ dataOut,
                          size_t offset,
                          bCuComplexElem element,
                          void * __restrict__ callerInfo,
                          void * __restrict__ sharedPtr)
{
  assert(offset < NUM_ELEMENTS);
  unsigned j = offset % EXTENT_j;
  unsigned i = (offset / EXTENT_j) % EXTENT_i;
  size_t index = offset + (EXTENT_i - 1) * j + (1 - (size_t) EXTENT_j) * i;
  assert(index < NUM_ELEMENTS);
  ((bCuComplexElem *) dataOut)[index] = element;
}

// global pointers to the array load/store callbacks
typedef typename std::conditional<std::is_same<bElem, double>::value, 
                                  cufftCallbackLoadZ,
                                  cufftCallbackLoadC>::type ArrayLoadCallback;
typedef typename std::conditional<std::is_same<bElem, double>::value, 
                                  cufftCallbackStoreZ,
                                  cufftCallbackStoreC>::type ArrayStoreCallback;
__device__ ArrayLoadCallback array_load_callback_ptr = array_load_callback;
__device__ ArrayStoreCallback array_store_callback_ptr = array_store_callback;

/**
 * @brief time computation of 1D FFT in j-dimension of in_arr into out_arr
 * 
 * Uses callbacks to perform a single call to cufft (no data reordering)
 * 
 * @param in_arr[in] host-side input array, of rank DIM with extents EXTENT_n,...,EXTENT_i
 * @param out_arr[out] host-side output array of same shape as in_arr
 * @param direction[in] either CUFFT_FORWARD or CUFFT_INVERSE
 * @param warmup the number of warmups
 * @param iter the number of iters
 * @return the average number of seconds to compute the FFT
 */
double complex_to_complex_1d_j_fft_array_callback(bComplexElem *in_arr, bComplexElem *out_arr, int warmup, int iter, int direction = CUFFT_FORWARD)
{
  if(direction != CUFFT_FORWARD && direction != CUFFT_INVERSE) throw std::runtime_error("Unrecognized direction");
  // build cufft plan
  cufftHandle plan;
  int fft_rank = 1;
  int array_rank[1] = {EXTENT_j};
  int embed[1] = {EXTENT_j};
  int logical_stride = 1;
  int logical_dist_between_batches = EXTENT_j;
  int num_batches = NUM_ELEMENTS / array_rank[0];
  static_assert(std::is_same<bElem, float>::value || std::is_same<bElem, double>::value);
  cufftType type = std::is_same<bElem, float>::value ? CUFFT_C2C : CUFFT_Z2Z;
  cufftAlwaysCheck(cufftPlanMany(&plan, fft_rank, array_rank,
                           embed, logical_stride, logical_dist_between_batches,
                           embed, logical_stride, logical_dist_between_batches,
                           type, num_batches));
  
  // copy callback addresses from symbol memory
  ArrayLoadCallback array_load_cb;
  ArrayStoreCallback array_store_cb;
  gpuCheck(cudaMemcpyFromSymbol(
                          &array_load_cb, 
                          array_load_callback_ptr,
                          sizeof(array_load_cb)));
  gpuCheck(cudaMemcpyFromSymbol(
                          &array_store_cb, 
                          array_store_callback_ptr,
                          sizeof(array_store_cb)));
  // setup callbacks for cufft plan
  cufftXtCallbackType load_cb_type, store_cb_type;
  if(std::is_same<bElem, double>::value) {
    load_cb_type = CUFFT_CB_LD_COMPLEX_DOUBLE;
    store_cb_type = CUFFT_CB_ST_COMPLEX_DOUBLE;
  }
  else {
    load_cb_type = CUFFT_CB_LD_COMPLEX;
    store_cb_type = CUFFT_CB_ST_COMPLEX;
  }

  cufftAlwaysCheck(cufftXtSetCallback(plan, (void **)&array_load_cb,  load_cb_type, 0));
  cufftAlwaysCheck(cufftXtSetCallback(plan, (void **)&array_store_cb, store_cb_type, 0));

  // copy data to device
  bCuComplexElem *in_arr_dev, *out_arr_dev;
  constexpr size_t ARR_SIZE = sizeof(bComplexElem) * NUM_ELEMENTS;
  gpuCheck(cudaMalloc(&in_arr_dev, ARR_SIZE));
  gpuCheck(cudaMalloc(&out_arr_dev, ARR_SIZE));
  gpuCheck(cudaMemcpy(in_arr_dev, in_arr, ARR_SIZE, cudaMemcpyHostToDevice));

  // lambda function for timing
  auto compute_fft = [&plan](bCuComplexElem *in_arr, bCuComplexElem *out_arr, int direction) -> void {
    // launch cufft execution
    cufftAlwaysCheck(cufftXtExec(plan, in_arr, out_arr, direction));
  };
  // time function (and compute fft in process)
  double num_seconds = cutime_func([&compute_fft, &in_arr_dev, &out_arr_dev, &direction]() -> void {
    compute_fft(in_arr_dev, out_arr_dev, direction);
  }, warmup, iter);

  // copy data back from device
  gpuCheck(cudaMemcpy(out_arr, out_arr_dev, ARR_SIZE, cudaMemcpyDeviceToHost));

  // sanity check: inverse should match in_arr (up to scaling)
  nvtxRangePushA("inverse_fft_check");
  {
    // malloc space for the inverse computation
    bComplexElem *out_check_arr = zeroComplexArray({EXTENT});
    bCuComplexElem *out_check_arr_dev;
    size_t size = NUM_ELEMENTS * sizeof(bComplexElem);
    gpuCheck(cudaMalloc(&out_check_arr_dev, size));
    // perform the inverse computation
    int inverse_direction = (direction == CUFFT_FORWARD) ? CUFFT_INVERSE : CUFFT_FORWARD;
    compute_fft(out_arr_dev, out_check_arr_dev, inverse_direction);
    gpuCheck(cudaDeviceSynchronize());
    // copy check back to host and make sure it is correct
    gpuCheck(cudaMemcpy(out_check_arr, out_check_arr_dev, size, cudaMemcpyDeviceToHost));
    check_close(in_arr, out_check_arr, NUM_ELEMENTS, "cufft 1d j array callback inverse check failed", 1.0 / EXTENT_j);
    // free memroy
    gpuCheck(cudaFree(out_check_arr_dev));
    free(out_check_arr);
  }
  nvtxRangePop();

  // free memory
  gpuCheck(cudaFree(out_arr_dev));
  gpuCheck(cudaFree(in_arr_dev));
  cufftAlwaysCheck(cufftDestroy(plan));

  // return timing
  return num_seconds;
}

/**
 * @brief time computation of 1D FFT in j-dimension of in_arr into out_arr
 * 
 * Uses a transpose, then a FFT in the contiguous dimension, then a transpose
 * 
 * @param in_arr[in] host-side input array, of rank DIM with extents EXTENT_n,...,EXTENT_i
 * @param out_arr[out] host-side output array of same shape as in_arr
 * @param performTranspose[in] whether to perform a transpose before/after the FFT.
 * @param direction[in] either CUFFT_FORWARD or CUFFT_INVERSE
 * @param warmup the number of warmups
 * @param iter the number of iters
 * @return the average number of seconds to compute the FFT
 */
double complex_to_complex_1d_j_fft_array_transpose(bComplexElem *in_arr, bComplexElem *out_arr, int warmup, int iter, bool performTranspose = true, int direction = CUFFT_FORWARD)
{
  if(direction != CUFFT_FORWARD && direction != CUFFT_INVERSE) throw std::runtime_error("Unrecognized direction");
  // build cufft plan
  cufftHandle plan;
  int fft_rank = 1;
  int array_rank[1] = {EXTENT_j};
  int embed[1] = {EXTENT_j};
  int stride = 1;
  int distBetweenBatches = EXTENT_j;
  int numBatches = NUM_ELEMENTS / distBetweenBatches;
  static_assert(std::is_same<bElem, float>::value || std::is_same<bElem, double>::value);
  cufftType type = std::is_same<bElem, float>::value ? CUFFT_C2C : CUFFT_Z2Z;
  cufftAlwaysCheck(cufftPlanMany(&plan, fft_rank, array_rank,
                           embed, stride, distBetweenBatches,
                           embed, stride, distBetweenBatches,
                           type, numBatches));

  // copy data to device
  bCuComplexElem *in_arr_dev, *out_arr_dev, *intermed_arr_dev;
  constexpr size_t ARR_SIZE = sizeof(bComplexElem) * NUM_ELEMENTS;
  gpuCheck(cudaMalloc(&in_arr_dev, ARR_SIZE));
  gpuCheck(cudaMalloc(&out_arr_dev, ARR_SIZE));
  gpuCheck(cudaMalloc(&intermed_arr_dev, ARR_SIZE));
  gpuCheck(cudaMemcpy(in_arr_dev, in_arr, ARR_SIZE, cudaMemcpyHostToDevice));

  // lambda function for timing
  auto compute_fft = [&plan, performTranspose](bCuComplexElem *in_arr, bCuComplexElem *intermed_arr, bCuComplexElem *out_arr, int direction) -> void
  {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int blockSize = 128;
    int numBlocks = prop.multiProcessorCount * (prop.maxThreadsPerMultiProcessor / blockSize) * 4;
    // transpose in -> out 
    constexpr unsigned TileJ = 32, TileI = 8;
    if(performTranspose) {
      transpose_ij<TileJ, TileI><< <numBlocks, blockSize>> >(in_arr, out_arr, NUM_ELEMENTS / EXTENT_j / EXTENT_i, EXTENT_j, EXTENT_i);
    // fft out -> intermed
    cufftAlwaysCheck(cufftXtExec(plan, out_arr, intermed_arr, direction));
    // transpose intermed -> out
      transpose_ij<TileI, TileJ><<<numBlocks, blockSize>>>(
          intermed_arr, out_arr, NUM_ELEMENTS / EXTENT_j / EXTENT_i, EXTENT_i, EXTENT_j);
    } else {
      cufftAlwaysCheck(cufftXtExec(plan, in_arr, out_arr, direction));
    }
  };
  // time function (and compute fft in process)
  double num_seconds = cutime_func([&compute_fft, &in_arr_dev, &intermed_arr_dev, &out_arr_dev, &direction]() -> void {
    compute_fft(in_arr_dev, intermed_arr_dev, out_arr_dev, direction);
  }, warmup, iter);

  // copy data back from device
  gpuCheck(cudaMemcpy(out_arr, out_arr_dev, ARR_SIZE, cudaMemcpyDeviceToHost));

  // sanity check: inverse should match in_arr (up to scaling)
  nvtxRangePushA("inverse_fft_check");
  {
    // malloc space for the inverse computation
    bComplexElem *out_check_arr = zeroComplexArray({EXTENT});
    bCuComplexElem *out_check_arr_dev;
    size_t size = NUM_ELEMENTS * sizeof(bComplexElem);
    gpuCheck(cudaMalloc(&out_check_arr_dev, size));
    // perform the inverse computation
    int inverse_direction = (direction == CUFFT_FORWARD) ? CUFFT_INVERSE : CUFFT_FORWARD;
    compute_fft(out_arr_dev, intermed_arr_dev, out_check_arr_dev, inverse_direction);
    gpuCheck(cudaDeviceSynchronize());
    // copy check back to host and make sure it is correct
    gpuCheck(cudaMemcpy(out_check_arr, out_check_arr_dev, size, cudaMemcpyDeviceToHost));
    check_close(in_arr, out_check_arr, NUM_ELEMENTS, "cufft 1d j array transpose inverse check failed", 1.0 / EXTENT_j);
    // free memory
    gpuCheck(cudaFree(out_check_arr_dev));
    free(out_check_arr);
  }
  nvtxRangePop();

  // free memory
  gpuCheck(cudaFree(out_arr_dev));
  gpuCheck(cudaFree(intermed_arr_dev));
  gpuCheck(cudaFree(in_arr_dev));
  cufftAlwaysCheck(cufftDestroy(plan));

  // return timing
  return num_seconds;
}

/**
 * @brief time computation of 1D FFT in j-dimension of in_arr into out_arr
 * 
 * Uses bricks data layout
 * 
 * @param in_arr[in] host-side input array, of rank DIM with extents EXTENT_n,...,EXTENT_i
 * @param out_arr[out] host-side output array of same shape as in_arr
 * @param direction[in] either CUFFT_FORWARD or CUFFT_INVERSE
 * @param warmup number of warmup iterations
 * @param iter number of iterations
 * @return the average number of seconds to compute the FFT
 */
double complex_to_complex_1d_j_fft_brick(bComplexElem *in_arr, bComplexElem *out_arr, int warmup, int iter, int direction = CUFFT_FORWARD)
{
  if(direction != CUFFT_FORWARD && direction != CUFFT_INVERSE) throw std::runtime_error("Unrecognized direction");

  // useful typedefs for brick
  typedef Dim<BDIM_n,BDIM_m,BDIM_l,BDIM_k,BDIM_j,BDIM_i> BrickDims;
  typedef Dim<1> VFold;
  constexpr bool isComplex = true;
  typedef CommDims<false,false,false,false,false,false> NoComm;
  typedef Brick<BrickDims, VFold, isComplex, NoComm> ComplexBrick;

  // move arrays to host-side bricks
  unsigned *grid_ptr = nullptr;
  BrickInfo<RANK, NoComm> bInfo = init_grid<RANK, NoComm>(grid_ptr, {BRICK_EXTENT});
  BrickStorage bStorage = bInfo.allocate(2 * ComplexBrick::BRICKSIZE);
  ComplexBrick bIn(&bInfo, bStorage, 0),
               bOut(&bInfo, bStorage, ComplexBrick::BRICKSIZE);
  copyToBrick<RANK>({EXTENT}, std::vector<long>(RANK, 0), std::vector<long>(RANK, 0), in_arr, grid_ptr, bIn);

  // set up brick info in cuda
  BrickInfo<RANK, NoComm> *bInfo_dev;
  BrickInfo<RANK, NoComm> _bInfo_dev = movBrickInfo(bInfo, cudaMemcpyHostToDevice);
  gpuCheck(cudaMalloc(&bInfo_dev, sizeof(BrickInfo<RANK, NoComm>)));
  gpuCheck(cudaMemcpy(bInfo_dev, &_bInfo_dev, sizeof(BrickInfo<RANK, NoComm>), cudaMemcpyHostToDevice));
  // mov brick storage to cuda
  BrickStorage bStorage_dev = movBrickStorage(bStorage, cudaMemcpyHostToDevice);
  // set up brick in cuda
  ComplexBrick bIn_dev(bInfo_dev, bStorage_dev, 0);
  ComplexBrick bOut_dev(bInfo_dev, bStorage_dev, ComplexBrick::BRICKSIZE);
  // copy grids to device
  unsigned *grid_ptr_dev;
  size_t size = sizeof(unsigned) * NUM_BRICKS;
  gpuCheck(cudaMalloc(&grid_ptr_dev, size));
  gpuCheck(cudaMemcpy(grid_ptr_dev, grid_ptr, size, cudaMemcpyHostToDevice));

  // set up FFT for bricks
  typedef BricksCufftPlan<ComplexBrick, FourierType<ComplexToComplex, 1> > FFTPlanType;
  FFTPlanType plan({BRICK_EXTENT});
  plan.setup(bIn_dev, grid_ptr_dev, bOut_dev, grid_ptr_dev);

  auto compute_fft = [&plan] (bool direction = FFTPlanType::BRICKS_FFT_FORWARD) -> void { plan.launch(direction); };
  // time function (and compute fft in process)
  double num_seconds = cutime_func(compute_fft, warmup, iter);

  // copy data back from device
  gpuCheck(cudaMemcpy(bStorage.dat.get(), bStorage_dev.dat.get(), bInfo.nbricks * bStorage.step * sizeof(bElem),
                       cudaMemcpyDeviceToHost));
  // copy data back into array
  copyFromBrick<RANK>({EXTENT}, std::vector<long>(RANK, 0), std::vector<long>(RANK, 0), out_arr, grid_ptr, bOut);

  // sanity check: inverse should match in_arr (up to scaling)
  nvtxRangePushA("inverse_fft_check");
  {
    // malloc space for the inverse computation
    bComplexElem *out_check_arr = zeroComplexArray({EXTENT});
    bCuComplexElem *out_check_arr_dev;
    size_t size = NUM_ELEMENTS * sizeof(bComplexElem);
    gpuCheck(cudaMalloc(&out_check_arr_dev, size));
    // perform the inverse computation
    int inverse_direction = (direction == CUFFT_FORWARD) ? FFTPlanType::BRICKS_FFT_INVERSE : FFTPlanType::BRICKS_FFT_FORWARD;
    plan.setup(bOut_dev, grid_ptr_dev, bOut_dev, grid_ptr_dev);
    compute_fft(inverse_direction);
    gpuCheck(cudaDeviceSynchronize());
    // copy data back from device
    gpuCheck(cudaMemcpy(bStorage.dat.get(), bStorage_dev.dat.get(), bInfo.nbricks * bStorage.step * sizeof(bElem),
                        cudaMemcpyDeviceToHost));
    // copy data back into check-array
    copyFromBrick<RANK>({EXTENT}, std::vector<long>(RANK, 0), std::vector<long>(RANK, 0), out_check_arr, grid_ptr, bOut);
    check_close(in_arr, out_check_arr, NUM_ELEMENTS, "cufft 1d j brick inverse check failed", 1.0 / EXTENT_j);
    // free memroy
    gpuCheck(cudaFree(out_check_arr_dev));
    free(out_check_arr);
  }
  nvtxRangePop();

  // free memory
  gpuCheck(cudaFree(grid_ptr_dev));
  gpuCheck(cudaFree(_bInfo_dev.adj));
  gpuCheck(cudaFree(bInfo_dev));
  free(grid_ptr);

  // return timing
  return num_seconds;
}

namespace { // anonymous namespace
  constexpr int gcd(int a, int b) {
    if (b == 0)
    return a;
    return gcd(b, a % b);
  }
} // anonymous namespace

/**
 * @brief time computation of 1D FFT in j-dimension of in_arr into out_arr
 * 
 * Uses bricks data layout with a transpose before/after
 * 
 * @param in_arr[in] host-side input array, of rank DIM with extents EXTENT_n,...,EXTENT_i
 * @param out_arr[out] host-side output array of same shape as in_arr
 * @param direction[in] either CUFFT_FORWARD or CUFFT_INVERSE
 * @param warmup number of warmup iterations
 * @param iter number of iterations
 * @return the average number of seconds to compute the FFT
 */
double complex_to_complex_1d_j_fft_brick_transpose(bComplexElem *in_arr, bComplexElem *out_arr, int warmup, int iter, int direction = CUFFT_FORWARD)
{
  if(direction != CUFFT_FORWARD && direction != CUFFT_INVERSE) throw std::runtime_error("Unrecognized direction");

  // useful typedefs for brick
  typedef Dim<BDIM_n,BDIM_m,BDIM_l,BDIM_k,BDIM_j,BDIM_i> BrickDims;
  typedef Dim<BDIM_n,BDIM_m,BDIM_l,BDIM_k,BDIM_i,BDIM_j> BrickDimsTransposed;
  typedef Dim<1> VFold;
  constexpr bool isComplex = true;
  typedef CommDims<false,false,false,false,false,false> NoComm;
  typedef Brick<BrickDims, VFold, isComplex, NoComm> ComplexBrick;
  typedef Brick<BrickDimsTransposed, VFold, isComplex, NoComm> ComplexBrickTransposed;

  // move arrays to host-side bricks
  unsigned *grid_ptr = nullptr;
  BrickInfo<RANK, NoComm> bInfo = init_grid<RANK, NoComm>(grid_ptr, {BRICK_EXTENT});
  BrickStorage bInStorage = bInfo.allocate(ComplexBrick::BRICKSIZE),
               bOutStorage = bInfo.allocate(ComplexBrick::BRICKSIZE);
  ComplexBrick bIn(&bInfo, bInStorage, 0),
               bOut(&bInfo, bOutStorage, 0);
  copyToBrick<RANK>({EXTENT}, std::vector<long>(RANK, 0), std::vector<long>(RANK, 0), in_arr, grid_ptr, bIn);

  // set up brick info in cuda
  BrickInfo<RANK, NoComm> *bInfo_dev;
  BrickInfo<RANK, NoComm> _bInfo_dev = movBrickInfo(bInfo, cudaMemcpyHostToDevice);
  gpuCheck(cudaMalloc(&bInfo_dev, sizeof(BrickInfo<RANK, NoComm>)));
  gpuCheck(cudaMemcpy(bInfo_dev, &_bInfo_dev, sizeof(BrickInfo<RANK, NoComm>), cudaMemcpyHostToDevice));
  // mov brick storage to cuda
  BrickStorage bInStorage_dev = movBrickStorage(bInStorage, cudaMemcpyHostToDevice);
  BrickStorage bOutStorage_dev = movBrickStorage(bOutStorage, cudaMemcpyHostToDevice);
  // set up brick in cuda
  ComplexBrick bIn_dev(bInfo_dev, bInStorage_dev, 0);
  ComplexBrick bOut_dev(bInfo_dev, bOutStorage_dev, 0);
  // copy grids to device
  unsigned *grid_ptr_dev;
  size_t size = sizeof(unsigned) * NUM_BRICKS;
  gpuCheck(cudaMalloc(&grid_ptr_dev, size));
  gpuCheck(cudaMemcpy(grid_ptr_dev, grid_ptr, size, cudaMemcpyHostToDevice));

  // set up brickinfo for transposed bricks
  BrickInfo<RANK, NoComm> _bInfoTransposed_dev = movBrickInfo(bInfo, cudaMemcpyHostToDevice);
  BrickInfo<RANK, NoComm> *bInfoTransposed_dev;
  gpuCheck(cudaMalloc(&bInfoTransposed_dev, sizeof(BrickInfo<RANK, NoComm>)));
  gpuCheck(cudaMemcpy(bInfoTransposed_dev, &_bInfoTransposed_dev, sizeof(BrickInfo<RANK, NoComm>), cudaMemcpyHostToDevice));
  // set up transposed grid ptr
  unsigned *grid_ptr_transposed_dev;
  gpuCheck(cudaMalloc(&grid_ptr_transposed_dev, size));
  transpose_brick_info_ij_on_device(bInfo.nbricks, 64, bInfo_dev, bInfoTransposed_dev);
  constexpr size_t CollapsedDims = BRICK_EXTENT_n * BRICK_EXTENT_m * BRICK_EXTENT_l * BRICK_EXTENT_k;
  constexpr unsigned TileJ = 32, TileI = 8;
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  int blockSize = 128;
  int numBlocks = prop.multiProcessorCount * (prop.maxThreadsPerMultiProcessor / blockSize) * 4;
  transpose_ij<TileI, TileJ><< <numBlocks, blockSize>> >(grid_ptr_dev, grid_ptr_transposed_dev, CollapsedDims, BRICK_EXTENT_i, BRICK_EXTENT_j);
  gpuCheck(cudaDeviceSynchronize());
  gpuCheck(cudaPeekAtLastError());

  // set up intermediate brick on device
  BrickStorage intermedStorage_dev = movBrickStorage(bInStorage, cudaMemcpyHostToDevice);
  ComplexBrickTransposed bIntermed_dev(bInfoTransposed_dev, intermedStorage_dev, 0);
  // access bOutStorage with a transposed-brick layout
  ComplexBrickTransposed bOut_as_transposed_dev(bInfoTransposed_dev, bOutStorage_dev, 0);

  // set up FFT in 0-dimensional for bricks
  typedef BricksCufftPlan<ComplexBrickTransposed, FourierType<ComplexToComplex, 0> > FFTPlanType;
  std::array<size_t, RANK> transposed_grid_size = {BRICK_EXTENT_j, BRICK_EXTENT_i, BRICK_EXTENT_k,
                                                  BRICK_EXTENT_l, BRICK_EXTENT_m, BRICK_EXTENT_n};
  FFTPlanType plan(transposed_grid_size);
  plan.setup(bOut_as_transposed_dev, grid_ptr_transposed_dev, bIntermed_dev, grid_ptr_transposed_dev);
  auto compute_fft = [&plan,
                      &bInfo, &bIn_dev, &grid_ptr_dev, &bOut_dev,
                      &bOut_as_transposed_dev, &bIntermed_dev, &grid_ptr_transposed_dev]
  (bool direction = FFTPlanType::BRICKS_FFT_FORWARD) -> void {
    // transpose into out-brick
    transpose_brick_ij<< <bInfo.nbricks, 128>> >(bIn_dev, grid_ptr_dev,
                                                 bOut_as_transposed_dev, grid_ptr_transposed_dev,
                                                 BRICK_EXTENT_j, BRICK_EXTENT_i);
    // compute FFT on out -> intermed
    plan.launch(direction); 
    // transpose intermed back into out-brick
    transpose_brick_ij<< <bInfo.nbricks, 128>> >(bIntermed_dev, grid_ptr_transposed_dev,
                                                 bOut_dev, grid_ptr_dev,
                                                 BRICK_EXTENT_i, BRICK_EXTENT_j);
  };
  // time function (and compute fft in process)
  double num_seconds = cutime_func(compute_fft, warmup, iter);

  // copy data back from device
  gpuCheck(cudaMemcpy(bOutStorage.dat.get(), bOutStorage_dev.dat.get(), bInfo.nbricks * bOutStorage.step * sizeof(bElem),
                       cudaMemcpyDeviceToHost));
  // copy data back into array
  copyFromBrick<RANK>({EXTENT}, std::vector<long>(RANK, 0), std::vector<long>(RANK, 0), out_arr, grid_ptr, bOut);

  // sanity check: inverse should match in_arr (up to scaling)
  nvtxRangePushA("inverse_fft_check");
  {
    // malloc space for the inverse computation
    bComplexElem *out_check_arr = zeroComplexArray({EXTENT});
    bCuComplexElem *out_check_arr_dev;
    size_t size = NUM_ELEMENTS * sizeof(bComplexElem);
    gpuCheck(cudaMalloc(&out_check_arr_dev, size));
    // perform the inverse computation
    int inverse_direction = (direction == CUFFT_FORWARD) ? FFTPlanType::BRICKS_FFT_INVERSE : FFTPlanType::BRICKS_FFT_FORWARD;
    gpuCheck(cudaMemcpy(bInStorage_dev.dat.get(), bOutStorage_dev.dat.get(), bInfo.nbricks * bOutStorage.step * sizeof(bElem),
                         cudaMemcpyDeviceToDevice));
    compute_fft(inverse_direction);
    gpuCheck(cudaDeviceSynchronize());
    gpuCheck(cudaPeekAtLastError());
    // copy data back from device
    gpuCheck(cudaMemcpy(bOutStorage.dat.get(), bOutStorage_dev.dat.get(), bInfo.nbricks * bOutStorage.step * sizeof(bElem),
                        cudaMemcpyDeviceToHost));
    // copy data back into check-array
    copyFromBrick<RANK>({EXTENT}, std::vector<long>(RANK, 0), std::vector<long>(RANK, 0), out_check_arr, grid_ptr, bOut);
    check_close(in_arr, out_check_arr, NUM_ELEMENTS, "cufft 1d j brick transpose inverse check failed", 1.0 / EXTENT_j);
    // free memroy
    gpuCheck(cudaFree(out_check_arr_dev));
    free(out_check_arr);
  }
  nvtxRangePop();

  // free memory
  gpuCheck(cudaFree(grid_ptr_dev));
  gpuCheck(cudaFree(_bInfo_dev.adj));
  gpuCheck(cudaFree(bInfo_dev));
  gpuCheck(cudaFree(_bInfoTransposed_dev.adj));
  gpuCheck(cudaFree(bInfoTransposed_dev));
  gpuCheck(cudaFree(grid_ptr_transposed_dev));
  free(grid_ptr);

  // return timing
  return num_seconds;
}

int main(int argc, char **argv)
{
  if(argc > 4) throw std::runtime_error("too many arguments");
  bool run_1d_j_array = true,
       run_1d_j_brick = true;
  int warmup = CU_WARMUP, iter = CU_ITER;
  if(argc >= 2) warmup = std::stoi(argv[1]);
  if(argc >= 3) iter = std::stoi(argv[2]);
  if(argc >= 4) 
  {
    char to_run = argv[3][0];
    if(to_run == 'a') run_1d_j_brick = false;
    else if(to_run == 'b') run_1d_j_array = false;
    else throw std::runtime_error("Unrecognized argument, expected 'a' or 'b'");
  }

  CSVDataRecorder dataRecorder;
  std::string axis = "i";
  for(const auto &e : {EXTENT_i, EXTENT_j, EXTENT_k, EXTENT_m, EXTENT_n}) {
    dataRecorder.setDefaultValue("extent_" + axis, e);
    axis[0]++;
  }

  int cufft_major_version, cufft_minor_version, cufft_patch_level;
  cufftAlwaysCheck(cufftGetProperty(MAJOR_VERSION, &cufft_major_version));
  cufftAlwaysCheck(cufftGetProperty(MINOR_VERSION, &cufft_minor_version));
  cufftAlwaysCheck(cufftGetProperty(PATCH_LEVEL, &cufft_patch_level));
  std::cout << "cufft Version: "
            << cufft_major_version << "."
            << cufft_minor_version << "."
            << cufft_patch_level << std::endl;
  std::cout << "Array Size: "
            << std::setw(2) << EXTENT_n << " x "
            << std::setw(2) << EXTENT_m << " x "
            << std::setw(2) << EXTENT_l << " x "
            << std::setw(2) << EXTENT_k << " x "
            << std::setw(2) << EXTENT_j << " x "
            << std::setw(2) << EXTENT_i << "\n"
            << "Brick Size: "
            << std::setw(2) << BDIM_n << " x "
            << std::setw(2) << BDIM_m << " x "
            << std::setw(2) << BDIM_l << " x "
            << std::setw(2) << BDIM_k << " x "
            << std::setw(2) << BDIM_j << " x "
            << std::setw(2) << BDIM_i << "\n"
            <<  "WARMUP: " << warmup << ", ITERATIONS: " << iter
            << std::endl;

  // set up arrays
  bComplexElem *in_arr = randomComplexArray({EXTENT}),
               *out_arr = zeroComplexArray({EXTENT}),
               *out_check_arr = zeroComplexArray({EXTENT});
  
  // print table
  int colWidth = 30;
  std::cout << std::setw(colWidth) << "method"
            << std::setw(colWidth) << "time(ms)" 
            << std::endl;
  // time cufft for arrays
  dataRecorder.setDefaultValue("Layout", "array");
  if(run_1d_j_array)
  {
    dataRecorder.newRow();
    nvtxRangePushA("cufft_1d_j_array_callback");
    double cufft_1d_j_array_callback_num_seconds = complex_to_complex_1d_j_fft_array_callback(in_arr, out_check_arr, warmup, iter);
    std::cout << std::setw(colWidth) << "cufft_1d_j_array_callback "
              << std::setw(colWidth) << 1000 * cufft_1d_j_array_callback_num_seconds
              << std::endl;
    nvtxRangePop();
    dataRecorder.record("transpose", "false");
    dataRecorder.record("callback", "true");
    dataRecorder.record("avgtime(s)", cufft_1d_j_array_callback_num_seconds);

    dataRecorder.newRow();
    nvtxRangePushA("cufft_1d_j_array_transpose");
    double cufft_1d_j_array_transpose_num_seconds = complex_to_complex_1d_j_fft_array_transpose(in_arr, out_arr, warmup, iter);
    std::cout << std::setw(colWidth) << "cufft_1d_j_array_transpose "
              << std::setw(colWidth) << 1000 * cufft_1d_j_array_transpose_num_seconds
              << std::endl;
    nvtxRangePop();
    check_close(out_check_arr, out_arr, NUM_ELEMENTS, "Mismatch between cufft_1d_j_array_callback and cufft_1d_j_array_transpose");
    dataRecorder.record("transpose", "true");
    dataRecorder.record("callback", "false");
    dataRecorder.record("avgtime(s)", cufft_1d_j_array_transpose_num_seconds);


    dataRecorder.newRow();
    nvtxRangePushA("cufft_1d_j_array_excluding_transpose");
    double cufft_1d_j_array_excluding_transpose_num_seconds = complex_to_complex_1d_j_fft_array_transpose(in_arr, out_arr, warmup, iter, false);
    std::cout << std::setw(colWidth) << "cufft_1d_j_array_excluding_transpose "
              << std::setw(colWidth) << 1000 * cufft_1d_j_array_excluding_transpose_num_seconds
              << std::endl;
    nvtxRangePop();

    dataRecorder.newRow();
    nvtxRangePushA("cufft_1d_j_array");
    double cufft_1d_j_array_num_seconds = complex_to_complex_1d_j_fft_array(in_arr, out_arr, warmup, iter);
    std::cout << std::setw(colWidth) << "cufft_1d_j_array "
              << std::setw(colWidth) << 1000 * cufft_1d_j_array_num_seconds
              << std::endl;
    nvtxRangePop();
    check_close(out_check_arr, out_arr, NUM_ELEMENTS, "Mismatch between cufft_1d_j_array and cufft_1d_j_array_transpose");
    dataRecorder.record("transpose", "false");
    dataRecorder.record("callback", "false");
    dataRecorder.record("avgtime(s)", cufft_1d_j_array_num_seconds);
  }

  // time cufft for bricks
  axis = "i";
  for(const auto &e : {BDIM_i, BDIM_j, BDIM_k, BDIM_m, BDIM_n}) {
    dataRecorder.setDefaultValue("brickdim_" + axis, e);
    axis[0]++;
  }
  dataRecorder.setDefaultValue("Layout", "bricks");
  if(run_1d_j_brick)
  {
    // re-zero out out_arr
    #pragma omp parallel
    for(unsigned i = 0; i < NUM_ELEMENTS; ++i) out_arr[i] = 0.0;
    dataRecorder.newRow();
    nvtxRangePushA("cufft_1d_j_brick_transpose");
    double cufft_1d_j_brick_transpose_num_seconds = complex_to_complex_1d_j_fft_brick_transpose(in_arr, out_arr, warmup, iter);
    std::cout << std::setw(colWidth) << "cufft_1d_j_brick_transpose"
              << std::setw(colWidth) << 1000 * cufft_1d_j_brick_transpose_num_seconds
              << std::endl;
    nvtxRangePop();
    dataRecorder.record("transpose", "true");
    dataRecorder.record("callback", "true");
    dataRecorder.record("avgtime(s)", cufft_1d_j_brick_transpose_num_seconds);

    // run correctness check
    if(run_1d_j_array)
    {
      check_close(out_check_arr, out_arr, NUM_ELEMENTS, "Mismatch between 1d_j_array and 1d_j_brick_transpose");
    }
    // re-zero out out_arr
    #pragma omp parallel
    for(unsigned i = 0; i < NUM_ELEMENTS; ++i) out_arr[i] = 0.0;
    dataRecorder.newRow();
    nvtxRangePushA("cufft_1d_j_brick");
    double cufft_1d_j_brick_num_seconds = complex_to_complex_1d_j_fft_brick(in_arr, out_arr, warmup, iter);
    std::cout << std::setw(colWidth) << "cufft_1d_j_brick "
              << std::setw(colWidth) << 1000 * cufft_1d_j_brick_num_seconds
              << std::endl;
    nvtxRangePop();
    // run correctness check
    if(run_1d_j_array)
    {
      check_close(out_check_arr, out_arr, NUM_ELEMENTS, "Mismatch between 1d_j_array and 1d_j_brick");
    }
    dataRecorder.record("transpose", "false");
    dataRecorder.record("callback", "true");
    dataRecorder.record("avgtime(s)", cufft_1d_j_brick_num_seconds);
  }

  // free memory
  free(out_arr);
  free(in_arr);

  // TODO: MAKE FILENAME CLI PARAMETER
  dataRecorder.writeToFile("fft_results.csv", true);
}