#include <cmath>
#include <iostream>
#include <iomanip>
#include "nvToolsExt.h"
#include "transpose-cu.h"
#include "fft.h"
#include "stencils/gene-6d.h"

// NOTE: none of the arrays use padding or ghost-zones

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
  cufftCheck(cufftPlanMany(&plan, fft_rank, array_rank,
                           embed, stride, distBetweenBatches,
                           embed, stride, distBetweenBatches,
                           type, numBatches));

  // copy data to device
  bCuComplexElem *in_arr_dev, *out_arr_dev;
  constexpr size_t ARR_SIZE = sizeof(bComplexElem) * NUM_ELEMENTS;
  cudaCheck(cudaMalloc(&in_arr_dev, ARR_SIZE));
  cudaCheck(cudaMalloc(&out_arr_dev, ARR_SIZE));
  cudaCheck(cudaMemcpy(in_arr_dev, in_arr, ARR_SIZE, cudaMemcpyHostToDevice));

  // lambda function for timing
  auto compute_fft = [&plan](bCuComplexElem *in_arr, bCuComplexElem *out_arr, int direction) -> void 
  {
    // launch cufft execution
    for(unsigned i = 0; i < EXTENT_i; ++i)
    {
      cufftCheck(cufftXtExec(plan, in_arr + i, out_arr + i, direction));
    }
  };
  // time function (and compute fft in process)
  double num_seconds = cutime_func([&compute_fft, &in_arr_dev, &out_arr_dev, &direction]() -> void {
    compute_fft(in_arr_dev, out_arr_dev, direction);
  }, warmup, iter);

  // copy data back from device
  cudaCheck(cudaMemcpy(out_arr, out_arr_dev, ARR_SIZE, cudaMemcpyDeviceToHost));

  // sanity check: inverse should match in_arr (up to scaling)
  nvtxRangePushA("inverse_fft_check");
  {
    // malloc space for the inverse computation
    bComplexElem *out_check_arr = zeroComplexArray({EXTENT});
    bCuComplexElem *out_check_arr_dev;
    size_t size = NUM_ELEMENTS * sizeof(bComplexElem);
    cudaCheck(cudaMalloc(&out_check_arr_dev, size));
    // perform the inverse computation
    int inverse_direction = (direction == CUFFT_FORWARD) ? CUFFT_INVERSE : CUFFT_FORWARD;
    compute_fft(out_arr_dev, out_check_arr_dev, inverse_direction);
    cudaCheck(cudaDeviceSynchronize());
    // copy check back to host and make sure it is correct
    cudaCheck(cudaMemcpy(out_check_arr, out_check_arr_dev, size, cudaMemcpyDeviceToHost));
    check_close(in_arr, out_check_arr, NUM_ELEMENTS, "cufft 1d j array inverse check failed", 1.0 / EXTENT_j);
    // free memroy
    cudaCheck(cudaFree(out_check_arr_dev));
    free(out_check_arr);
  }
  nvtxRangePop();

  // free memory
  cudaCheck(cudaFree(out_arr_dev));
  cudaCheck(cudaFree(in_arr_dev));
  cufftCheck(cufftDestroy(plan));

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
 * @param direction[in] either CUFFT_FORWARD or CUFFT_INVERSE
 * @param warmup the number of warmups
 * @param iter the number of iters
 * @return the average number of seconds to compute the FFT
 */
double complex_to_complex_1d_j_fft_array_transpose(bComplexElem *in_arr, bComplexElem *out_arr, int warmup, int iter, int direction = CUFFT_FORWARD)
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
  cufftCheck(cufftPlanMany(&plan, fft_rank, array_rank,
                           embed, stride, distBetweenBatches,
                           embed, stride, distBetweenBatches,
                           type, numBatches));

  // copy data to device
  bCuComplexElem *in_arr_dev, *out_arr_dev, *intermed_arr_dev;
  constexpr size_t ARR_SIZE = sizeof(bComplexElem) * NUM_ELEMENTS;
  cudaCheck(cudaMalloc(&in_arr_dev, ARR_SIZE));
  cudaCheck(cudaMalloc(&out_arr_dev, ARR_SIZE));
  cudaCheck(cudaMalloc(&intermed_arr_dev, ARR_SIZE));
  cudaCheck(cudaMemcpy(in_arr_dev, in_arr, ARR_SIZE, cudaMemcpyHostToDevice));

  // lambda function for timing
  auto compute_fft = [&plan](const bCuComplexElem *in_arr, bCuComplexElem *intermed_arr, bCuComplexElem *out_arr, int direction) -> void 
  {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int blockSize = 128;
    int numBlocks = prop.multiProcessorCount * (prop.maxThreadsPerMultiProcessor / blockSize) * 4;
    // transpose in -> out 
    constexpr unsigned TileJ = 32, TileI = 8;
    transpose_ij<TileJ, TileI><< <numBlocks, blockSize>> >(in_arr, out_arr, NUM_ELEMENTS / EXTENT_j / EXTENT_i, EXTENT_j, EXTENT_i);
    // fft out -> intermed
    cufftCheck(cufftXtExec(plan, out_arr, intermed_arr, direction));
    // transpose intermed -> out
    transpose_ij<TileI, TileJ><< <numBlocks, blockSize>> >(intermed_arr, out_arr, NUM_ELEMENTS / EXTENT_j / EXTENT_i, EXTENT_i, EXTENT_j);
  };
  // time function (and compute fft in process)
  double num_seconds = cutime_func([&compute_fft, &in_arr_dev, &intermed_arr_dev, &out_arr_dev, &direction]() -> void {
    compute_fft(in_arr_dev, intermed_arr_dev, out_arr_dev, direction);
  }, warmup, iter);

  // copy data back from device
  cudaCheck(cudaMemcpy(out_arr, out_arr_dev, ARR_SIZE, cudaMemcpyDeviceToHost));

  // sanity check: inverse should match in_arr (up to scaling)
  nvtxRangePushA("inverse_fft_check");
  {
    // malloc space for the inverse computation
    bComplexElem *out_check_arr = zeroComplexArray({EXTENT});
    bCuComplexElem *out_check_arr_dev;
    size_t size = NUM_ELEMENTS * sizeof(bComplexElem);
    cudaCheck(cudaMalloc(&out_check_arr_dev, size));
    // perform the inverse computation
    int inverse_direction = (direction == CUFFT_FORWARD) ? CUFFT_INVERSE : CUFFT_FORWARD;
    compute_fft(out_arr_dev, intermed_arr_dev, out_check_arr_dev, inverse_direction);
    cudaCheck(cudaDeviceSynchronize());
    // copy check back to host and make sure it is correct
    cudaCheck(cudaMemcpy(out_check_arr, out_check_arr_dev, size, cudaMemcpyDeviceToHost));
    check_close(in_arr, out_check_arr, NUM_ELEMENTS, "cufft 1d j array transpose inverse check failed", 1.0 / EXTENT_j);
    // free memory
    cudaCheck(cudaFree(out_check_arr_dev));
    free(out_check_arr);
  }
  nvtxRangePop();

  // free memory
  cudaCheck(cudaFree(out_arr_dev));
  cudaCheck(cudaFree(intermed_arr_dev));
  cudaCheck(cudaFree(in_arr_dev));
  cufftCheck(cufftDestroy(plan));

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
  BrickInfo<DIM, NoComm> bInfo = init_grid<DIM, NoComm>(grid_ptr, {BRICK_EXTENT});
  BrickStorage bStorage = bInfo.allocate(2 * ComplexBrick::BRICKSIZE);
  ComplexBrick bIn(&bInfo, bStorage, 0),
               bOut(&bInfo, bStorage, ComplexBrick::BRICKSIZE);
  copyToBrick<DIM>({EXTENT}, std::vector<long>(DIM, 0), std::vector<long>(DIM, 0), in_arr, grid_ptr, bIn);

  // set up brick info in cuda
  BrickInfo<DIM, NoComm> *bInfo_dev;
  BrickInfo<DIM, NoComm> _bInfo_dev = movBrickInfo(bInfo, cudaMemcpyHostToDevice);
  cudaCheck(cudaMalloc(&bInfo_dev, sizeof(BrickInfo<DIM, NoComm>)));
  cudaCheck(cudaMemcpy(bInfo_dev, &_bInfo_dev, sizeof(BrickInfo<DIM, NoComm>), cudaMemcpyHostToDevice));
  // mov brick storage to cuda
  BrickStorage bStorage_dev = movBrickStorage(bStorage, cudaMemcpyHostToDevice);
  // set up brick in cuda
  ComplexBrick bIn_dev(bInfo_dev, bStorage_dev, 0);
  ComplexBrick bOut_dev(bInfo_dev, bStorage_dev, ComplexBrick::BRICKSIZE);
  // copy grids to device
  unsigned *grid_ptr_dev;
  size_t size = sizeof(unsigned) * NUM_BRICKS;
  cudaCheck(cudaMalloc(&grid_ptr_dev, size));
  cudaCheck(cudaMemcpy(grid_ptr_dev, grid_ptr, size, cudaMemcpyHostToDevice));

  // set up FFT for bricks
  typedef BricksCufftPlan<ComplexBrick, FourierType<ComplexToComplex, 1> > FFTPlanType;
  FFTPlanType plan({BRICK_EXTENT});
  plan.setup(bIn_dev, grid_ptr_dev, bOut_dev, grid_ptr_dev);

  auto compute_fft = [&plan] (bool direction = FFTPlanType::BRICKS_FFT_FORWARD) -> void { plan.launch(direction); };
  // time function (and compute fft in process)
  double num_seconds = cutime_func(compute_fft, warmup, iter);

  // copy data back from device
  cudaCheck(cudaMemcpy(bStorage.dat.get(), bStorage_dev.dat.get(), bInfo.nbricks * bStorage.step * sizeof(bElem),
                       cudaMemcpyDeviceToHost));
  // copy data back into array
  copyFromBrick<DIM>({EXTENT}, std::vector<long>(DIM, 0), std::vector<long>(DIM, 0), out_arr, grid_ptr, bOut);

  // sanity check: inverse should match in_arr (up to scaling)
  nvtxRangePushA("inverse_fft_check");
  {
    // malloc space for the inverse computation
    bComplexElem *out_check_arr = zeroComplexArray({EXTENT});
    bCuComplexElem *out_check_arr_dev;
    size_t size = NUM_ELEMENTS * sizeof(bComplexElem);
    cudaCheck(cudaMalloc(&out_check_arr_dev, size));
    // perform the inverse computation
    int inverse_direction = (direction == CUFFT_FORWARD) ? FFTPlanType::BRICKS_FFT_INVERSE : FFTPlanType::BRICKS_FFT_FORWARD;
    plan.setup(bOut_dev, grid_ptr_dev, bOut_dev, grid_ptr_dev);
    compute_fft(inverse_direction);
    cudaCheck(cudaDeviceSynchronize());
    // copy data back from device
    cudaCheck(cudaMemcpy(bStorage.dat.get(), bStorage_dev.dat.get(), bInfo.nbricks * bStorage.step * sizeof(bElem),
                        cudaMemcpyDeviceToHost));
    // copy data back into check-array
    copyFromBrick<DIM>({EXTENT}, std::vector<long>(DIM, 0), std::vector<long>(DIM, 0), out_check_arr, grid_ptr, bOut);
    check_close(in_arr, out_check_arr, NUM_ELEMENTS, "cufft 1d j brick inverse check failed", 1.0 / EXTENT_j);
    // free memroy
    cudaCheck(cudaFree(out_check_arr_dev));
    free(out_check_arr);
  }
  nvtxRangePop();

  // free memory
  cudaCheck(cudaFree(grid_ptr_dev));
  cudaCheck(cudaFree(_bInfo_dev.adj));
  cudaCheck(cudaFree(bInfo_dev));
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
  BrickInfo<DIM, NoComm> bInfo = init_grid<DIM, NoComm>(grid_ptr, {BRICK_EXTENT});
  BrickStorage bInStorage = bInfo.allocate(ComplexBrick::BRICKSIZE),
               bOutStorage = bInfo.allocate(ComplexBrick::BRICKSIZE);
  ComplexBrick bIn(&bInfo, bInStorage, 0),
               bOut(&bInfo, bOutStorage, 0);
  copyToBrick<DIM>({EXTENT}, std::vector<long>(DIM, 0), std::vector<long>(DIM, 0), in_arr, grid_ptr, bIn);

  // set up brick info in cuda
  BrickInfo<DIM, NoComm> *bInfo_dev;
  BrickInfo<DIM, NoComm> _bInfo_dev = movBrickInfo(bInfo, cudaMemcpyHostToDevice);
  cudaCheck(cudaMalloc(&bInfo_dev, sizeof(BrickInfo<DIM, NoComm>)));
  cudaCheck(cudaMemcpy(bInfo_dev, &_bInfo_dev, sizeof(BrickInfo<DIM, NoComm>), cudaMemcpyHostToDevice));
  // mov brick storage to cuda
  BrickStorage bInStorage_dev = movBrickStorage(bInStorage, cudaMemcpyHostToDevice);
  BrickStorage bOutStorage_dev = movBrickStorage(bOutStorage, cudaMemcpyHostToDevice);
  // set up brick in cuda
  ComplexBrick bIn_dev(bInfo_dev, bInStorage_dev, 0);
  ComplexBrick bOut_dev(bInfo_dev, bOutStorage_dev, 0);
  // copy grids to device
  unsigned *grid_ptr_dev;
  size_t size = sizeof(unsigned) * NUM_BRICKS;
  cudaCheck(cudaMalloc(&grid_ptr_dev, size));
  cudaCheck(cudaMemcpy(grid_ptr_dev, grid_ptr, size, cudaMemcpyHostToDevice));

  // set up brickinfo for transposed bricks
  BrickInfo<DIM, NoComm> _bInfoTransposed_dev = movBrickInfo(bInfo, cudaMemcpyHostToDevice);
  BrickInfo<DIM, NoComm> *bInfoTransposed_dev;
  cudaCheck(cudaMalloc(&bInfoTransposed_dev, sizeof(BrickInfo<DIM, NoComm>)));
  cudaCheck(cudaMemcpy(bInfoTransposed_dev, &_bInfoTransposed_dev, sizeof(BrickInfo<DIM, NoComm>), cudaMemcpyHostToDevice));
  // set up transposed grid ptr
  unsigned *grid_ptr_transposed_dev;
  cudaCheck(cudaMalloc(&grid_ptr_transposed_dev, size));
  transpose_brick_info_ij_on_device(bInfo.nbricks, 64, bInfo_dev, bInfoTransposed_dev);
  constexpr size_t CollapsedDims = BRICK_EXTENT_n * BRICK_EXTENT_m * BRICK_EXTENT_l * BRICK_EXTENT_k;
  constexpr unsigned TileJ = 32, TileI = 8;
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  int blockSize = 128;
  int numBlocks = prop.multiProcessorCount * (prop.maxThreadsPerMultiProcessor / blockSize) * 4;
  transpose_ij<TileI, TileJ><< <numBlocks, blockSize>> >(grid_ptr_dev, grid_ptr_transposed_dev, CollapsedDims, BRICK_EXTENT_i, BRICK_EXTENT_j);
  cudaCheck(cudaDeviceSynchronize());
  cudaCheck(cudaPeekAtLastError());

  // set up intermediate brick on device
  BrickStorage intermedStorage_dev = movBrickStorage(bInStorage, cudaMemcpyHostToDevice);
  ComplexBrickTransposed bIntermed_dev(bInfoTransposed_dev, intermedStorage_dev, 0);
  // access bOutStorage with a transposed-brick layout
  ComplexBrickTransposed bOut_as_transposed_dev(bInfoTransposed_dev, bOutStorage_dev, 0);

  // set up FFT in 0-dimensional for bricks
  typedef BricksCufftPlan<ComplexBrickTransposed, FourierType<ComplexToComplex, 0> > FFTPlanType;
  std::array<size_t, DIM> transposed_grid_size = {BRICK_EXTENT_j, BRICK_EXTENT_i, BRICK_EXTENT_k,
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
  cudaCheck(cudaMemcpy(bOutStorage.dat.get(), bOutStorage_dev.dat.get(), bInfo.nbricks * bOutStorage.step * sizeof(bElem),
                       cudaMemcpyDeviceToHost));
  // copy data back into array
  copyFromBrick<DIM>({EXTENT}, std::vector<long>(DIM, 0), std::vector<long>(DIM, 0), out_arr, grid_ptr, bOut);

  // sanity check: inverse should match in_arr (up to scaling)
  nvtxRangePushA("inverse_fft_check");
  {
    // malloc space for the inverse computation
    bComplexElem *out_check_arr = zeroComplexArray({EXTENT});
    bCuComplexElem *out_check_arr_dev;
    size_t size = NUM_ELEMENTS * sizeof(bComplexElem);
    cudaCheck(cudaMalloc(&out_check_arr_dev, size));
    // perform the inverse computation
    int inverse_direction = (direction == CUFFT_FORWARD) ? FFTPlanType::BRICKS_FFT_INVERSE : FFTPlanType::BRICKS_FFT_FORWARD;
    cudaCheck(cudaMemcpy(bInStorage_dev.dat.get(), bOutStorage_dev.dat.get(), bInfo.nbricks * bOutStorage.step * sizeof(bElem),
                         cudaMemcpyDeviceToDevice));
    compute_fft(inverse_direction);
    cudaCheck(cudaDeviceSynchronize());
    cudaCheck(cudaPeekAtLastError());
    // copy data back from device
    cudaCheck(cudaMemcpy(bOutStorage.dat.get(), bOutStorage_dev.dat.get(), bInfo.nbricks * bOutStorage.step * sizeof(bElem),
                        cudaMemcpyDeviceToHost));
    // copy data back into check-array
    copyFromBrick<DIM>({EXTENT}, std::vector<long>(DIM, 0), std::vector<long>(DIM, 0), out_check_arr, grid_ptr, bOut);
    check_close(in_arr, out_check_arr, NUM_ELEMENTS, "cufft 1d j brick transpose inverse check failed", 1.0 / EXTENT_j);
    // free memroy
    cudaCheck(cudaFree(out_check_arr_dev));
    free(out_check_arr);
  }
  nvtxRangePop();

  // free memory
  cudaCheck(cudaFree(grid_ptr_dev));
  cudaCheck(cudaFree(_bInfo_dev.adj));
  cudaCheck(cudaFree(bInfo_dev));
  cudaCheck(cudaFree(_bInfoTransposed_dev.adj));
  cudaCheck(cudaFree(bInfoTransposed_dev));
  cudaCheck(cudaFree(grid_ptr_transposed_dev));
  free(grid_ptr);

  // return timing
  return num_seconds;
}

/**
 * @brief time computation of 1D FFT over collapsed  i-j-dimensions of in_arr into out_arr
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
double complex_to_complex_1d_collaped_ij_fft_brick(bComplexElem *in_arr, bComplexElem *out_arr, int warmup, int iter, int direction = CUFFT_FORWARD)
{
  if(direction != CUFFT_FORWARD && direction != CUFFT_INVERSE) throw std::runtime_error("Unrecognized direction");

  // useful typedefs for brick
  typedef Dim<BDIM_n,BDIM_m,BDIM_l,BDIM_k,BDIM_j * BDIM_i> BrickDims;
  typedef Dim<1> VFold;
  constexpr bool isComplex = true;
  typedef CommDims<false,false,false,false,false> NoComm;
  typedef Brick<BrickDims, VFold, isComplex, NoComm> ComplexBrick;

  // move arrays to host-side bricks
  unsigned *grid_ptr = nullptr;
  const std::vector<long> brick_extent = {(EXTENT_j * BDIM_i) / (BDIM_i * BDIM_j), 
                                          ((EXTENT_i / BDIM_i) * EXTENT_k) / BDIM_k,
                                          BRICK_EXTENT_l,
                                          BRICK_EXTENT_m,
                                          BRICK_EXTENT_n};
  BrickInfo<DIM-1, NoComm> bInfo = init_grid<DIM-1, NoComm>(grid_ptr, brick_extent);
  BrickStorage bInStorage = bInfo.allocate(ComplexBrick::BRICKSIZE),
               bOutStorage = bInfo.allocate(ComplexBrick::BRICKSIZE);
  ComplexBrick bIn(&bInfo, bInStorage, 0),
               bOut(&bInfo, bOutStorage, 0);
  const std::vector<long> arr_extent = {BDIM_i * EXTENT_j, EXTENT_i / BDIM_i * EXTENT_k, EXTENT_l, EXTENT_m, EXTENT_n};
  copyToBrick<DIM-1>(arr_extent, std::vector<long>(DIM-1, 0), std::vector<long>(DIM-1, 0), in_arr, grid_ptr, bIn);

  // set up brick info in cuda
  BrickInfo<DIM-1, NoComm> *bInfo_dev;
  BrickInfo<DIM-1, NoComm> _bInfo_dev = movBrickInfo(bInfo, cudaMemcpyHostToDevice);
  cudaCheck(cudaMalloc(&bInfo_dev, sizeof(BrickInfo<DIM-1, NoComm>)));
  cudaCheck(cudaMemcpy(bInfo_dev, &_bInfo_dev, sizeof(BrickInfo<DIM-1, NoComm>), cudaMemcpyHostToDevice));
  // mov brick storage to cuda
  BrickStorage bInStorage_dev = movBrickStorage(bInStorage, cudaMemcpyHostToDevice),
               bOutStorage_dev = movBrickStorage(bOutStorage, cudaMemcpyHostToDevice);
  // set up brick in cuda
  ComplexBrick bIn_dev(bInfo_dev, bInStorage_dev, 0);
  ComplexBrick bOut_dev(bInfo_dev, bOutStorage_dev, 0);
  // copy grids to device
  unsigned *grid_ptr_dev;
  size_t size = sizeof(unsigned) * NUM_BRICKS;
  cudaCheck(cudaMalloc(&grid_ptr_dev, size));
  cudaCheck(cudaMemcpy(grid_ptr_dev, grid_ptr, size, cudaMemcpyHostToDevice));

  // set up FFT for bricks
  typedef BricksCufftPlan<ComplexBrick, FourierType<ComplexToComplex, 0> > FFTPlanType;
  std::array<size_t, DIM-1> brick_extent_as_arr;
  std::copy(brick_extent.begin(), brick_extent.end(), brick_extent_as_arr.data());
  FFTPlanType plan(brick_extent_as_arr);
  plan.setup(bIn_dev, grid_ptr_dev, bOut_dev, grid_ptr_dev);

  auto compute_fft = [&plan] (bool direction = FFTPlanType::BRICKS_FFT_FORWARD) -> void { plan.launch(direction); };
  // time function (and compute fft in process)
  double num_seconds = cutime_func(compute_fft, warmup, iter);

  // copy data back from device
  cudaCheck(cudaMemcpy(bOutStorage.dat.get(), bOutStorage_dev.dat.get(), bInfo.nbricks * bOutStorage.step * sizeof(bElem),
                       cudaMemcpyDeviceToHost));
  // copy data back into array
  copyFromBrick<DIM-1>(arr_extent, std::vector<long>(DIM-1, 0), std::vector<long>(DIM-1, 0), out_arr, grid_ptr, bOut);

  // sanity check: inverse should match in_arr (up to scaling)
  nvtxRangePushA("inverse_fft_check");
  {
    // malloc space for the inverse computation
    bComplexElem *out_check_arr = zeroComplexArray({EXTENT});
    bCuComplexElem *out_check_arr_dev;
    size_t size = NUM_ELEMENTS * sizeof(bComplexElem);
    cudaCheck(cudaMalloc(&out_check_arr_dev, size));
    // perform the inverse computation
    int inverse_direction = (direction == CUFFT_FORWARD) ? FFTPlanType::BRICKS_FFT_INVERSE : FFTPlanType::BRICKS_FFT_FORWARD;
    plan.setup(bOut_dev, grid_ptr_dev, bOut_dev, grid_ptr_dev);
    compute_fft(inverse_direction);
    cudaCheck(cudaDeviceSynchronize());
    cudaCheck(cudaPeekAtLastError());
    // copy data back from device
    cudaCheck(cudaMemcpy(bOutStorage.dat.get(), bOutStorage_dev.dat.get(), bInfo.nbricks * bOutStorage.step * sizeof(bElem),
                        cudaMemcpyDeviceToHost));
    // copy data back into check-array
    copyFromBrick<DIM-1>(arr_extent, std::vector<long>(DIM-1, 0), std::vector<long>(DIM-1, 0), out_check_arr, grid_ptr, bOut);
    check_close(in_arr, out_check_arr, NUM_ELEMENTS, "cufft 1d collaped i-j brick inverse check failed", 1.0 / arr_extent[0]);
    // free memroy
    cudaCheck(cudaFree(out_check_arr_dev));
    free(out_check_arr);
  }
  nvtxRangePop();

  // free memory
  cudaCheck(cudaFree(grid_ptr_dev));
  cudaCheck(cudaFree(_bInfo_dev.adj));
  cudaCheck(cudaFree(bInfo_dev));
  free(grid_ptr);

  // return timing
  return num_seconds;
}

int main(int argc, char **argv)
{
  if(argc > 4) throw std::runtime_error("too many arguments");
  bool run_1d_j_array = true,
       run_1d_j_brick = true,
       run_1d_collapsed_ij_brick = false;
  int warmup = CU_WARMUP, iter = CU_ITER;
  if(argc >= 2) warmup = std::stoi(argv[1]);
  if(argc >= 3) iter = std::stoi(argv[2]);
  if(argc >= 4) 
  {
    char to_run = argv[3][0];
    if(to_run == 'a') run_1d_j_brick = false;
    else if(to_run == 'b') run_1d_j_array = false;
    else if(to_run == 'c') {
      run_1d_j_array = false;
      run_1d_j_brick = false;
      run_1d_collapsed_ij_brick = true;
    }
    else throw std::runtime_error("Unrecognized argument, expected 'a', 'b', or 'c'");
  }

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
  if(run_1d_j_array)
  {
    nvtxRangePushA("cufft_1d_j_array_transpose");
    double cufft_1d_j_array_transpose_num_seconds = complex_to_complex_1d_j_fft_array_transpose(in_arr, out_check_arr, warmup, iter);
    std::cout << std::setw(colWidth) << "cufft_1d_j_array_transpose "
              << std::setw(colWidth) << 1000 * cufft_1d_j_array_transpose_num_seconds
              << std::endl;
    nvtxRangePop();
    nvtxRangePushA("cufft_1d_j_array");
    double cufft_1d_j_array_num_seconds = complex_to_complex_1d_j_fft_array(in_arr, out_arr, warmup, iter);
    std::cout << std::setw(colWidth) << "cufft_1d_j_array "
              << std::setw(colWidth) << 1000 * cufft_1d_j_array_num_seconds
              << std::endl;
    nvtxRangePop();
    check_close(out_check_arr, out_arr, NUM_ELEMENTS, "Mismatch between cufft_1d_j_array and cufft_1d_j_array_transpose");
  }
  // time cufft for bricks
  if(run_1d_j_brick)
  {
    // re-zero out out_arr
    #pragma omp parallel
    for(unsigned i = 0; i < NUM_ELEMENTS; ++i) out_arr[i] = 0.0;
    nvtxRangePushA("cufft_1d_j_brick_transpose");
    double cufft_1d_j_brick_transpose_num_seconds = complex_to_complex_1d_j_fft_brick_transpose(in_arr, out_arr, warmup, iter);
    std::cout << std::setw(colWidth) << "cufft_1d_j_brick_transpose"
              << std::setw(colWidth) << 1000 * cufft_1d_j_brick_transpose_num_seconds
              << std::endl;
    nvtxRangePop();
    // run correctness check
    if(run_1d_j_array)
    {
      check_close(out_check_arr, out_arr, NUM_ELEMENTS, "Mismatch between 1d_j_array and 1d_j_brick_transpose");
    }
    // re-zero out out_arr
    #pragma omp parallel
    for(unsigned i = 0; i < NUM_ELEMENTS; ++i) out_arr[i] = 0.0;
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
  }
  if(run_1d_collapsed_ij_brick)
  {
    // re-zero out out_arr
    #pragma omp parallel
    for(unsigned i = 0; i < NUM_ELEMENTS; ++i) out_arr[i] = 0.0;
    nvtxRangePushA("cufft_1d_collapsed_ij_brick");
    double cufft_1d_collapsed_ij_num_seconds = complex_to_complex_1d_collaped_ij_fft_brick(in_arr, out_arr, warmup, iter);
    std::cout << std::setw(colWidth) << "cufft_1d_collapsed_ij"
              << std::setw(colWidth) << 1000 * cufft_1d_collapsed_ij_num_seconds
              << std::endl;
    nvtxRangePop();
  }

  // free memory
  free(out_arr);
  free(in_arr);
}