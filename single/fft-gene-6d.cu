#include <cmath>
#include <iostream>
#include <iomanip>
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

  // free memory
  cudaCheck(cudaFree(out_arr_dev));
  cudaCheck(cudaFree(in_arr_dev));

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
  int colWidth = 20;
  std::cout << std::setw(colWidth) << "method"
            << std::setw(colWidth) << "time(s)" 
            << std::endl;
  // time cufft for arrays
  if(run_1d_j_array)
  {
    double cufft_1d_j_array_num_seconds = complex_to_complex_1d_j_fft_array(in_arr, out_check_arr, warmup, iter);
    std::cout << std::setw(colWidth) << "cufft_1d_j_array "
              << std::setw(colWidth) << cufft_1d_j_array_num_seconds
              << std::endl;
  }
  // time cufft for bricks
  if(run_1d_j_brick)
  {
    double cufft_1d_j_brick_num_seconds = complex_to_complex_1d_j_fft_brick(in_arr, out_arr, warmup, iter);
    std::cout << std::setw(colWidth) << "cufft_1d_j_brick "
              << std::setw(colWidth) << cufft_1d_j_brick_num_seconds
              << std::endl;
    // run correctness check
    if(run_1d_j_array)
    {
      check_close(out_check_arr, out_arr, NUM_ELEMENTS, "Mismatch between 1d_j_array and 1d_j_brick");
    }
  }

  // free memory
  free(out_arr);
  free(in_arr);
}