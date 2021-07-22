#include <cmath>
#include "fft.h"
#include "brick-cuda.h"
#include "bricksetup.h"
#include "multiarray.h"

constexpr unsigned DIM = 3;
constexpr unsigned EXTENT = 4;
constexpr unsigned BDIM = 2;
typedef Brick<Dim<BDIM,BDIM,BDIM>, Dim<1>, true> BrickType;
typedef BricksCufftPlan<BrickType, FourierType<ComplexToComplex, 1> > PlanType;

void correctness_check(bComplexElem *in_arr, bComplexElem *out_arr)
{
  for(unsigned i = 0; i < static_power<EXTENT, DIM>::value; ++i)
  {
    std::complex<bElem> diff = (out_arr[i] - in_arr[i]);
    if(std::abs(diff) > 1e-8)
    {
      std::ostringstream errorMsgStream;
      errorMsgStream << "in[" << i << "]:" << in_arr[i] << " != out[" << i << "]:" << out_arr[i] << "\n";
      errorMsgStream << "in[" << i << "] / out[" << i << "] =" << in_arr[i] / out_arr[i] << "\n";
      throw std::runtime_error(errorMsgStream.str());
    }
  }
}

int main()
{
  const std::vector<long> extents(DIM, EXTENT);
  const std::vector<long> padding(DIM, 0);
  const std::vector<long> ghost_zone(DIM, 0);
  // set up arrays
  bComplexElem *in_arr = randomComplexArray(extents),
               *out_arr = zeroComplexArray(extents),
               *out_check_arr = zeroComplexArray(extents);
  
  // move arrays to host-side bricks
  const std::vector<long> grid_extents(DIM, EXTENT / BDIM);
  unsigned *grid_ptr = nullptr;
  BrickInfo<DIM> bInfo = init_grid<DIM>(grid_ptr, grid_extents);
  BrickStorage bStorage = bInfo.allocate(2 * BrickType::BRICKSIZE);
  BrickType inBrick(&bInfo, bStorage, 0),
           outBrick(&bInfo, bStorage, BrickType::BRICKSIZE);
  copyToBrick<DIM>(extents, in_arr, grid_ptr, inBrick);

  // set up brick info in cuda
  BrickInfo<DIM> *bInfo_dev;
  BrickInfo<DIM> _bInfo_dev = movBrickInfo(bInfo, cudaMemcpyHostToDevice);
  cudaCheck(cudaMalloc(&bInfo_dev, sizeof(BrickInfo<DIM>)));
  cudaCheck(cudaMemcpy(bInfo_dev, &_bInfo_dev, sizeof(BrickInfo<DIM>), cudaMemcpyHostToDevice));
  // mov brick storage to cuda
  BrickStorage bStorage_dev = movBrickStorage(bStorage, cudaMemcpyHostToDevice);
  // set up brick in cuda
  BrickType inBrick_dev(bInfo_dev, bStorage_dev, 0);
  BrickType outBrick_dev(bInfo_dev, bStorage_dev, BrickType::BRICKSIZE);
  // copy grids to device
  unsigned *grid_ptr_dev;
  size_t size = sizeof(unsigned) * static_power<EXTENT / BDIM, DIM>::value;
  cudaCheck(cudaMalloc(&grid_ptr_dev, size));
  cudaCheck(cudaMemcpy(grid_ptr_dev, grid_ptr, size, cudaMemcpyHostToDevice));

  // set up FFT for bricks
  PlanType myPlan({EXTENT/BDIM, EXTENT/BDIM, EXTENT/BDIM});
  myPlan.setup(inBrick_dev, grid_ptr_dev, outBrick_dev, grid_ptr_dev);
  // compute FFT
  std::cout << "Starting FFT" << std::endl;
  myPlan.launch();
  cudaCheck(cudaDeviceSynchronize());

  // now compute FFT on regular array
  cuDoubleComplex *in_arr_dev, *out_arr_dev;
  size = sizeof(bComplexElem) * static_power<EXTENT, DIM>::value;
  cudaCheck(cudaMalloc(&in_arr_dev, size));
  cudaCheck(cudaMalloc(&out_arr_dev, size));
  cudaCheck(cudaMemcpy(in_arr_dev, in_arr, size, cudaMemcpyHostToDevice));
  cufftHandle arr_plan;
  int n[1] = {EXTENT};
  int embed[1] = {EXTENT*EXTENT};
  int stride = EXTENT;
  int batchDist = EXTENT*EXTENT;
  int numBatches = EXTENT;
  cufftCheck(cufftPlanMany(&arr_plan, 1, n, embed, stride, batchDist, embed, stride, batchDist, CUFFT_Z2Z, numBatches));
  for(unsigned i = 0; i < EXTENT; ++i)
  {
    cufftExecZ2Z(arr_plan, in_arr_dev + i, out_arr_dev + i, CUFFT_FORWARD);
  }
  cudaCheck(cudaDeviceSynchronize());
  // Now compare array FFT with bricks fft
  std::cout << "Comparing bricksFFT result with array FFT result" << std::endl;
  // make sure out_arr != out_check_arr before memcpy
  for(unsigned i = 0; i < static_power<EXTENT, DIM>::value; ++i) out_arr[i] = 1.0;
  for(unsigned i = 0; i < static_power<EXTENT, DIM>::value; ++i) out_check_arr[i] = 0.0;
  cudaCheck(cudaMemcpy(
    bStorage.dat.get(),
    bStorage_dev.dat.get(),
    bInfo.nbricks * bStorage.step * sizeof(bElem),
    cudaMemcpyDeviceToHost
  ));
  copyFromBrick<DIM>(extents, padding, ghost_zone, out_arr, grid_ptr, outBrick);
  cudaCheck(cudaMemcpy(out_check_arr, out_arr_dev, size, cudaMemcpyDeviceToHost));
  correctness_check(out_arr, out_check_arr);
  // zero-out out arrays
  for(unsigned i = 0; i < static_power<EXTENT, DIM>::value; ++i) out_arr[i] = 0.0;
  for(unsigned i = 0; i < static_power<EXTENT, DIM>::value; ++i) out_check_arr[i] = 0.0;
  std::cout << "bricksFFT result matches array FFT result" << std::endl;

  // now perform inverse FFTs
  for(unsigned i = 0; i < EXTENT; ++i)
  {
    cufftExecZ2Z(arr_plan, out_arr_dev + i, out_arr_dev + i, CUFFT_INVERSE);
  }
  // copy result back to host
  cudaCheck(cudaMemcpy(out_arr, out_arr_dev, size, cudaMemcpyDeviceToHost));
  std::cout << "Starting array FFT inverse correctness check" << std::endl;
  // handle scaling https://docs.nvidia.com/cuda/cufft/index.html#cufft-transform-directions
  for(unsigned i = 0; i < static_power<EXTENT, DIM>::value; ++i) out_arr[i] /= EXTENT;
  correctness_check(in_arr, out_arr);
  // zero-out out_arr
  for(unsigned i = 0; i < static_power<EXTENT, DIM>::value; ++i) out_arr[i] = 0.0;

  // compute inverse
  std::cout << "Starting bricks FFT inverse" << std::endl;
  myPlan.setup(outBrick_dev, grid_ptr_dev, inBrick_dev, grid_ptr_dev);
  myPlan.launch(PlanType::BRICKS_FFT_INVERSE);
  cudaCheck(cudaDeviceSynchronize());

  // zero out data before copy-back just to be careful
  for(unsigned i = 0; i < bInfo.nbricks * bStorage.step; ++i) bStorage.dat.get()[i] = 0.0;
  // copy output back from device
  std::cout << "Starting bricks FFT inverse correctness check" << std::endl;
  cudaCheck(cudaMemcpy(
    bStorage.dat.get(),
    bStorage_dev.dat.get(),
    bInfo.nbricks * bStorage.step * sizeof(bElem),
    cudaMemcpyDeviceToHost
  ));
  copyFromBrick<DIM>(extents, padding, ghost_zone, out_arr, grid_ptr, inBrick);
  
  // correctness check
  // handle scaling https://docs.nvidia.com/cuda/cufft/index.html#cufft-transform-directions
  for(unsigned i = 0; i < static_power<EXTENT, DIM>::value; ++i) out_arr[i] /= EXTENT;
  correctness_check(in_arr, out_arr);

  // free memory
  cudaCheck(cudaFree(grid_ptr_dev));
  cudaCheck(cudaFree(_bInfo_dev.adj));
  cudaCheck(cudaFree(bInfo_dev));
  free(grid_ptr);
  free(out_check_arr);
  free(out_arr);
  free(in_arr);
}