#include <cmath>
#include "fft.h"
#include "brick-cuda.h"
#include "bricksetup.h"
#include "multiarray.h"

constexpr unsigned DIM = 3;
constexpr unsigned EXTENT = 4;
constexpr unsigned BDIM = 2;
typedef Brick<Dim<BDIM,BDIM,BDIM>, Dim<1>, true> BrickType;
typedef BricksCufftPlan<BrickType, FourierType<ComplexToComplex, 1, 2> > PlanType;

int main()
{
  const std::vector<long> extents(DIM, EXTENT);
  const std::vector<long> padding(DIM, 0);
  const std::vector<long> ghost_zone(DIM, 0);
  // set up arrays
  bComplexElem *in_arr = randomComplexArray(extents),
               *out_arr = zeroComplexArray(extents);
  
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
  // compute inverse
  std::cout << "Starting FFT^{-1}" << std::endl;
  myPlan.setup(outBrick_dev, grid_ptr_dev, inBrick_dev, grid_ptr_dev);
  myPlan.launch(PlanType::BRICKS_FFT_INVERSE);
  cudaCheck(cudaDeviceSynchronize());
  std::cout << "FFTs complete" << std::endl;

  // zero out data before copy-back just to be careful
  for(unsigned i = 0; i < bInfo.nbricks * bStorage.step; ++i) bStorage.dat.get()[i] = 0.0;
  // copy output back from device
  std::cout << "Starting memcpy to host" << std::endl;
  cudaCheck(cudaMemcpy(
    bStorage.dat.get(),
    bStorage_dev.dat.get(),
    bInfo.nbricks * bStorage.step * sizeof(bElem),
    cudaMemcpyDeviceToHost
  ));
  copyFromBrick<DIM>(extents, padding, ghost_zone, out_arr, grid_ptr, inBrick);
  
  // correctness check
  for(unsigned i = 0; i < static_power<EXTENT, DIM>::value; ++i)
  {
    std::complex<bElem> diff = (out_arr[i] - in_arr[i]);
    if(std::abs(diff) > 1e-8)
    {
      std::ostringstream errorMsgStream;
      errorMsgStream << "in[" << i << "]:" << in_arr[i] << " != out[" << i << "]:" << out_arr[i] << "\n";
      throw std::runtime_error(errorMsgStream.str());
    }
  }

  // free memory
  cudaCheck(cudaFree(grid_ptr_dev));
  cudaCheck(cudaFree(_bInfo_dev.adj));
  cudaCheck(cudaFree(bInfo_dev));
  cudaCheck(cudaFree(grid_ptr_dev));
  free(grid_ptr);
  free(out_arr);
  free(in_arr);
}