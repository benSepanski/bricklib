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

__constant__ typename PlanType::BricksCufftInfo cufftInfo;

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

  // set up brick storage in cuda
  BrickInfo<DIM> _bInfo_dev = movBrickInfo<DIM>(bInfo, cudaMemcpyHostToDevice);
  BrickInfo<DIM> *bInfo_dev;
  cudaCheck(cudaMalloc(&bInfo_dev, sizeof(BrickInfo<DIM>)));
  cudaCheck(cudaMemcpy(bInfo_dev, &_bInfo_dev, sizeof(BrickInfo<DIM>), cudaMemcpyHostToDevice));
  BrickStorage bStorage_dev = movBrickStorage(bStorage, cudaMemcpyHostToDevice);
  BrickType inBrick_dev(bInfo_dev, bStorage_dev, 0),
            outBrick_dev(bInfo_dev, bStorage_dev, BrickType::BRICKSIZE);
  unsigned *grid_ptr_dev;
  size_t size = sizeof(unsigned) * static_power<EXTENT / BDIM, DIM>::value;
  cudaCheck(cudaMalloc(&grid_ptr_dev, size));
  cudaCheck(cudaMemcpy(grid_ptr_dev, grid_ptr, size, cudaMemcpyHostToDevice));

  // set up FFT for bricks
  PlanType myPlan({EXTENT/BDIM, EXTENT/BDIM, EXTENT/BDIM});
  myPlan.setup(&inBrick_dev, grid_ptr_dev, &outBrick_dev, grid_ptr_dev, &cufftInfo);
  // compute FFT
  myPlan.launch();
  cudaCheck(cudaDeviceSynchronize());
  // compute inverse
  myPlan.launch(PlanType::BRICKS_FFT_INVERSE);
  cudaCheck(cudaDeviceSynchronize());

  // copy output back from device
  cudaCheck(cudaMemcpy(bStorage.dat.get(), bStorage_dev.dat.get(),
                       sizeof(bComplexElem) * static_power<EXTENT, DIM>::value,
                       cudaMemcpyDeviceToHost));
  copyFromBrick<DIM>(extents, padding, ghost_zone, out_arr, grid_ptr, outBrick);
  
  // correctness check
  for(unsigned i = 0; i < static_power<EXTENT, DIM>::value; ++i)
  {
    std::complex<bElem> diff = (out_arr[i] - in_arr[i]);
    if(std::abs(diff) > 1e-8)
    {
      std::ostringstream errorMsgStream;
      errorMsgStream << "in:" << in_arr[i] << " != out:" << out_arr[i] << "\n";
      throw std::runtime_error(errorMsgStream.str());
    }
  }

  // free memory
  cudaCheck(cudaFree(_bInfo_dev.adj));
  cudaCheck(cudaFree(bInfo_dev));
  cudaCheck(cudaFree(grid_ptr_dev));
  free(out_arr);
  free(in_arr);
}