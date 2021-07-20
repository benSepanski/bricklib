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

__constant__ typename PlanType::BricksCufftInfo cuFFTInfo;

int main()
{
    const std::vector<long> extents(DIM, EXTENT);
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
    cudaMalloc(&bInfo_dev, sizeof(BrickInfo<DIM>));
    cudaMemcpy(bInfo_dev, &_bInfo_dev, sizeof(BrickInfo<DIM>), cudaMemcpyHostToDevice);
    BrickStorage bStorage_dev = movBrickStorage(bStorage, cudaMemcpyHostToDevice);
    BrickType inBrick_dev(bInfo_dev, bStorage_dev, 0),
              outBrick_dev(bInfo_dev, bStorage_dev, BrickType::BRICKSIZE);
    unsigned *grid_ptr_dev;
    size_t size = sizeof(unsigned) * static_power<EXTENT / BDIM, DIM>::value;
    cudaMalloc(&grid_ptr_dev, size);
    cudaMemcpy(grid_ptr_dev, grid_ptr, size, cudaMemcpyHostToDevice);

    // set up FFT for bricks
    PlanType myPlan({EXTENT, EXTENT, EXTENT});
    myPlan.setup(&inBrick_dev, grid_ptr_dev, &outBrick_dev, grid_ptr_dev, &cuFFTInfo);

    // free memory
    cudaFree(_bInfo_dev.adj);
    cudaFree(bInfo_dev);
    cudaFree(grid_ptr_dev);
    free(out_arr);
    free(in_arr);
}