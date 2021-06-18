#include "gene-5d.h"

template<typename elemType>
__global__ void kernel_double(const elemType* in, elemType* out)
{
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < NUM_PADDED_ELEMENTS) {
    out[i] = 2 * in[i];;
  }
}

void cudaDouble(gt::complex<bElem> *inPtr, gt::complex<bElem> *outPtr)
{
    kernel_double<<<NUM_PADDED_ELEMENTS / 256, 256>>>(inPtr, outPtr);
    cudaDeviceSynchronize();
}