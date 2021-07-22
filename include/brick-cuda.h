/**
 * @file
 * @brief For using bricks with CUDA
 */

#ifndef BRICK_BRICK_CUDA_H
#define BRICK_BRICK_CUDA_H

#include <cassert>
#include <cstdio>
#include <brick.h>
#include <cuda_runtime.h>

/**
 * @brief Check the return of CUDA calls, do nothing during release build
 */
#ifdef NDEBUG
#define cudaCheck(x) x
#else
#define cudaCheck(x) _cudaCheck(x, #x ,__FILE__, __LINE__)
#endif


/// Internal for #cudaCheck(x)
template<typename T>
void _cudaCheck(T e, const char *func, const char *call, const int line) {
  if (e != cudaSuccess) {
    printf("\"%s\" at %d in %s\n\treturned %d\n-> %s\n", func, line, call, (int) e, cudaGetErrorString(e));
    exit(EXIT_FAILURE);
  }
}

#define validateIsDevicePointer(x) _validateIsDevicePointer(x, #x, __FILE__, __LINE__)
#ifdef NDEBUG
#define validateIsDevicePointer_if_dbg(x)
#else
#define validateIsDevicePointer_if_dbg(x) _validateIsDevicePointer(x, #x, __FILE__, __LINE__)
#endif
/**
 * @brief throw std::runtime_error if ptr is not a device pointer.
 * Exits if cudaPointerGetAttributes does not terminate successfully.
 * @param ptr the pointer to check
 */
inline void _validateIsDevicePointer(const void *ptr, const char *errorMsg, const char *file, const int line)
{
  cudaPointerAttributes attributes;
  cudaError_t e = cudaPointerGetAttributes(&attributes, ptr);
  if(e != cudaSuccess)
  {
    printf("\"%s\" called from %s at line %d in %s\n\treturned %d\n-> %s\n",
           "cudaPointerGetAttributes(&attributes, ptr)", errorMsg, line, file, (int) e, cudaGetErrorString(e));
    exit(EXIT_FAILURE);
  }
  if(attributes.type != cudaMemoryTypeDevice)
  {
    std::ostringstream errorMsgStream;
    errorMsgStream << "ERROR at" << *errorMsg << " (" << *file << ":" << line << ")";
    errorMsgStream << " ptr points to ";
    switch(attributes.type)
    {
      case cudaMemoryTypeUnregistered: errorMsgStream << "unregistered"; break;
      case cudaMemoryTypeHost: errorMsgStream << "host"; break;
      case cudaMemoryTypeManaged: errorMsgStream << "managed"; break;
      default: errorMsgStream << "unrecognized";
    }
    errorMsgStream << " memory space, not device.";
    throw std::runtime_error(errorMsgStream.str());
  }
}

/**
 * @brief Moving BrickInfo to or from GPU (allocate new)
 * @tparam dims implicit when used with bInfo argument
 * @param bInfo BrickInfo to copy from host or GPU
 * @param kind Currently must be cudaMemcpyHostToDevice or cudaMemcpyDeviceToHost
 * @return a new BrickInfo struct allocated on the destination
 */
template<unsigned dims, typename CommunicatingDims>
BrickInfo<dims, CommunicatingDims> movBrickInfo(BrickInfo<dims, CommunicatingDims> &bInfo, cudaMemcpyKind kind) {
  assert(kind == cudaMemcpyHostToDevice || kind == cudaMemcpyDeviceToHost);

  // Make a copy
  BrickInfo<dims, CommunicatingDims> ret = bInfo;
  constexpr unsigned numCommDims = CommunicatingDims::numCommunicatingDims(dims);
  size_t size = bInfo.nbricks * static_power<3, numCommDims>::value * sizeof(unsigned);

  if (kind == cudaMemcpyHostToDevice) {
    cudaCheck(cudaMalloc(&ret.adj, size));
  } else {
    ret.adj = (unsigned (*)[static_power<3, numCommDims>::value]) malloc(size);
  }
  cudaCheck(cudaMemcpy(ret.adj, bInfo.adj, size, kind));
  return ret;
}

/**
 * @brief Moving BrickStorage to or from GPU (allocate new)
 * @param bStorage BrickStorage to copy from
 * @param kind Currently must be either cudaMemcpyHostToDevice or cudaMemcpyDeviceToHost
 * @return a new BrickStorage struct allocated on the destination
 */
inline BrickStorage movBrickStorage(BrickStorage &bStorage, cudaMemcpyKind kind) {
  assert(kind == cudaMemcpyHostToDevice || kind == cudaMemcpyDeviceToHost);

  bool isToDevice = (kind == cudaMemcpyHostToDevice);
  // Make a copy
  BrickStorage ret = bStorage;
  size_t size = bStorage.step * bStorage.chunks * sizeof(bElem);
  bElem *datptr;
  if (isToDevice) {
    cudaCheck(cudaMalloc(&datptr, size));
  } else {
    datptr = (bElem *) malloc(size);
  }
  cudaCheck(cudaMemcpy(datptr, bStorage.dat.get(), size, kind));
  if (isToDevice) {
    ret.dat = std::shared_ptr<bElem>(datptr, [](bElem *p) { cudaCheck(cudaFree(p));});
  } else {
    ret.dat = std::shared_ptr<bElem>(datptr, [](bElem *p) { free(p); });
  }
  return ret;
}

#include "dev_shl.h"

#endif //BRICK_BRICK_CUDA_H
