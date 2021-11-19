//
// Created by Ben_Sepanski on 11/6/2021.
//

#ifndef BRICK_MANAGEDBRICKSTORAGE_H
#define BRICK_MANAGEDBRICKSTORAGE_H

#include "brick.h"
#ifdef __CUDACC__
#include "brick-cuda.h"
#endif
#include <memory>

namespace brick {
/**
 * Used to handle automatic allocation of an associated
 * BrickStorage on cuda devices
 */
class ManagedBrickStorage {
private:
  BrickStorage hostStorage;
  std::shared_ptr<BrickStorage> cudaStoragePtr{new BrickStorage};
  std::shared_ptr<bool> allocatedOnDevice{new bool};
public:
  const long chunks;
  const long step;
  explicit ManagedBrickStorage(long chunks, long step);
  explicit ManagedBrickStorage(long chunks, long step, void *mmap_fd, size_t offset = 0);
  BrickStorage getHostStorage() const;
#ifdef __CUDACC__
  /**
   * Defined in header so that nvcc can find it
   */
  inline BrickStorage getCudaStorage() {
    if (!*allocatedOnDevice) {
      *allocatedOnDevice = true;
      BrickStorage bStorage_dev = getHostStorage();
      bStorage_dev.mmap_info = nullptr;
      size_t size = bStorage_dev.step * bStorage_dev.chunks * sizeof(bElem);
      bElem *datptr;
      cudaCheck(cudaMalloc(&datptr, size));
      bStorage_dev.dat = std::shared_ptr<bElem>(
          datptr,
          [](bElem *p) {
            cudaCheck(cudaFree(p));
          });
      *cudaStoragePtr = bStorage_dev;
    }
    return *cudaStoragePtr;
  }
#endif
};
} // end namespace brick

#endif // BRICK_MANAGEDBRICKSTORAGE_H
