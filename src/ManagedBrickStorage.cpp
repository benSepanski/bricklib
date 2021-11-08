//
// Created by Ben_Sepanski on 11/6/2021.
//

#include "ManagedBrickStorage.h"

brick::ManagedBrickStorage::ManagedBrickStorage(long chunks, long step)
    : hostStorage{BrickStorage::allocate(chunks, step)}
    , chunks{chunks}
    , step{step}
{
  *allocatedOnDevice = false;
}

brick::ManagedBrickStorage::ManagedBrickStorage(long chunks, long step, void *mmap_fd,
                                         size_t offset)
    : chunks{chunks}, step{step}
{
  *allocatedOnDevice = false;
  if (mmap_fd == nullptr) {
    hostStorage = BrickStorage::mmap_alloc(chunks, step);
  } else {
    hostStorage = BrickStorage::mmap_alloc(chunks, step, mmap_fd, offset);
  }
}

BrickStorage brick::ManagedBrickStorage::getHostStorage() const { return hostStorage; }
