#include "multiarray.h"
#include "cmpconst.h"
#include <iostream>
#include <random>

namespace {
  std::mt19937_64 *mt = nullptr;
  std::uniform_real_distribution<bElem> *u = nullptr;

#pragma omp threadprivate(mt)
#pragma omp threadprivate(u)

  bElem randD() {
    if (mt == nullptr) {
#pragma omp critical
      {
        std::random_device r;
        mt = new std::mt19937_64(r());
        u = new std::uniform_real_distribution<bElem>(0, 1);
      }
    }
    return (*u)(*mt);
  }
}

bElem *uninitArray(const std::vector<long> &list, long &size) {
  size = 1;
  for (auto i: list)
    size *= i;
  return (bElem *) aligned_alloc(ALIGN, size * sizeof(bElem));
}

bComplexElem *uninitComplexArray(const std::vector<long> &list, long &size) {
  size = 1;
  for (auto i: list)
    size *= i;
  return (bComplexElem *) aligned_alloc(ALIGN, size * sizeof(bComplexElem));
}

bElem *randomArray(const std::vector<long> &list) {
  long size;
  bElem *arr = uninitArray(list, size);
#pragma omp parallel for
  for (long l = 0; l < size; ++l)
    arr[l] = randD();
  return arr;
}

bComplexElem *randomComplexArray(const std::vector<long> &list) {
  long size;
  bComplexElem *arr = uninitComplexArray(list, size);
#pragma omp parallel for
  for (long l = 0; l < size; ++l)
    arr[l] = randD();
  return arr;
}

bElem *zeroArray(const std::vector<long> &list) {
  long size;
  bElem *arr = uninitArray(list, size);
#pragma omp parallel for
  for (long l = 0; l < size; ++l)
    arr[l] = 0.0;
  return arr;
}

bComplexElem *zeroComplexArray(const std::vector<long> &list) {
  long size;
  bComplexElem *arr = uninitComplexArray(list, size);
#pragma omp parallel for
  for (long l = 0; l < size; ++l)
    arr[l] = 0.0;
  return arr;
}
