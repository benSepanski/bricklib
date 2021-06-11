/**
 * @file
 * @brief Multidimensional array shortcuts
 */

#ifndef MULTIARRAY_H
#define MULTIARRAY_H

#include <vector>
#include <brick.h>
#include "cmpconst.h"

/**
 * @brief Create an uninitialized multidimensional array
 * @param[in] list dimensions
 * @param[out] size the total size of the array in number of bElem
 * @return pointer to the newly created array
 */
bElem *uninitArray(const std::vector<long> &list, long &size);

/**
 * @brief Create an uninitialized multidimensional complex-valued array
 * @param[in] list dimensions
 * @param[out] size the total size of the array in number of bElem
 * @return pointer to the newly created array
 */
bComplexElem *uninitComplexArray(const std::vector<long> &list, long &size);

/**
 * @brief Create an multidimensional array initialized with random values
 * @param[in] list dimensions
 * @return pointer to the newly created array
 */
bElem *randomArray(const std::vector<long> &list);

/**
 * @brief Create an multidimensional array initialized with random complex values
 * @param[in] list dimensions
 * @return pointer to the newly created array
 */
bComplexElem *randomComplexArray(const std::vector<long> &list);

/**
 * @brief Create an multidimensional array initialized with zeros
 * @param[in] list dimensions
 * @return pointer to the newly created array
 */
bElem *zeroArray(const std::vector<long> &list);

/**
 * @brief Create an multidimensional complex-valued array initialized with zeros
 * @param[in] list dimensions
 * @return pointer to the newly created array
 */
bComplexElem *zeroComplexArray(const std::vector<long> &list);

/**
 * @brief Compare the value in two multidimensional arrays (within tolerance)
 * @tparam elemType the type of the elements (either bElem or bComplexElem)
 * @param[in] list dimensions
 * @param arrA
 * @param arrB
 * @return False when not equal
 */
template<typename elemType>
bool compareArray(const std::vector<long> &list, elemType *arrA, elemType *arrB) {
  static_assert(std::is_same<bElem, elemType>::value || std::is_same<bComplexElem, elemType>::value,
                "elemType expected to be bElem or bComplexElem");  

  long size = 1;
  for (auto i: list)
    size *= i;
  bool same = true;
#pragma omp parallel for reduction(&&: same)
  for (long l = 0; l < size; ++l) {
    elemType diff = std::abs(arrA[l] - arrB[l]);
    bool r = (diff < BRICK_TOLERANCE) || (diff < (std::abs(arrA[l]) + std::abs(arrB[l])) * BRICK_TOLERANCE);
    same = same && r;
  }
  return same;
}

#endif
