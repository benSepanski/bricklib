/**
 * @file
 * @brief Compare content from bricks with arrays
 */

#ifndef BRICK_BRICKCOMPARE_H
#define BRICK_BRICKCOMPARE_H

#include <iostream>
#include <cmath>
#include <complex>
#include "bricksetup.h"
#include "cmpconst.h"

extern bool compareBrick_b;     ///< Thread-private comparison accumulator

#pragma omp threadprivate(compareBrick_b)

// template<typename elemType>
// elemType abs(const elemType &val)
// {
//   // must be elem-type or value
//   static_assert(std::is_same<bElem, elemType>::value || std::is_same<bComplexElem, elemType>::value);
// }


/**
 * @brief Compare values between bricks and an array
 * @tparam dims number of dimensions
 * @tparam T type for brick
 * @param dimlist dimensions, contiguous first
 * @param padding padding applied to array format (skipped)
 * @param ghost padding applied to array and brick (skipped)
 * @param arr array input
 * @param grid_ptr the grid array contains indices of bricks
 * @param brick the brick data structure
 * @return False when not equal (with tolerance)
 */
template<unsigned dims, typename T>
inline bool
compareBrick(const std::vector<long> &dimlist, const std::vector<long> &padding, const std::vector<long> &ghost,
    typename T::elemType *arr, unsigned *grid_ptr, T &brick) {
  typedef typename T::elemType elemType; ///< convenient alias to avoid writing "typename" everywhere
  typedef typename T::stdElemType stdElemType; ///< STL-type of element

  bool ret = true;
  auto f = [&ret](elemType &brick, const elemType *arr) -> void {
    double diff = std::abs((stdElemType) (brick - *arr));
    bool r = (diff < BRICK_TOLERANCE) || (diff < (std::abs((stdElemType) brick) + std::abs((stdElemType) *arr)) * BRICK_TOLERANCE);
    compareBrick_b = (compareBrick_b && r);
  };

#pragma omp parallel default(none)
  {
    compareBrick_b = true;
  }

  iter_grid<dims>(dimlist, padding, ghost, arr, grid_ptr, brick, f);

#pragma omp parallel default(none) shared(ret)
  {
#pragma omp critical
    {
      ret = ret && compareBrick_b;
    }
  }

  return ret;
}

/**
 * @brief Compare all values between bricks and an array without ghost or padding
 * @tparam dims
 * @tparam T
 * @param dimlist
 * @param arr
 * @param grid_ptr
 * @param brick
 * @return
 *
 * For parameters see compareBrick(const std::vector<long> &dimlist, const std::vector<long> &padding, const std::vector<long> &ghost, typename T::elemType *arr, unsigned *grid_ptr, T &brick)
 */
template<unsigned dims, typename T>
inline bool
compareBrick(const std::vector<long> &dimlist, typename T::elemType *arr, unsigned *grid_ptr,
             T &brick) {
  std::vector<long> padding(dimlist.size(), 0);
  std::vector<long> ghost(dimlist.size(), 0);

  return compareBrick<dims, T>(dimlist, padding, ghost, arr, grid_ptr, brick);
}

#endif //BRICK_BRICKCOMPARE_H
