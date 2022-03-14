//
// Created by Benjamin Sepanski on 1/11/22.
//

#ifndef BRICK_SINGLE_UTIL_H
#define BRICK_SINGLE_UTIL_H

#include "util.h"
#include "brick-stencils.h"
// global constants set by CLI
extern unsigned NUM_ITERATIONS; ///< how many measured iters?
extern unsigned NUM_WARMUPS;    ///< how many warmup iters?

/**
 * @brief parse args
 * @param[out] perProcessDomainSize the extent in each dimension of the domain
 * @param[out] outputFileName the output file to write to
 * @param[out] appendToFile true if the file should be appended to
 * @param[in] in input stream to read from
 *
 * @return the number of iterations, with default 100
 */
trial_iter_count parseSingleArgs(std::array<unsigned, RANK> *perProcessDomainSize,
                                 std::string *outputFileName, bool *appendToFile, std::istream &in);

/**
 * Get device peak memory bandwidth in GB/s (GB=10**9 bytes)
 * @param device the device number
 * @param print if true, print some device info
 * @return the peak memory bandwidth
 */
double getDevicePeakMemoryBandwidthGBPerS(int device, bool print = false);

/**
 * Print theoretical limits on AI/GStencil/s to console
 * @param minNumBytesTransferred minimum number of bytes to be transferrerd
 * @param numStencils number of stencils being computed
 * @param flopsPerStencil number of flops per stencil
 * @param dataRecorder the recorder to record the limits in
 * @param device the CUDA device
 */
void printTheoreticalLimits(size_t minNumBytesTransferred, size_t numStencils,
                            size_t flopsPerStencil, CSVDataRecorder &dataRecorder, int device = 0);

/**
 * @brief times func and prints stats
 *
 * @param func[in] the func to run
 * @param totElems[in] the number of stencils computed
 * @param csvDataRecorder[in,out] csv data recorder to record data in
 */
void timeAndPrintStats(std::function<void(void)> func, size_t numStencils,
                       CSVDataRecorder &csvDataRecorder);

/**
 * Pad the array arrayToPad with zeros on each side of the axes as described by padSize.
 *
 * If padSize is negative, padding is removed instead of added
 * @param arrayToPad the array to pad
 * @param padSize the amount of padding to add (or remove) on each axis
 */
template <typename DataType, typename Padding, typename SizeType, typename IndexType>
brick::Array<DataType, 6, Padding, SizeType, IndexType>
padWithZeros6D(const brick::Array<DataType, 6, Padding, SizeType, IndexType> &arrayToPad,
               std::array<int, 6> padSize) {
  std::array<unsigned, 6> paddedExtent{};
  for (unsigned d = 0; d < 6; ++d) {
    paddedExtent[d] = arrayToPad.extent[d] + 2 * padSize[d];
    if (paddedExtent[d] <= 0) {
      throw std::runtime_error("padSize * 2 must be < arrayToPad.extent");
    }
  }

  brick::Array<DataType, 6, Padding, SizeType, IndexType> paddedArray{paddedExtent, 0.0};
  for (unsigned n = 0; n < arrayToPad.extent[5]; ++n) {
    if (0 <= padSize[5] + (int)n && padSize[5] + (int)n <= (int)paddedArray.extent[5]) {
      for (unsigned m = 0; m < arrayToPad.extent[4]; ++m) {
        if (0 <= padSize[4] + (int)m && padSize[4] + (int)m <= (int)paddedArray.extent[4]) {
          for (unsigned l = 0; l < arrayToPad.extent[3]; ++l) {
            if (0 <= padSize[3] + (int)l && padSize[3] + (int)l <= (int)paddedArray.extent[3]) {
              for (unsigned k = 0; k < arrayToPad.extent[2]; ++k) {
                if (0 <= padSize[2] + (int)k && padSize[2] + (int)k <= (int)paddedArray.extent[2]) {
                  for (unsigned j = 0; j < arrayToPad.extent[1]; ++j) {
                    if (0 <= padSize[1] + (int)j &&
                        padSize[1] + (int)j <= (int)paddedArray.extent[1]) {
                      for (unsigned i = 0; i < arrayToPad.extent[0]; ++i) {
                        if (0 <= padSize[0] + (int)i &&
                            padSize[0] + (int)i <= (int)paddedArray.extent[0]) {
                          paddedArray(i + padSize[0], j + padSize[1], k + padSize[2],
                                      l + padSize[3], m + padSize[4], n + padSize[5]) =
                              arrayToPad.get(i, j, k, l, m, n);
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  return paddedArray;
}

/**
 * Pad the array arrayToPad with zeros on each side of the axes as described by padSize.
 *
 * If padSize is negative, padding is removed instead of added
 * @param arrayToPad the array to pad
 * @param padSize the amount of padding to add (or remove) on each axis
 */
template <typename DataType, typename Padding, typename SizeType, typename IndexType>
brick::Array<DataType, 5, Padding, SizeType, IndexType>
padWithZeros5D(const brick::Array<DataType, 5, Padding, SizeType, IndexType> &arrayToPad,
               std::array<int, 5> padSize) {
  std::array<unsigned, 5> paddedExtent{};
  for (unsigned d = 0; d < 5; ++d) {
    paddedExtent[d] = arrayToPad.extent[d] + 2 * padSize[d];
    if (paddedExtent[d] <= 0) {
      throw std::runtime_error("padSize * 2 must be < arrayToPad.extent");
    }
  }

  brick::Array<DataType, 5, Padding, SizeType, IndexType> paddedArray{paddedExtent, 0.0};
  for (unsigned m = 0; m < arrayToPad.extent[4]; ++m) {
    if (0 <= padSize[4] + (int)m && padSize[4] + (int)m <= (int)paddedArray.extent[4]) {
      for (unsigned l = 0; l < arrayToPad.extent[3]; ++l) {
        if (0 <= padSize[3] + (int)l && padSize[3] + (int)l <= (int)paddedArray.extent[3]) {
          for (unsigned k = 0; k < arrayToPad.extent[2]; ++k) {
            if (0 <= padSize[2] + (int)k && padSize[2] + (int)k <= (int)paddedArray.extent[2]) {
              for (unsigned j = 0; j < arrayToPad.extent[1]; ++j) {
                if (0 <= padSize[1] + (int)j && padSize[1] + (int)j <= (int)paddedArray.extent[1]) {
                  for (unsigned i = 0; i < arrayToPad.extent[0]; ++i) {
                    if (0 <= padSize[0] + (int)i &&
                        padSize[0] + (int)i <= (int)paddedArray.extent[0]) {
                      paddedArray(i + padSize[0], j + padSize[1], k + padSize[2], l + padSize[3],
                                  m + padSize[4]) = arrayToPad.get(i, j, k, l, m);
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  return paddedArray;
}

#endif // BRICK_SINGLE_UTIL_H
