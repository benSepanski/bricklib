//
// Created by Benjamin Sepanski on 1/11/22.
//

#ifndef BRICK_SINGLE_UTIL_H
#define BRICK_SINGLE_UTIL_H

#include "util.h"
// global constants set by CLI
extern unsigned NUM_ITERATIONS; ///< how many measured iters?
extern unsigned NUM_WARMUPS;   ///< how many warmup iters?

/**
 * @brief parse args
 * @param[out] perProcessDomainSize the extent in each dimension of the domain
 * @param[out] outputFileName the output file to write to
 * @param[out] appendToFile true if the file should be appended to
 * @param[in] in input stream to read from
 *
 * @return the number of iterations, with default 100
 */
trial_iter_count parse_single_args(std::array<unsigned, RANK> *perProcessDomainSize,
                                   std::string *outputFileName,
                                   bool *appendToFile,
                                   std::istream &in);

/**
 * @brief times func and prints stats
 *
 * @param func[in] the func to run
 * @param totElems[in] the number of stencils computed
 * @param csvDataRecorder[in,out] csv data recorder to record data in
 */
void timeAndPrintStats(std::function<void(void)> func, size_t numStencils,
                       CSVDataRecorder &csvDataRecorder);

#endif // BRICK_SINGLE_UTIL_H
