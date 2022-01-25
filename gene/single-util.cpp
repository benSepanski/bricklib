//
// Created by Benjamin Sepanski on 1/11/22.
//

#include "single-util.h"

unsigned NUM_ITERATIONS, NUM_WARMUPS;

trial_iter_count parse_single_args(std::array<unsigned, RANK> *perProcessDomainSize,
                                   std::string *outputFileName,
                                   bool *appendToFile,
                                   std::istream &in) {
  std::string optionString;
  trial_iter_count iterCount{};
  iterCount.num_iters = 100;
  iterCount.num_warmups = 5;
  std::vector<unsigned> tuple;
  bool readDomSize = false, readNumIters = false, readNumWarmups = false;
  *outputFileName = "results.csv";
  *appendToFile = false;
  std::string helpString = "Program options\n"
                           "  -h: show help (this message)\n"
                           "  Domain size,  in array order contiguous first\n"
                           "  -d: comma separated Int[6], domain size\n"
                           "  Num Tasks per dimension, in array order contiguous first\n"
                           "  Benchmark control:\n"
                           "  -I: number of iterations, default 100 \n"
                           "  -W: number of warmup iterations, default 5\n"
                           "  -o: csv file to write to (default results.csv)\n"
                           "  -a: If passed, will append data to output file (if it already exists)\n"
                           "Example usage:\n"
                           "  weak/gene6d -d 70,16,24,48,32,2\n";
  std::ostringstream errorStream;
  while (in >> optionString) {
    if (optionString[0] != '-' || optionString.size() != 2) {
      errorStream << "Unrecognized option " << optionString << std::endl;
    }
    if (!errorStream.str().empty()) {
      errorStream << helpString;
      throw std::runtime_error(errorStream.str());
    }
    switch (optionString[1]) {
    case 'a':
      *appendToFile = true;
      break;
    case 'd':
      tuple = read_uint_tuple(in, ',');
      if (readDomSize) {
        errorStream << "-d option should only be passed once" << std::endl;
      } else if (tuple.size() != RANK) {
        errorStream << "Expected extent of length " << RANK << ", not " << tuple.size();
      } else {
        std::copy(tuple.begin(), tuple.end(), perProcessDomainSize->begin());
      }
      readDomSize = true;
      break;
    case 'o':
      in >> *outputFileName;
      break;
    case 'I':
      if (readNumIters) {
        errorStream << "-I option should only be passed once" << std::endl;
      } else {
        in >> iterCount.num_iters;
      }
      readNumIters = true;
      break;
    case 'W':
      if (readNumWarmups) {
        errorStream << "-W option should only be passed once" << std::endl;
      } else {
        in >> iterCount.num_warmups;
      }
      readNumWarmups = true;
      break;
    default:
      errorStream << "Unrecognized option " << optionString << std::endl;
    }
  }
  if (!readDomSize) {
    errorStream << "Missing -d option" << std::endl << helpString;
    throw std::runtime_error(errorStream.str());
  }
  return iterCount;
}
void timeAndPrintStats(std::function<void(void)> func, size_t numStencils,
                       CSVDataRecorder &csvDataRecorder) {
  // setup for timing the function
  cudaEvent_t start, stop;
  float elapsed = 0.0;
  gpuCheck(cudaDeviceSynchronize());
  gpuCheck(cudaEventCreate(&start));
  gpuCheck(cudaEventCreate(&stop));

  // warmup function
  for (int i = 0; i < NUM_WARMUPS; ++i) {
    func();
  }

  // timei the function
  gpuCheck(cudaEventRecord(start));
  for (int i = 0; i < NUM_ITERATIONS; ++i) {
    func();
  }
  gpuCheck(cudaEventRecord(stop));
  gpuCheck(cudaEventSynchronize(stop));
  gpuCheck(cudaEventElapsedTime(&elapsed, start, stop));
  double avg_time = elapsed / (double) NUM_ITERATIONS / 1000.0; ///< in seconds
  double avg_gstencils_s = (double) numStencils / avg_time;

  csvDataRecorder.newRow();
  csvDataRecorder.record("avgTime(s)", avg_time);
  std::cout << avg_time << "(s)"
            << " " << avg_gstencils_s << "(GStencil/s)" << std::endl;
}
