#include "stencils/gene-6d.h"
#include <iomanip>

// usage: (Optional) [num iterations] 
//        (Optional) [num warmup iterations] 
//        (Optional) [which kernel (ij or arakawa)] 
//        (Optional) [g for just gtensor, b for just bricks]
int main(int argc, char * const argv[]) {
  // read command line args
  if(argc > 5) throw std::runtime_error("Expected at most 2 arguments");
  if(argc >= 2) NUM_ITERS = std::stoi(argv[1]);
  if(argc >= 3) NUM_WARMUP_ITERS = std::stoi(argv[2]);
  bool run_ij = true,
       run_arakawa = true;
  if(argc >= 4)
  {
    std::string which_kernel(argv[3]);
    if(which_kernel[0] == 'i') run_arakawa = false;
    else if(which_kernel[0] == 'a') run_ij = false;
    else throw std::runtime_error("Expected 'ij' or 'arakawa'");
  }
  bool run_bricks = true,
       run_gtensor = true;
  if(argc >= 5)
  {
    std::string which_method(argv[4]);
    if(which_method[0] == 'g') run_bricks = false;
    else if(which_method[0] == 'b') run_gtensor = false;
    else throw std::runtime_error("Expected 'g' or 'b'");
  }

  // print some helpful cuda info
  cudaDeviceProp prop;
  cudaCheck(cudaGetDeviceProperties(&prop, 0));
  std::cout << std::left;
  std::cout << std::setw(24) << "Device name" << " : " << prop.name << "\n"
            << std::setw(24) << "Compute Capability" << " : " << prop.major << "." << prop.minor << "\n"
            << std::setw(24) << "L2 cache size" << " : " << prop.l2CacheSize << "\n"
            << std::setw(24) << "Multiprocessor Count" << " : " << prop.multiProcessorCount << "\n"
            << std::setw(24) << "Max Warps / SM" << " : " << prop.maxThreadsPerMultiProcessor / prop.warpSize << "\n"
            << std::flush;

  // print trial info
  #ifndef NDEBUG
  std::cout << "NDEBUG is not defined" << std::endl;
  #else
  std::cout << "NDEBUG is defined" << std::endl;
  // check cuda device
  unsigned *dummy;
  _cudaCheck(cudaMalloc(&dummy, sizeof(decltype(dummy))), "cudaMalloc(&dummy, sizeof(unsigned))", __FILE__, __LINE__);
  _cudaCheck(cudaFree(dummy), "cudaFree(dummy)", __FILE__, __LINE__);
  #endif
  unsigned width = 2;
  std::cout << std::right;
  std::cout << "GRID      : n x m x l x k x j x i = " << std::setw(width) << EXTENT_n << " x " 
                                                      << std::setw(width) << EXTENT_m << " x " 
                                                      << std::setw(width) << EXTENT_l << " x "
                                                      << std::setw(width) << EXTENT_k << " x " 
                                                      << std::setw(width) << EXTENT_j << " x " 
                                                      << std::setw(width) << EXTENT_i << " \n"
            << "BRICK DIMS: n x m x l x k x j x i = " << std::setw(width) << BDIM_n << " x " 
                                                      << std::setw(width) << BDIM_m << " x " 
                                                      << std::setw(width) << BDIM_l << " x "
                                                      << std::setw(width) << BDIM_k << " x " 
                                                      << std::setw(width) << BDIM_j << " x " 
                                                      << std::setw(width) << BDIM_i << "\n"
            << "WARM UP:" << NUM_WARMUP_ITERS << "\n"
            << "ITERATIONS:" << NUM_ITERS << std::endl;

  if(run_ij)
  {
    ij_deriv(run_bricks, run_gtensor);
  }
  if(run_arakawa)
  {
    semi_arakawa(run_bricks, run_gtensor);
  }
  return 0;
}