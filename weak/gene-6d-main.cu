/**
 * @file gene-6d-main.cu
 * @author Ben Sepanski (ben_sepanski@utexas.edu)
 * @brief 
 * @version 0.1
 * @date 2021-08-03
 * 
 * @copyright Copyright (c) 2021
 * 
 * Copied and modified from main.cu written by Tuowen Zhao
 * 
 */

#include <mpi.h>
#include <iostream>
#include <iomanip>

#include <brick.h>
#include <brick-mpi.h>
#include <bricksetup.h>
#include <brick-cuda.h>

#include "bitset.h"
#include <multiarray.h>
#include <brickcompare.h>
#include "stencils/cudaarray.h"
#include "stencils/gene-6d.h"

#include <unistd.h>
#include <array-mpi.h>

unsigned NUM_GHOST_ELEMENTS;   ///< how many ghost elements?
typedef BrickDecomp<DIM, BDIM_n, BDIM_m, BDIM_l, BDIM_k, BDIM_j, BDIM_i> GENEBrickDecomp;

/**
 * @brief check for MPI failure
 * 
 * @param return_value the value returned from an MPI call
 * @param func name of the MPI function being invoked
 * @param filename filename of call site
 * @param line line number of call site
 */
void _check_MPI(int return_value, const char *func, const char *filename, const int line) {
  if(return_value != MPI_SUCCESS) {
    char error_msg[MPI_MAX_ERROR_STRING + 1];
    int error_length;
    std::ostringstream error_stream;
    error_stream << "MPI Error during call " << func << " " << filename << ":" << line << std::endl;
    if(MPI_Error_string(return_value, error_msg, &error_length) != MPI_SUCCESS) {
      error_stream << "Invalid argument passed to MPI_Error_string" << std::endl;
    }
    else {
      error_stream << error_msg << std::endl;
    }
    throw std::runtime_error(error_stream.str());
  }
}

#define check_MPI(x) _check_MPI(x, #x ,__FILE__, __LINE__)

/**
 * @brief build a cartesian communicator
 * 
 * Assumes MPI_Init_thread has already been called.
 * 
 * Prints some useful information about the MPI setup.
 * 
 * @param[in] num_procs_per_dim the number of MPI processes to put in each dimension.
 *                                  Product must match the number of MPI processes.
 * @param[in] per_process_extent extent in each dimension for each individual MPI processes.
 * @return MPI_comm a cartesian communicator built from MPI_COMM_WORLD
 */
MPI_Comm build_cartesian_comm(std::array<int, DIM> num_procs_per_dim, std::array<int, DIM> per_process_extent) {
  // get number of MPI processes and my rank
  int size, rank;
  check_MPI(MPI_Comm_size(MPI_COMM_WORLD, &size));
  check_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

  // make sure num_procs_per_dim has product to number of processes
  int prod_of_procs_per_dim = std::accumulate(num_procs_per_dim.begin(), num_procs_per_dim.end(), 1, std::multiplies<size_t>());
  if(prod_of_procs_per_dim != size) {
    std::ostringstream error_stream;
    error_stream << "Product of number of processes per dimension is " << prod_of_procs_per_dim
                 << " which does not match number of MPI procceses (" << size << ")\n";
    throw std::runtime_error(error_stream.str());
  }

  // set up processes on a cartesian communication grid
  std::array<int, DIM> periodic = { true }; ///< use periodic BCs
  bool allow_ranking_reordering = true;
  MPI_Comm cartesian_comm;
  check_MPI(MPI_Cart_create(MPI_COMM_WORLD,
                            DIM,
                            num_procs_per_dim.data(),
                            periodic.data(),
                            allow_ranking_reordering,
                            &cartesian_comm));
  if(cartesian_comm == MPI_COMM_NULL) {
    throw std::runtime_error("Failure in cartesian comm setup");
  }

  // return the communicator
  return cartesian_comm;
}

__constant__ __device__ unsigned array_stride_dev[DIM]; ///< per-process with ghost-zones
__constant__ __device__ unsigned array_extent_dev[DIM]; ///< per-process with ghost-zones
__constant__ __device__ unsigned per_process_extent_dev[DIM]; ///< per-process without ghost-zones

// TODO
__global__
void semi_arakawa_arr_kernel(bComplexElem * __restrict__ out_ptr,
                             const bComplexElem * __restrict__ in_ptr,
                             bElem * __restrict__ coeff
                             )
{
  unsigned global_z_idx = threadIdx.z + blockIdx.z * blockDim.z;
  unsigned ijmn = (global_z_idx % (array_extent_dev[0] * array_extent_dev[1]))
                + (global_z_idx / (array_extent_dev[0] * array_extent_dev[1])) * array_stride_dev[3];
  unsigned k = PADDING_k + threadIdx.x + blockIdx.x * blockDim.x;
  unsigned l = PADDING_l + threadIdx.y + blockIdx.y * blockDim.y;
  size_t ijklmn = ijmn + k * array_stride_dev[2]
                       + l * array_stride_dev[3];
  size_t iklmn = ijklmn / array_stride_dev[1] 
               + ijklmn % array_extent_dev[0];
  const unsigned k_stride = array_stride_dev[2];
  const unsigned l_stride = array_stride_dev[3];
  out_ptr[ijklmn] = coeff[0 + ARAKAWA_STENCIL_SIZE * iklmn] * in_ptr[ijklmn - 2 * l_stride]
                  + coeff[1 + ARAKAWA_STENCIL_SIZE * iklmn] * in_ptr[ijklmn - l_stride - k_stride]
                  + coeff[2 + ARAKAWA_STENCIL_SIZE * iklmn] * in_ptr[ijklmn - l_stride]
                  + coeff[3 + ARAKAWA_STENCIL_SIZE * iklmn] * in_ptr[ijklmn - l_stride + k_stride]
                  + coeff[4 + ARAKAWA_STENCIL_SIZE * iklmn] * in_ptr[ijklmn - 2 * k_stride]
                  + coeff[5 + ARAKAWA_STENCIL_SIZE * iklmn] * in_ptr[ijklmn - k_stride]
                  + coeff[6 + ARAKAWA_STENCIL_SIZE * iklmn] * in_ptr[ijklmn]
                  + coeff[7 + ARAKAWA_STENCIL_SIZE * iklmn] * in_ptr[ijklmn + k_stride]
                  + coeff[8 + ARAKAWA_STENCIL_SIZE * iklmn] * in_ptr[ijklmn + 2 * k_stride]
                  + coeff[9 + ARAKAWA_STENCIL_SIZE * iklmn] * in_ptr[ijklmn + l_stride - k_stride]
                  + coeff[10 + ARAKAWA_STENCIL_SIZE * iklmn] * in_ptr[ijklmn + l_stride]
                  + coeff[11 + ARAKAWA_STENCIL_SIZE * iklmn] * in_ptr[ijklmn + l_stride + k_stride]
                  + coeff[12 + ARAKAWA_STENCIL_SIZE * iklmn] * in_ptr[ijklmn + 2 * l_stride];
}

/**
 * @brief times func and prints stats
 * 
 * @param func the func to run
 * @param b_decomp the brick decomposition used
 * @param tot_elems the number of elements
 */
void time_and_print_mpi_stats(std::function<void(void)> func, GENEBrickDecomp b_decomp, double tot_elems) {
  // time function
  int it = 100;  //< TODO: read from cmdline
  int warmup = 5;  //<  TODO: read from cmdline
  int cnt = it * NUM_GHOST_ELEMENTS / 2;
  for(int i = 0; i < warmup; ++i) func();
  packtime = calltime = waittime = movetime = calctime = 0;
  double start = omp_get_wtime(), end;
  for (int i = 0; i < it; ++i) func();
  end = omp_get_wtime();

  size_t tsize = 0;
  for (auto g: b_decomp.ghost)
    tsize += g.len * FieldBrick_kl::BRICKSIZE * sizeof(bElem);

  double total = (end - start) / cnt;
  mpi_stats calc_s = mpi_statistics(calctime / cnt, MPI_COMM_WORLD);
  mpi_stats pack_s = mpi_statistics(packtime / cnt, MPI_COMM_WORLD);
  mpi_stats pspd_s = mpi_statistics(tsize / 1.0e9 / packtime * cnt, MPI_COMM_WORLD);
  mpi_stats call_s = mpi_statistics(calltime / cnt, MPI_COMM_WORLD);
  mpi_stats wait_s = mpi_statistics(waittime / cnt, MPI_COMM_WORLD);
  mpi_stats mspd_s = mpi_statistics(tsize / 1.0e9 / (calltime + waittime) * cnt, MPI_COMM_WORLD);
  mpi_stats size_s = mpi_statistics((double) tsize * 1.0e-6, MPI_COMM_WORLD);
  total = calc_s.avg + wait_s.avg + call_s.avg + pack_s.avg;

  int rank;
  check_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

  if (rank == 0) {
    std::cout << "Arr: " << total << std::endl;

    std::cout << "calc " << calc_s << std::endl;
    std::cout << "pack " << pack_s << std::endl;
    std::cout << "  | Pack speed (GB/s): " << pspd_s << std::endl;
    std::cout << "call " << call_s << std::endl;
    std::cout << "wait " << wait_s << std::endl;
    std::cout << "  | MPI size (MB): " << size_s << std::endl;
    std::cout << "  | MPI speed (GB/s): " << mspd_s << std::endl;

    double perf = (double) tot_elems * 1.0e-9;
    perf = perf / total;
    std::cout << "perf " << perf << " GStencil/s" << std::endl;
    std::cout << std::endl;
  }
}

/**
 * @brief perform semi-arakawa k-l derivative kernel weak-scaling benchmark
 * 
 * Uses array layout
 * 
 * @param[out] out_ptr output data
 * @param[in] in_ptr input data
 * @param[in] coeffs input coefficients
 * @param[in] b_decomp the brick decomposition
 * @param[in] num_procs_per_dim number of processes in each dimension of cartesian communicator
 * @param[in] per_process_extent extent in each dimension (per MPI process)
 */
void semi_arakawa_distributed_array(bComplexElem *out_ptr,
                                    bComplexElem *in_ptr,
                                    bElem * coeffs,
                                    GENEBrickDecomp b_decomp,
                                    std::array<int, DIM> num_procs_per_dim,
                                    std::array<int, DIM> per_process_extent) {
  // set up MPI types for transfer
  std::unordered_map<uint64_t, MPI_Datatype> stypemap;
  std::unordered_map<uint64_t, MPI_Datatype> rtypemap;
  std::vector<long> array_padding = {PADDING_i + GHOST_ZONE_i,
                                     PADDING_j + GHOST_ZONE_j,
                                     PADDING_k,
                                     PADDING_l,
                                     PADDING_m + GHOST_ZONE_m,
                                     PADDING_n + GHOST_ZONE_n},
                    array_ghost_zone = {0, 0, GHOST_ZONE_k, GHOST_ZONE_l, 0, 0},
                    per_process_extent_long(per_process_extent.begin(), per_process_extent.end());
  exchangeArrPrepareTypes<DIM, bComplexElem>(stypemap, rtypemap, per_process_extent_long, array_padding, array_ghost_zone);
  // set up in/out ptrs on device
  std::array<long, DIM> padded_array_extent, coeff_extent;
  for(int i = 0; i < DIM; ++i) {
    padded_array_extent[i] = 2 * (array_padding[i] + array_ghost_zone[i]) + per_process_extent[i];
    coeff_extent[i] = per_process_extent[i];
  }
  coeff_extent[0] = ARAKAWA_STENCIL_SIZE;
  coeff_extent[1] = per_process_extent[0];

  bComplexElem *in_ptr_dev, *out_ptr_dev;
  std::vector<long> padded_array_extent_long(padded_array_extent.begin(), padded_array_extent.end()),
                    coeff_extent_long(coeff_extent.begin(), coeff_extent.end());
  copyToDevice(padded_array_extent_long, in_ptr_dev, in_ptr);
  copyToDevice(padded_array_extent_long, out_ptr_dev, out_ptr);
  bElem *coeff_dev;
  copyToDevice(coeff_extent_long, coeff_dev, coeffs);

  // build function to perform computation
  auto arr_func = [&padded_array_extent_long,
                   &per_process_extent_long,
                   &array_padding,
                   &array_ghost_zone,
                   &b_decomp,
                   &in_ptr,
                   &in_ptr_dev,
                   &out_ptr_dev,
                   &coeff_dev]() -> void {
    float elapsed;
    cudaEvent_t c_0, c_1;
    cudaEventCreate(&c_0);
    cudaEventCreate(&c_1);
#if !defined(CUDA_AWARE) || !defined(USE_TYPES)
    // Copy everything back from device
    double st = omp_get_wtime();
    copyFromDevice(padded_array_extent_long, in_ptr, in_ptr_dev);
    movetime += omp_get_wtime() - st;
    exchangeArr<DIM>(in_ptr, b_decomp.comm, b_decomp.rank_map,
                      per_process_extent_long,
                      array_padding,
                      array_ghost_zone);
    st = omp_get_wtime();
    copyToDevice(padded_array_extent_long, in_ptr_dev, in_ptr);
    movetime += omp_get_wtime() - st;
#else
    exchangeArrTypes<DIM>(in_ptr_dev, b_decomp.comm, b_decomp.rank_map, stypemap, rtypemap);
#endif
    cudaEventRecord(c_0);
    dim3 block(TILE, TILE),
          grid((padded_array_extent_long[2] + block.x - 1) / block.x,
              (padded_array_extent_long[3] + block.y - 1) / block.y,
              (  padded_array_extent_long[0] * padded_array_extent_long[1]
               * padded_array_extent_long[4] * padded_array_extent_long[5] + block.z - 1
               ) / block.z);
    for (int i = 0; i < NUM_GHOST_ELEMENTS / 4; ++i) {
      semi_arakawa_arr_kernel << < grid, block>> > (out_ptr_dev, in_ptr_dev, coeff_dev);
      semi_arakawa_arr_kernel << < grid, block>> > (in_ptr_dev, out_ptr_dev, coeff_dev);
    }
    if(NUM_GHOST_ELEMENTS % 4 == 2) semi_arakawa_arr_kernel << < grid, block>> > (out_ptr_dev, in_ptr_dev, coeff_dev);
    cudaEventRecord(c_1);
    cudaEventSynchronize(c_1);
    cudaEventElapsedTime(&elapsed, c_0, c_1);
    calctime += elapsed / 1000.0;
  };

  int rank;
  check_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  if (rank == 0)
    std::cout << "array MPI decomp" << std::endl;
  size_t tot_num_elements = std::accumulate(per_process_extent.begin(), per_process_extent.end(), 1, std::multiplies<size_t>())
                          * std::accumulate(num_procs_per_dim.begin(), num_procs_per_dim.end(), 1, std::multiplies<size_t>());
  time_and_print_mpi_stats(arr_func, b_decomp, tot_num_elements);

  // Copy back
  if(NUM_GHOST_ELEMENTS % 4 == 2) copyFromDevice(padded_array_extent_long, out_ptr, out_ptr_dev);
  else copyFromDevice(padded_array_extent_long, out_ptr, in_ptr_dev);
  
  // free memory
  cudaCheck(cudaFree(coeff_dev));
  cudaCheck(cudaFree(out_ptr_dev));
  cudaCheck(cudaFree(in_ptr_dev));
  cudaCheck(cudaFree(array_stride_dev));
}

/**
 * @brief Run weak-scaling gene6d benchmark
 */
int main(int argc, char **argv) {
  int provided;
  // setup MPI environment
  check_MPI(MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided));
  if (provided != MPI_THREAD_SERIALIZED) {
    check_MPI(MPI_Finalize());
    return 1;
  }
  // TODO: get per-process extent and number of processors per-dimension from
  //       cmdline
  std::array<int, DIM> num_procs_per_dim, global_extent, per_process_extent;
  per_process_extent = {70, 16, 24, 48, 32, 2};
  num_procs_per_dim = {1, 1, 3, 1, 2, 1};
  for(int i = 0; i < DIM; ++i) {
    global_extent[i] = per_process_extent[i] * num_procs_per_dim[i];
  }

  // Print information about setup (copied from Tuowen Zhao's args.cpp)
  // https://github.com/CtopCsUtahEdu/bricklib/blob/ef28a307962fe319cd723a589df4ff6fb4a75d18/weak/args.cpp#L133-L144
  int rank, size;
  check_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  check_MPI(MPI_Comm_size(MPI_COMM_WORLD, &size));
  if (rank == 0) {
    int numthreads;
    #pragma omp parallel shared(numthreads) default(none)
    numthreads = omp_get_num_threads();
    long page_size = sysconf(_SC_PAGESIZE);
    size_t tot_elems = std::accumulate(global_extent.begin(), global_extent.end(), 1, std::multiplies<size_t>());
    std::cout << "Pagesize       : " << page_size << "\n"
              << "MPI Processes  : " << size << "\n"
              << "OpenMP threads : " << numthreads << "\n" 
              << "Domain size of : " << tot_elems << "( ";
    for(int i = 0; i < DIM; ++i) {
      std::cout << std::setw(2) << per_process_extent[i];
      if(i < DIM - 1) std::cout << " x ";
    }
    std::cout << " per MPI process )\n"
              << "A total of " << size << " MPI processes ( ";
    for(int i = 0; i < DIM; ++i) {
      std::cout << std::setw(2) << num_procs_per_dim[i];
      if(i < DIM - 1) std::cout << " x ";
    }
    std::cout << " )\n"
              << "Brick Size: ";
    for(int i = 0; i < DIM; ++i) {
      std::cout << std::setw(2) << BDIM_arr[i];
      if(i < DIM - 1) std::cout << " x ";
    }
    // TODO: Get ghost-layer thickness from args
    NUM_GHOST_ELEMENTS = 4;
    if(NUM_GHOST_ELEMENTS % 2 != 0) throw std::runtime_error("NUM_GHOST_ELEMENTS must be even");
    std::cout << "\n"
              << "Iters Between exchanges : " << NUM_GHOST_ELEMENTS / 2 << "\n"
              << "Num Exchanges : " << 100 << std::endl; ///< TODO: read from cmd line
  }

  // build cartesian communicator and setup MEMFD
  MPI_Comm cartesian_comm = build_cartesian_comm(num_procs_per_dim, per_process_extent);
  MEMFD::setup_prefix("weak/gene-6d-main", rank);
  // get array/brick extents set up for my MPI process (all include ghost-zones)
  std::array<int, DIM> array_extent,
                       array_stride,
                       padded_array_extent,
                       brick_grid_extent;
  for(int i = 0; i < DIM; ++i) {
    array_extent[i] = per_process_extent[i] + 2 * GHOST_ZONE_arr[i];
    if(i == 0) array_stride[0] = 1;
    else array_stride[i] = array_stride[i-1] * array_extent[i];
    padded_array_extent[i] = array_extent[i] + 2 * PADDING_arr[i];
    if(array_extent[i] % BDIM_arr[i] != 0) {
      throw std::runtime_error("brick-dimension does not divide per-process extent");
    }
    brick_grid_extent[i] = array_extent[i] / BDIM_arr[i];
  }
  cudaCheck(cudaMemcpyToSymbol(array_extent_dev, array_extent.data(), DIM * sizeof(unsigned), cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpyToSymbol(array_stride_dev, array_stride.data(), DIM * sizeof(unsigned), cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpyToSymbol(per_process_extent_dev, per_process_extent.data(), DIM * sizeof(unsigned), cudaMemcpyHostToDevice));
  // set up shape of coeffs
  std::array<int, DIM> coeff_extent = per_process_extent;
  coeff_extent[0] = ARAKAWA_STENCIL_SIZE;
  coeff_extent[1] = per_process_extent[0];
  // initialize my part of the grid to random data
  bComplexElem *in_ptr = randomComplexArray(std::vector<long>(padded_array_extent.begin(), padded_array_extent.end()));
  bElem *coeffs = randomArray(std::vector<long>(coeff_extent.begin(), coeff_extent.end()));
  bComplexElem *out_ptr = zeroComplexArray(std::vector<long>(padded_array_extent.begin(), padded_array_extent.end()));

  // TODO: Actually populate this
  GENEBrickDecomp b_decomp(std::vector<unsigned>(per_process_extent.begin(), per_process_extent.end()),
                           {0, 0, GHOST_ZONE_k, GHOST_ZONE_l, 0, 0}
                           );

  // run array computation
  semi_arakawa_distributed_array(out_ptr, in_ptr, coeffs, b_decomp, num_procs_per_dim, per_process_extent);

  // free memory
  free(out_ptr);
  free(in_ptr);
  return 0;
}
