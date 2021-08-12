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
unsigned NUM_EXCHANGES; ///< how many mpi exchanges?
typedef BrickDecomp<Dim<BDIM_n, BDIM_m, BDIM_l, BDIM_k, BDIM_j, BDIM_i> > GENEBrickDecomp;

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

__constant__ __device__ unsigned array_extent_dev[DIM]; ///< per-process with ghost-zones
__constant__ __device__ unsigned padded_array_extent_dev[DIM]; ///< per-process with ghost-zones and padding
__constant__ __device__ unsigned padded_array_stride_dev[DIM]; ///< per-process with ghost-zones and padding
__constant__ __device__ unsigned per_process_extent_dev[DIM]; ///< per-process without ghost-zones

/**
 * @brief cuda kernel to compute k-l arakawa derivative (array layout)
 * 
 * Should be invoked 1 thread per array element (not including ghosts or padding)
 * global thread idx should by x.y.z = K.L.IJMN
 * 
 * @param out_ptr output array of shape padded_array_extent_dev
 * @param in_ptr input array of shape padded_array_extent_dev
 * @param coeff stencil coefficients of shape (13,I,K,L,M,N) (no padding, no ghosts)
 */
__global__
void semi_arakawa_arr_kernel(bComplexElem * __restrict__ out_ptr,
                             const bComplexElem * __restrict__ in_ptr,
                             bElem * __restrict__ coeff
                             )
{
  size_t global_z_idx = threadIdx.z + blockIdx.z * blockDim.z;
  unsigned i = global_z_idx % array_extent_dev[0];
  unsigned j = (global_z_idx / array_extent_dev[0]) % array_extent_dev[1];
  unsigned m = (global_z_idx / array_extent_dev[0] / array_extent_dev[1]) % array_extent_dev[4];
  unsigned n = (global_z_idx / array_extent_dev[0] / array_extent_dev[1] / array_extent_dev[4]);
  unsigned k = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned l = threadIdx.y + blockIdx.y * blockDim.y;
  size_t padded_ijklmn = PADDING_i + GHOST_ZONE_i + i 
                       + padded_array_extent_dev[0] * (PADDING_j + GHOST_ZONE_j + j
                       + padded_array_extent_dev[1] * (PADDING_k + GHOST_ZONE_k + k
                       + padded_array_extent_dev[2] * (PADDING_l + GHOST_ZONE_l + l
                       + padded_array_extent_dev[3] * (PADDING_m + GHOST_ZONE_m + m
                       + padded_array_extent_dev[4] * (PADDING_n + GHOST_ZONE_n + n
                       )))));
  size_t unpadded_iklmn = i
                        + per_process_extent_dev[0] * (j
                        + per_process_extent_dev[1] * (k
                        + per_process_extent_dev[2] * (l
                        + per_process_extent_dev[3] * (m
                        + per_process_extent_dev[4] * (n
                        )))));
  const size_t k_stride = padded_array_stride_dev[2];
  const size_t l_stride = padded_array_stride_dev[3];
  out_ptr[padded_ijklmn] = coeff[ 0 + ARAKAWA_STENCIL_SIZE * unpadded_iklmn] * in_ptr[padded_ijklmn - 2 * l_stride]
                         + coeff[ 1 + ARAKAWA_STENCIL_SIZE * unpadded_iklmn] * in_ptr[padded_ijklmn - l_stride - k_stride]
                         + coeff[ 2 + ARAKAWA_STENCIL_SIZE * unpadded_iklmn] * in_ptr[padded_ijklmn - l_stride]
                         + coeff[ 3 + ARAKAWA_STENCIL_SIZE * unpadded_iklmn] * in_ptr[padded_ijklmn - l_stride + k_stride]
                         + coeff[ 4 + ARAKAWA_STENCIL_SIZE * unpadded_iklmn] * in_ptr[padded_ijklmn - 2 * k_stride]
                         + coeff[ 5 + ARAKAWA_STENCIL_SIZE * unpadded_iklmn] * in_ptr[padded_ijklmn - k_stride]
                         + coeff[ 6 + ARAKAWA_STENCIL_SIZE * unpadded_iklmn] * in_ptr[padded_ijklmn]
                         + coeff[ 7 + ARAKAWA_STENCIL_SIZE * unpadded_iklmn] * in_ptr[padded_ijklmn + k_stride]
                         + coeff[ 8 + ARAKAWA_STENCIL_SIZE * unpadded_iklmn] * in_ptr[padded_ijklmn + 2 * k_stride]
                         + coeff[ 9 + ARAKAWA_STENCIL_SIZE * unpadded_iklmn] * in_ptr[padded_ijklmn + l_stride - k_stride]
                         + coeff[10 + ARAKAWA_STENCIL_SIZE * unpadded_iklmn] * in_ptr[padded_ijklmn + l_stride]
                         + coeff[11 + ARAKAWA_STENCIL_SIZE * unpadded_iklmn] * in_ptr[padded_ijklmn + l_stride + k_stride]
                         + coeff[12 + ARAKAWA_STENCIL_SIZE * unpadded_iklmn] * in_ptr[padded_ijklmn + 2 * l_stride];
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
  int warmup = 5;  //<  TODO: read from cmdline
  int cnt = NUM_EXCHANGES * (NUM_GHOST_ELEMENTS / 2);
  for(int i = 0; i < warmup; ++i) func();
  packtime = calltime = waittime = movetime = calctime = 0;
  double start = omp_get_wtime(), end;
  for (int i = 0; i < NUM_EXCHANGES; ++i) func();
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

  bComplexElem *in_ptr_dev = nullptr, *out_ptr_dev = nullptr;
  std::vector<long> padded_array_extent_long(padded_array_extent.begin(), padded_array_extent.end()),
                    coeff_extent_long(coeff_extent.begin(), coeff_extent.end());
  copyToDevice(padded_array_extent_long, in_ptr_dev, in_ptr);
  copyToDevice(padded_array_extent_long, out_ptr_dev, out_ptr);
  bElem *coeff_dev = nullptr;
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
}

/**
 * @brief Reads a tuple of unsigneds delimited by delim
 * 
 * @param in the input stream to read from
 * @param delim the delimiter between unsigneds
 * @return std::vector<unsigned> of the values read in
 */
std::vector<unsigned> read_uint_tuple(std::istream &in = std::cin, char delim = ',') {
  std::vector<unsigned> tuple;
  unsigned value;
  do {
    if(in.peek() == delim) in.get();
    in >> value;
    tuple.push_back(value);
  } while(in.peek() == delim);
  return tuple;
}

/**
 * @brief parse args
 * @param[out] per_process_domain_size the extent in each dimension of the domain
 * @param[out] num_procs_per_dim the number of processes per dimension
 * @param[in] in input stream to read from
 * 
 * @return the number of iterations, with default 100
 */
unsigned parse_args(std::array<int, DIM> *per_process_domain_size,
                    std::array<int, DIM> *num_procs_per_dim,
                    std::istream &in = std::cin)
{
  std::string option_string;
  unsigned num_iters;
  std::vector<unsigned> tuple;
  bool read_dom_size = false, read_num_iters = false, read_num_procs_per_dim = false;
  std::string help_string = "Program options\n"
      "  -h: show help (this message)\n"
      "  Domain size,  in array order contiguous first\n"
      "  -d: comma separated Int[6], per-process domain size\n"
      "  Num Tasks per dimension, in array order contiguous first\n"
      "  -p: comma separated Int[6], num process per dimension"
      "  Benchmark control:\n"
      "  -I: number of iterations, default 100 \n"
      "Example usage:\n"
      "  weak/gene6d -d 70,16,24,48,32,2 -p 1,1,3,1,2,1\n";
  std::ostringstream error_stream;
  while(in >> option_string) {
    if(option_string[0] != '-' || option_string.size() != 2) {
      error_stream << "Unrecognized option " << option_string << std::endl;
    }
    if(error_stream.str().size() != 0) {
      error_stream << help_string;
      throw std::runtime_error(error_stream.str());
    }
    switch(option_string[1]) {
      case 'd': 
        tuple = read_uint_tuple(in, ',');
        if(read_dom_size) {
          error_stream << "-d option should only be passed once" << std::endl;
        } else if(tuple.size() != DIM) {
          error_stream << "Expected extent of length " << DIM << ", not " << tuple.size();
        } else {
          std::copy(tuple.begin(), tuple.end(), per_process_domain_size->begin());
        }
        read_dom_size = true;
        break;
      case 'p':
        tuple = read_uint_tuple(in, ',');
        if(read_num_procs_per_dim) {
          error_stream << "-p option should only be passed once" << std::endl;
        }
        else if(tuple.size() != DIM) {
          error_stream << "Expected num procs per dim of length " << DIM << ", not " << tuple.size();
        } else {
          std::copy(tuple.begin(), tuple.end(), num_procs_per_dim->begin());
        }
        read_num_procs_per_dim = true;
        break;
      case 'I':
        if(read_num_iters) {
          error_stream << "-I option should only be passed once" << std::endl;
        } else {
          in >> num_iters;
        }
        read_num_iters = true;
        break;
      default:
        error_stream << "Unrecognized option " << option_string << std::endl;
    }
  }
  if(!read_num_procs_per_dim) {
    throw std::runtime_error("Missing -p option");
  }
  if(!read_dom_size) {
    throw std::runtime_error("Missing -d option");
  }
  return num_iters;
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
  parse_args(&per_process_extent, &num_procs_per_dim);
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
              << "Num Exchanges : " << NUM_EXCHANGES << std::endl; 
  }

  // build cartesian communicator and setup MEMFD
  MPI_Comm cartesian_comm = build_cartesian_comm(num_procs_per_dim, per_process_extent);
  MEMFD::setup_prefix("weak/gene-6d-main", rank);
  // get array/brick extents set up for my MPI process (all include ghost-zones)
  std::array<int, DIM> array_extent,
                       padded_array_stride,
                       padded_array_extent,
                       brick_grid_extent;
  for(int i = 0; i < DIM; ++i) {
    array_extent[i] = per_process_extent[i] + 2 * GHOST_ZONE_arr[i];
    if(i == 0) padded_array_stride[0] = 1;
    else padded_array_stride[i] = padded_array_stride[i-1] * padded_array_extent[i];
    padded_array_extent[i] = array_extent[i] + 2 * PADDING_arr[i];
    if(per_process_extent[i] % BDIM_arr[i] != 0) {
      std::ostringstream error_stream;
      error_stream << "Brick-dimension " << i << " (" << BDIM_arr[i] << ")"
                   << " does not divide per-process extent " << i << " (" << per_process_extent[i] << ")";
      throw std::runtime_error(error_stream.str());
    }
    if(GHOST_ZONE_arr[i] % BDIM_arr[i] != 0) {
      std::ostringstream error_stream;
      error_stream << "Brick-dimension " << i << " (" << BDIM_arr[i] << ")"
                   << " does not divide ghost-zone " << i << " (" << GHOST_ZONE_arr[i] << ")";
      throw std::runtime_error(error_stream.str());
    }
    brick_grid_extent[i] = array_extent[i] / BDIM_arr[i];
  }
  cudaCheck(cudaMemcpyToSymbol(padded_array_stride_dev, padded_array_stride.data(), DIM * sizeof(unsigned)));
  cudaCheck(cudaMemcpyToSymbol(array_extent_dev, array_extent.data(), DIM * sizeof(unsigned)));
  cudaCheck(cudaMemcpyToSymbol(padded_array_extent_dev, padded_array_extent.data(), DIM * sizeof(unsigned)));
  cudaCheck(cudaMemcpyToSymbol(per_process_extent_dev, per_process_extent.data(), DIM * sizeof(unsigned)));
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
