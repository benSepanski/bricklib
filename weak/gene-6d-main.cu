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
#include <numeric>

#include <brick.h>
#include <brick-mpi.h>
#include <bricksetup.h>
#include <brick-cuda.h>

#include "bitset.h"
#include <multiarray.h>
#include <brickcompare.h>
#include "stencils/cudaarray.h"

#include <unistd.h>
#include <array-mpi.h>

// useful constants
constexpr unsigned DIM = 6;
constexpr std::array<unsigned, DIM> BRICK_DIM = {2, 16, 2, 2, 1, 1};
constexpr std::array<unsigned, DIM> COEFF_BRICK_DIM = {1, BRICK_DIM[0], BRICK_DIM[2], BRICK_DIM[3], BRICK_DIM[4], BRICK_DIM[5]};
constexpr unsigned NUM_GHOST_ZONES = 1;
constexpr std::array<unsigned, DIM> GHOST_ZONE = {0, 0, 2 * NUM_GHOST_ZONES, 2 * NUM_GHOST_ZONES, 0, 0};
constexpr std::array<unsigned, DIM> PADDING = {0, 0, 0, 0, 0, 0};
constexpr unsigned TILE_SIZE = 8;
constexpr unsigned ARAKAWA_STENCIL_SIZE = 13;

// check constants
static_assert(GHOST_ZONE[0] % BRICK_DIM[0] == 0);
static_assert(GHOST_ZONE[1] % BRICK_DIM[1] == 0);
static_assert(GHOST_ZONE[2] % BRICK_DIM[2] == 0);
static_assert(GHOST_ZONE[3] % BRICK_DIM[3] == 0);
static_assert(GHOST_ZONE[4] % BRICK_DIM[4] == 0);
static_assert(GHOST_ZONE[5] % BRICK_DIM[5] == 0);

// useful types
typedef Dim<BRICK_DIM[5], BRICK_DIM[4], BRICK_DIM[3], BRICK_DIM[2], BRICK_DIM[1], BRICK_DIM[0]> FieldBrickDimsType;
typedef Dim<COEFF_BRICK_DIM[5], COEFF_BRICK_DIM[4], COEFF_BRICK_DIM[3], COEFF_BRICK_DIM[2], COEFF_BRICK_DIM[1], COEFF_BRICK_DIM[0]> CoeffBrickDimsType;
typedef Dim<1> VectorFoldType;
typedef CommDims<false, false, true, true, false, false> CommIn_kl;
typedef CommDims<false, false, false, false, false, false> NoComm;
typedef Brick<FieldBrickDimsType, VectorFoldType, true, CommIn_kl> FieldBrick_kl;
typedef Brick<CoeffBrickDimsType, VectorFoldType, false, NoComm> RealCoeffBrick;
typedef BrickDecomp<FieldBrickDimsType, CommIn_kl> GENEBrickDecomp;

// global constants set by CLI
unsigned NUM_EXCHANGES; ///< how many mpi exchanges?

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
  std::array<int, DIM> periodic;
  for(int i = 0; i < DIM; ++i) {
    periodic[i] = true;
  }
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
  int cnt = NUM_EXCHANGES * NUM_GHOST_ZONES;
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

__device__ __constant__ unsigned array_extent_with_gz_dev[DIM];
__device__ __constant__ unsigned array_extent_with_padding_dev[DIM];

/**
 * @brief cuda kernel to compute k-l arakawa derivative (array layout)
 * 
 * Should be invoked 1 thread per array element (including ghosts, but not padding)
 * global thread idx should by x.y.z = K.L.IJMN
 * 
 * @param out_ptr output array of shape array_extent_with_padding_dev
 * @param in_ptr input array of shape array_extent_with_padding_dev
 * @param coeff stencil coefficients of shape (13,)+array_extent_with_gz_dev
 */
__global__
void semi_arakawa_arr_kernel(bComplexElem * __restrict__ out_ptr,
                             const bComplexElem * __restrict__ in_ptr,
                             bElem * __restrict__ coeff)
{
  // convenient aliases
  unsigned * extent_with_gz = array_extent_with_gz_dev;
  unsigned * extent_with_padding = array_extent_with_padding_dev;

  size_t global_z_idx = threadIdx.z + blockIdx.z * blockDim.z;
  unsigned i = global_z_idx % extent_with_gz[0];
  unsigned j = (global_z_idx / extent_with_gz[0]) % extent_with_gz[1];
  unsigned m = (global_z_idx / extent_with_gz[0] / extent_with_gz[1]) % extent_with_gz[4];
  unsigned n = (global_z_idx / extent_with_gz[0] / extent_with_gz[1] / extent_with_gz[4]);
  unsigned k = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned l = threadIdx.y + blockIdx.y * blockDim.y;
  size_t padded_ijklmn = PADDING[0] + GHOST_ZONE[0] + i 
                       + extent_with_padding[0] * (PADDING[1] + GHOST_ZONE[1] + j
                       + extent_with_padding[1] * (PADDING[2] + GHOST_ZONE[2] + k
                       + extent_with_padding[2] * (PADDING[3] + GHOST_ZONE[3] + l
                       + extent_with_padding[3] * (PADDING[4] + GHOST_ZONE[4] + m
                       + extent_with_padding[4] * (PADDING[5] + GHOST_ZONE[5] + n
                       )))));
  size_t unpadded_iklmn = i
                        + extent_with_gz[0] * (j
                        + extent_with_gz[1] * (k
                        + extent_with_gz[2] * (l
                        + extent_with_gz[3] * (m
                        + extent_with_gz[4] * (n
                        )))));
  const size_t k_stride = extent_with_padding[1] * extent_with_padding[0];
  const size_t l_stride = extent_with_padding[2] * k_stride;
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
 * @brief perform semi-arakawa k-l derivative kernel weak-scaling benchmark
 * 
 * Uses array layout
 * 
 * @param[out] out_ptr output data (has ghost-zones and padding)
 * @param[in] in_ptr input data (has ghost-zones and padding)
 * @param[in] coeffs input coefficients (has ghost-zones but no padding)
 * @param[in] b_decomp the brick decomposition
 * @param[in] num_procs_per_dim number of processes in each dimension of cartesian communicator
 * @param[in] extent extent in each dimension (per MPI process) without ghost-zone/padding
 */
void semi_arakawa_distributed_array(bComplexElem *out_ptr,
                                    bComplexElem *in_ptr,
                                    bElem * coeffs,
                                    GENEBrickDecomp b_decomp,
                                    std::array<int, DIM> num_procs_per_dim,
                                    std::array<int, DIM> extent) {
  // set up MPI types for transfer
  std::unordered_map<uint64_t, MPI_Datatype> stypemap;
  std::unordered_map<uint64_t, MPI_Datatype> rtypemap;
  // get arrays as vectors
  std::vector<long> extent_as_vector(extent.begin(), extent.end()),
                    padding_as_vector(PADDING.begin(), PADDING.end()),
                    ghost_zone_as_vector(GHOST_ZONE.begin(), GHOST_ZONE.end());
  exchangeArrPrepareTypes<DIM, bComplexElem>(stypemap,
                                             rtypemap,
                                             extent_as_vector,
                                             padding_as_vector,
                                             ghost_zone_as_vector);
  // set up in/out ptrs on device
  std::array<long, DIM> extent_with_gz, extent_with_padding, coeff_extent;
  for(int i = 0; i < DIM; ++i) {
    extent_with_gz[i] = 2 * GHOST_ZONE[i] + extent[i];
    extent_with_padding[i] = 2 * PADDING[i] + extent_with_gz[i];
  }
  coeff_extent[0] = ARAKAWA_STENCIL_SIZE;
  coeff_extent[1] = extent_with_gz[0];
  for(int i = 2; i < DIM; ++i) {
    coeff_extent[i] = extent_with_gz[i];
  }

  bComplexElem *in_ptr_dev = nullptr, *out_ptr_dev = nullptr;
  std::vector<long> extent_with_padding_as_vector(extent_with_padding.begin(), extent_with_padding.end()),
                    coeff_extent_as_vector(coeff_extent.begin(), coeff_extent.end());
  copyToDevice(extent_with_padding_as_vector, in_ptr_dev,  in_ptr);
  copyToDevice(extent_with_padding_as_vector, out_ptr_dev, out_ptr);
  bElem *coeff_dev = nullptr;
  copyToDevice(coeff_extent_as_vector, coeff_dev, coeffs);

  // build function to perform computation
  auto arr_func = [&extent_with_gz,
                   &extent_with_padding_as_vector,
                   &extent_as_vector,
                   &ghost_zone_as_vector,
                   &padding_as_vector,
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
    copyFromDevice(extent_with_padding_as_vector, in_ptr, in_ptr_dev);
    movetime += omp_get_wtime() - st;
    exchangeArr<DIM>(in_ptr,
                     b_decomp.comm,
                     b_decomp.rank_map,
                     extent_as_vector,
                     padding_as_vector,
                     ghost_zone_as_vector);
    st = omp_get_wtime();
    copyToDevice(extent_with_padding_as_vector, in_ptr_dev, in_ptr);
    movetime += omp_get_wtime() - st;
#else
    exchangeArrTypes<DIM>(in_ptr_dev, b_decomp.comm, b_decomp.rank_map, stypemap, rtypemap);
#endif
    cudaEventRecord(c_0);
    dim3 block(TILE_SIZE, TILE_SIZE),
          grid((extent_with_gz[2] + block.x - 1) / block.x,
               (extent_with_gz[3] + block.y - 1) / block.y,
               (extent_with_gz[0] * extent_with_gz[1] * extent_with_gz[3] * extent_with_gz[4]) / block.z);
    for (int i = 0; i < NUM_GHOST_ZONES; ++i) {
      semi_arakawa_arr_kernel << < grid, block>> > (out_ptr_dev, in_ptr_dev, coeff_dev);
      if(i + 1 < NUM_GHOST_ZONES) { 
        std::swap(out_ptr_dev, in_ptr_dev);
      }
    }
    cudaEventRecord(c_1);
    cudaEventSynchronize(c_1);
    cudaEventElapsedTime(&elapsed, c_0, c_1);
    calctime += elapsed / 1000.0;
  };

  int rank;
  check_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  if (rank == 0)
    std::cout << "array MPI decomp" << std::endl;
  size_t tot_num_elements = std::accumulate(extent.begin(), extent.end(), 1, std::multiplies<size_t>())
                          * std::accumulate(num_procs_per_dim.begin(), num_procs_per_dim.end(), 1, std::multiplies<size_t>());
  time_and_print_mpi_stats(arr_func, b_decomp, tot_num_elements);

  // Copy back
  copyFromDevice(extent_with_padding_as_vector, out_ptr, in_ptr_dev);
  
  // free memory
  cudaCheck(cudaFree(coeff_dev));
  cudaCheck(cudaFree(out_ptr_dev));
  cudaCheck(cudaFree(in_ptr_dev));
}

/**
 * @brief cuda kernel to compute k-l arakawa derivative (brick layout)
 * 
 * Should be invoked 1 thread per array element (including ghosts, but not padding)
 * and 1 thread-block per brick. Assumes thread-block index is K.L.IJMN
 * 
 * @param grid_ptr brick-grid for field bricks (includes ghost bricks)
 * @param coeff_grid_ptr brick-grid for coefficients (includes ghost bricks)
 * @param out_brick output brick
 * @param in_brick input brick
 * @param coeff_brick coefficients
 */
__global__
void semi_arakawa_brick_kernel(unsigned * grid_ptr,
                               unsigned * coeff_grid_ptr,
                               FieldBrick_kl out_brick,
                               FieldBrick_kl in_brick,
                               RealCoeffBrick coeff_brick)
{
   // TODO
}

/**
 * @brief perform semi-arakawa k-l derivative kernel weak-scaling benchmark
 * 
 * Uses bricks layout
 * 
 * @param[out] out_ptr output data (has ghost-zones and padding)
 * @param[in] in_ptr input data (has ghost-zones and padding)
 * @param[in] coeffs input coefficients (has ghost-zones but no padding)
 * @param[in] b_decomp the brick decomposition
 * @param[in] num_procs_per_dim number of processes in each dimension of cartesian communicator
 * @param[in] per_process_extent extent in each dimension (per MPI process)
 */
void semi_arakawa_distributed_brick(bComplexElem *out_ptr,
                                    bComplexElem *in_ptr,
                                    bElem * coeffs,
                                    GENEBrickDecomp b_decomp,
                                    std::array<int, DIM> num_procs_per_dim,
                                    std::array<int, DIM> per_process_extent) {
  // set up brick-info and storage on host
  using FieldBrick = FieldBrick_kl;
  BrickInfo<DIM, CommIn_kl> b_info = b_decomp.getBrickInfo();
#ifdef DECOMP_PAGEUNALIGN
  BrickStorage b_storage = b_info.allocate(FieldBrick::BRICKSIZE);
  BrickStorage b_storage_out = b_info.allocate(FieldBrick::BRICKSIZE);
#else
  BrickStorage b_storage = b_info.mmap_alloc(FieldBrick::BRICKSIZE);
  BrickStorage b_storage_out = b_info.mmap_alloc(FieldBrick::BRICKSIZE);
#endif
  std::array<int, DIM> per_process_extent_with_gz, brick_grid_extent_with_gz;
  for(unsigned i = 0; i < per_process_extent.size(); ++i) {
    per_process_extent_with_gz[i] = per_process_extent[i] + 2 * GHOST_ZONE[i];
    if(per_process_extent_with_gz[i] % BRICK_DIM[i] != 0) {
      throw std::runtime_error("Per-process array extent with ghost-zones is not divisible by brick dimensions");
    }
    brick_grid_extent_with_gz[i] = per_process_extent_with_gz[i] / BRICK_DIM[i];
  }
  const size_t num_bricks = std::accumulate(brick_grid_extent_with_gz.begin(), brick_grid_extent_with_gz.end(), 1, std::multiplies<size_t>());
  unsigned *grid_ptr = (unsigned *) malloc(sizeof(unsigned) * num_bricks);
  auto grid = (unsigned (*)[brick_grid_extent_with_gz[4]]
                           [brick_grid_extent_with_gz[3]]
                           [brick_grid_extent_with_gz[2]]
                           [brick_grid_extent_with_gz[1]]
                           [brick_grid_extent_with_gz[0]]
               ) grid_ptr;
  for(size_t n = 0; n < brick_grid_extent_with_gz[5]; ++n)
  for(size_t m = 0; m < brick_grid_extent_with_gz[4]; ++m)
  for(size_t l = 0; l < brick_grid_extent_with_gz[3]; ++l)
  for(size_t k = 0; k < brick_grid_extent_with_gz[2]; ++k)
  for(size_t j = 0; j < brick_grid_extent_with_gz[1]; ++j)
  for(size_t i = 0; i < brick_grid_extent_with_gz[0]; ++i) {
    grid[n][m][l][k][j][i] = b_decomp[n][m][l][k][j][i];
  }

  // setup coefficient bricks metadata 
  std::array<unsigned, DIM> coeff_extent_with_gz;
  coeff_extent_with_gz[0] = ARAKAWA_STENCIL_SIZE;
  coeff_extent_with_gz[1] = per_process_extent_with_gz[0];
  for(unsigned i = 2; i < DIM; ++i) {
    coeff_extent_with_gz[i] = per_process_extent_with_gz[i];
  }
  std::vector<long> coeff_brick_grid_extent_with_gz;
  coeff_brick_grid_extent_with_gz.reserve(DIM);
  for(unsigned i = 0; i < DIM; ++i) {
    coeff_brick_grid_extent_with_gz[i] = coeff_extent_with_gz[i] / COEFF_BRICK_DIM[i];
  }
  const size_t num_coeff_bricks = std::accumulate(coeff_extent_with_gz.begin() + 1, coeff_extent_with_gz.end(),
                                                  1, std::multiplies<size_t>());
  unsigned *coeff_grid_ptr;
  BrickInfo<DIM, NoComm> coeff_b_info = init_grid<DIM, NoComm>(coeff_grid_ptr, coeff_brick_grid_extent_with_gz);

  // setup coeff bricks on host
  BrickStorage coeff_b_storage = coeff_b_info.allocate(RealCoeffBrick::BRICKSIZE);
  RealCoeffBrick coeff_brick(&coeff_b_info, coeff_b_storage, 0);
  std::vector<long> coeff_extent_with_gz_as_vector(coeff_extent_with_gz.begin(), coeff_extent_with_gz.end());
  copyToBrick<DIM>(coeff_extent_with_gz_as_vector,
                   coeffs,
                   coeff_grid_ptr,
                   coeff_brick);

  // setup field bricks on host and copy in data
  FieldBrick b_in(&b_info, b_storage, 0),
             b_out(&b_info, b_storage_out, 0);
  std::vector<long> per_process_extent_with_gz_as_vector(per_process_extent_with_gz.begin(), per_process_extent_with_gz.end()),
                    padding_as_vector(PADDING.begin(), PADDING.end());
  copyToBrick<DIM>(per_process_extent_with_gz_as_vector,
                   padding_as_vector,
                   std::vector<long>(DIM, 0),
                   in_ptr,
                   grid_ptr,
                   b_in);
  
  // move bricks to device
  BrickInfo<DIM, CommIn_kl> *b_info_dev;
  BrickInfo<DIM, CommIn_kl> _b_info_dev = movBrickInfo(b_info, cudaMemcpyHostToDevice);
  {
    unsigned b_info_size = sizeof(BrickInfo<DIM, CommIn_kl>);
    cudaMalloc(&b_info_dev, b_info_size);
    cudaMemcpy(b_info_dev, &_b_info_dev, b_info_size, cudaMemcpyHostToDevice);
  }
  BrickInfo<DIM, NoComm> *coeff_b_info_dev;
  BrickInfo<DIM, NoComm> _coeff_b_info_dev = movBrickInfo(coeff_b_info, cudaMemcpyHostToDevice);
  {
    unsigned coeff_b_info_size = sizeof(BrickInfo<DIM, NoComm>);
    cudaMalloc(&coeff_b_info_dev, coeff_b_info_size);
    cudaMemcpy(coeff_b_info_dev, &_coeff_b_info_dev, coeff_b_info_size, cudaMemcpyHostToDevice);
  }

  BrickStorage b_storage_dev = movBrickStorage(b_storage, cudaMemcpyHostToDevice),
               b_storage_out_dev = movBrickStorage(b_storage_out, cudaMemcpyHostToDevice),
               coeff_b_storage_dev = movBrickStorage(coeff_b_storage, cudaMemcpyHostToDevice);

  FieldBrick b_in_dev(b_info_dev, b_storage_dev, 0),
             b_out_dev(b_info_dev, b_storage_out_dev, 0);
  RealCoeffBrick coeff_brick_dev(&coeff_b_info, coeff_b_storage, 0);
  
  // copy grid to device
  unsigned *grid_ptr_dev = nullptr, 
           *coeff_grid_ptr_dev = nullptr;
  std::vector<long> brick_grid_extent_with_gz_as_vector(brick_grid_extent_with_gz.begin(), brick_grid_extent_with_gz.end());
  copyToDevice(brick_grid_extent_with_gz_as_vector, grid_ptr_dev, grid_ptr);
  copyToDevice(coeff_brick_grid_extent_with_gz, coeff_grid_ptr_dev, coeff_grid_ptr);

#ifndef DECOMP_PAGEUNALIGN
  ExchangeView ev = b_decomp.exchangeView(b_storage);
#endif
  // setup brick function to compute stencil
  auto brick_func = [&]() -> void {
    float elapsed;
    cudaEvent_t c_0, c_1;
    cudaEventCreate(&c_0);
    cudaEventCreate(&c_1);
#ifndef CUDA_AWARE
  {
    double t_a = omp_get_wtime();
    cudaCheck(cudaMemcpy(b_storage.dat.get() + b_storage.step * b_decomp.sep_pos[0],
                         b_storage_dev.dat.get() + b_storage.step * b_decomp.sep_pos[0],
                         b_storage.step * (b_decomp.sep_pos[1] - b_decomp.sep_pos[0]) * sizeof(bElem),
                         cudaMemcpyDeviceToHost));
    double t_b = omp_get_wtime();
    movetime += t_b - t_a;
  #ifdef DECOMP_PAGEUNALIGN
    b_decomp.exchange(b_storage);
  #else
    ev.exchange();
  #endif
    t_a = omp_get_wtime();
    cudaCheck(cudaMemcpy(b_storage_dev.dat.get() + b_storage.step * b_decomp.sep_pos[1],
                         b_storage.dat.get() + b_storage.step * b_decomp.sep_pos[1],
                         b_storage.step * (b_decomp.sep_pos[2] - b_decomp.sep_pos[1]) * sizeof(bElem),
                         cudaMemcpyHostToDevice));
    t_b = omp_get_wtime();
    movetime += t_b - t_a;
  }
#else
  bDecomp.exchange(bStorage_dev);
#endif
    std::array<unsigned, DIM> brick_grid_extent;
    for(unsigned i = 0; i < DIM; ++i) {
      brick_grid_extent[i] = per_process_extent[i] / BRICK_DIM[i];
    }
    dim3 cuda_grid_size(brick_grid_extent[2],
                        brick_grid_extent[3],
                        num_bricks / brick_grid_extent[2] / brick_grid_extent[3]),
         cuda_block_size(BRICK_DIM[0], BRICK_DIM[1], FieldBrick::BRICKLEN / BRICK_DIM[0] / BRICK_DIM[1]);
    cudaEventRecord(c_0);
    for (int i = 0; i < NUM_GHOST_ZONES; ++i) {
      semi_arakawa_brick_kernel << < cuda_grid_size, cuda_block_size>> >(grid_ptr_dev,
                                                                         coeff_grid_ptr_dev,
                                                                         b_out_dev,
                                                                         b_in_dev,
                                                                         coeff_brick_dev);
      if(i + 1 < NUM_GHOST_ZONES) { 
        std::swap(b_out_dev, b_in_dev);
        std::swap(b_storage, b_storage_out);
        std::swap(b_storage_dev, b_storage_out_dev);
      }
    }
    cudaEventRecord(c_1);
    cudaEventSynchronize(c_1);
    cudaEventElapsedTime(&elapsed, c_0, c_1);
    calctime += elapsed / 1000.0;
  };

  // time function
  int rank;
  check_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  if (rank == 0)
    std::cout << "brick MPI decomp" << std::endl;
  size_t tot_num_elements = std::accumulate(per_process_extent.begin(), per_process_extent.end(), 1, std::multiplies<size_t>())
                          * std::accumulate(num_procs_per_dim.begin(), num_procs_per_dim.end(), 1, std::multiplies<size_t>());
  time_and_print_mpi_stats(brick_func, b_decomp, tot_num_elements);

  // Copy back
  cudaCheck(cudaMemcpy(b_storage_out.dat.get(),
                       b_storage_out_dev.dat.get(),
                       b_storage.step * b_info.nbricks * sizeof(bElem),
                       cudaMemcpyDeviceToHost));

  std::vector<long> per_process_extent_as_vector(per_process_extent.begin(), per_process_extent.end()),
                    ghost_zone_as_vector(GHOST_ZONE.begin(), GHOST_ZONE.end());
  copyFromBrick<DIM>(per_process_extent_as_vector,
                     padding_as_vector,
                     ghost_zone_as_vector,
                     in_ptr,
                     grid_ptr,
                     b_in);

  // free memory
  cudaCheck(cudaFree(coeff_grid_ptr_dev));
  cudaCheck(cudaFree(grid_ptr_dev));
  cudaCheck(cudaFree(_b_info_dev.adj));
  cudaCheck(cudaFree(b_info_dev));
  cudaCheck(cudaFree(_coeff_b_info_dev.adj));
  cudaCheck(cudaFree(coeff_b_info_dev));
  free(coeff_b_info.adj);
  free(coeff_grid_ptr);
  free(grid_ptr);
}

/**
 * @brief Reads a tuple of unsigneds delimited by delim
 * 
 * @param in the input stream to read from
 * @param delim the delimiter between unsigneds
 * @return std::vector<unsigned> of the values read in
 */
std::vector<unsigned> read_uint_tuple(std::istream &in, char delim = ',') {
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
                    std::istream &in)
{
  std::string option_string;
  unsigned num_iters = 100;
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
    error_stream << "Missing -p option" << std::endl << help_string;
    throw std::runtime_error(error_stream.str());
  }
  if(!read_dom_size) {
    error_stream << "Missing -d option" << std::endl << help_string;
    throw std::runtime_error(error_stream.str());
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
  std::stringstream input_stream;
  for(int i = 1; i < argc; ++i) {
    input_stream << argv[i] << " ";
  }
  NUM_EXCHANGES = parse_args(&per_process_extent, &num_procs_per_dim, input_stream);
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
    int io_col_width = 30;
    std::cout << std::setw(io_col_width) << "Pagesize :" << page_size << "\n"
              << std::setw(io_col_width) << "MPI Processes :" << size << "\n"
              << std::setw(io_col_width) << "OpenMP threads :" << numthreads << "\n" 
              << std::setw(io_col_width) << "Domain size (per-process) of :";
    for(int i = 0; i < DIM; ++i) {
      std::cout << std::setw(2) << per_process_extent[i];
      if(i < DIM - 1) std::cout << " x ";
    }
    std::cout << std::endl
              << std::setw(io_col_width) << "Ghost Zone :";
    for(int i = 0; i < DIM; ++i) {
      std::cout << std::setw(2) << GHOST_ZONE[i];
      if(i < DIM - 1) std::cout << " x ";
    }
    std::cout << std::endl
              << std::setw(io_col_width) << "Array Padding :";
    for(int i = 0; i < DIM; ++i) {
      std::cout << std::setw(2) << PADDING[i];
      if(i < DIM - 1) std::cout << " x ";
    }
    std::cout << std::endl
              << std::setw(io_col_width) << "MPI Processes :";
    for(int i = 0; i < DIM; ++i) {
      std::cout << std::setw(2) << num_procs_per_dim[i];
      if(i < DIM - 1) std::cout << " x ";
    }
    std::cout << " for a total of " << size << " processes" << std::endl
              << std::setw(io_col_width) << "Brick Size :";
    for(int i = 0; i < DIM; ++i) {
      std::cout << std::setw(2) << BRICK_DIM[i];
      if(i < DIM - 1) std::cout << " x ";
    }
    std::cout << "\n"
              << "Iters Between exchanges : " << NUM_GHOST_ZONES << "\n"
              << "Num Exchanges : " << NUM_EXCHANGES << std::endl; 
  }

  // build cartesian communicator and setup MEMFD
  MPI_Comm cartesian_comm = build_cartesian_comm(num_procs_per_dim, per_process_extent);
  MEMFD::setup_prefix("weak/gene-6d-main", rank);
  // get array/brick extents set up for my MPI process (all include ghost-zones)
  std::array<int, DIM> per_process_extent_with_gz,
                       per_process_extent_with_padding;
  for(int i = 0; i < DIM; ++i) {
    per_process_extent_with_gz[i] = per_process_extent[i] + 2 * GHOST_ZONE[i];
    per_process_extent_with_padding[i] = per_process_extent_with_gz[i] + 2 * PADDING[i];
    if(per_process_extent[i] % BRICK_DIM[i] != 0) {
      std::ostringstream error_stream;
      error_stream << "Brick-dimension " << i << " (" << BRICK_DIM[i] << ")"
                   << " does not divide per-process extent " << i << " (" << per_process_extent[i] << ")";
      throw std::runtime_error(error_stream.str());
    }
    if(GHOST_ZONE[i] % BRICK_DIM[i] != 0) {
      std::ostringstream error_stream;
      error_stream << "Brick-dimension " << i << " (" << BRICK_DIM[i] << ")"
                   << " does not divide ghost-zone " << i << " (" << GHOST_ZONE[i] << ")";
      throw std::runtime_error(error_stream.str());
    }
  }
  cudaCheck(cudaMemcpyToSymbol(array_extent_with_gz_dev, per_process_extent_with_gz.data(), DIM * sizeof(unsigned)));
  cudaCheck(cudaMemcpyToSymbol(array_extent_with_padding_dev, per_process_extent_with_padding.data(), DIM * sizeof(unsigned)));
  // set up shape of coeffs
  std::array<int, DIM> coeff_extent_with_gz = per_process_extent_with_gz;
  coeff_extent_with_gz[0] = ARAKAWA_STENCIL_SIZE;
  coeff_extent_with_gz[1] = per_process_extent[0] + GHOST_ZONE[0];
  for(unsigned i = 2; i < DIM; ++i) {
    coeff_extent_with_gz[i] = per_process_extent[i] + 2 * GHOST_ZONE[i];
  }
  // initialize my part of the grid to random data
  bComplexElem *in_ptr = randomComplexArray(std::vector<long>(per_process_extent_with_padding.begin(), per_process_extent_with_padding.end()));
  bComplexElem *array_out_ptr = zeroComplexArray(std::vector<long>(per_process_extent_with_padding.begin(), per_process_extent_with_padding.end()));
  bComplexElem *brick_out_ptr = zeroComplexArray(std::vector<long>(per_process_extent_with_padding.begin(), per_process_extent_with_padding.end()));

  // build brick decomp
  GENEBrickDecomp b_decomp(std::vector<unsigned>(per_process_extent.begin(), per_process_extent.end()),
                           std::vector<unsigned>(GHOST_ZONE.begin(), GHOST_ZONE.end())
                          );
  b_decomp.comm = cartesian_comm;
  std::array<int, DIM> coords_of_proc;
  check_MPI(MPI_Cart_coords(cartesian_comm, rank, DIM, coords_of_proc.data()));
  populate(cartesian_comm, b_decomp, 0, 1, coords_of_proc.data());
  // build 2d skin from 3d skin by removing all faces with a 3,
  // and replacing 1 -> 3, 2 -> 4
  std::vector<BitSet> skin2d = skin3d_good;
  auto set_contains_three = [](BitSet set) -> bool {
    return set.get(3) || set.get(-3);
  };
  skin2d.erase(std::remove_if(skin2d.begin(), skin2d.end(), set_contains_three),
               skin2d.end()) ;
  for(BitSet &bitset : skin2d) {
    if(bitset.get(1)) {
      bitset.flip(1);
      bitset.flip(3);
    }
    if(bitset.get(-1)) {
      bitset.flip(-1);
      bitset.flip(-3);
    }
    if(bitset.get(2)) {
      bitset.flip(2);
      bitset.flip(4);
    }
    if(bitset.get(-2)) {
      bitset.flip(-2);
      bitset.flip(-4);
    }
  }
  b_decomp.initialize(skin2d);
  std::cout << "initialization complete" << std::endl;
  exit(123);

  // initialize my coefficients to random data, and receive coefficients for ghost-zones
  bElem *coeffs = randomArray(std::vector<long>(coeff_extent_with_gz.begin(), coeff_extent_with_gz.end()));
  // build extent/ghost-zones for coeff extent
  std::vector<long> coeff_extent, coeff_ghost_zone;
  coeff_extent.push_back(ARAKAWA_STENCIL_SIZE);
  coeff_ghost_zone.push_back(0);
  coeff_extent.push_back(per_process_extent[0]);
  coeff_ghost_zone.push_back(GHOST_ZONE[0]);
  for(unsigned i = 2; i < DIM; ++i) {
    coeff_extent.push_back(per_process_extent[i]);
    coeff_ghost_zone.push_back(GHOST_ZONE[i]);
  }
  std::cout << "Beginning coefficient exchange" << std::endl;
#if defined(USE_TYPES)
  // set up MPI types for transfer of ghost-zone coeffs
  std::unordered_map<uint64_t, MPI_Datatype> coeffs_stypemap;
  std::unordered_map<uint64_t, MPI_Datatype> coeffs_rtypemap;
  exchangeArrPrepareTypes<DIM, bElem>(coeffs_stypemap,
                                      coeffs_rtypemap,
                                      coeff_extent,
                                      std::vector<long>(coeff_extent.size(), 0), ///< no padding for coeffs
                                      coeff_ghost_zone);
  exchangeArrTypes<DIM>(coeffs, b_decomp.comm, b_decomp.rank_map, coeffs_stypemap, coeffs_rtypemap);
#else
  std::cout << b_decomp.rank_map.size() << std::endl;
  exit(1);
  exchangeArr<DIM>(coeffs, b_decomp.comm, b_decomp.rank_map,
                   coeff_extent,
                   std::vector<long>(coeff_extent.size(), 0), ///< no padding for coeffs
                   coeff_ghost_zone);
#endif

  std::cout << "Beginning array computation" << std::endl;
  // run array computation
  semi_arakawa_distributed_array(array_out_ptr, in_ptr, coeffs, b_decomp, num_procs_per_dim, per_process_extent);
  std::cout << "Beginning brickc computation" << std::endl;
  semi_arakawa_distributed_brick(brick_out_ptr, in_ptr, coeffs, b_decomp, num_procs_per_dim, per_process_extent);

  // check for correctness
  auto array_result = (unsigned (*) [per_process_extent_with_padding[4]]
                                    [per_process_extent_with_padding[3]]
                                    [per_process_extent_with_padding[2]]
                                    [per_process_extent_with_padding[1]]
                                    [per_process_extent_with_padding[0]]) array_out_ptr;
  auto brick_result = (unsigned (*) [per_process_extent_with_padding[4]]
                                    [per_process_extent_with_padding[3]]
                                    [per_process_extent_with_padding[2]]
                                    [per_process_extent_with_padding[1]]
                                    [per_process_extent_with_padding[0]]) brick_out_ptr;
  #pragma omp parallel for collapse(5)
  for(unsigned n = PADDING[5] + GHOST_ZONE[5]; n < PADDING[5] + GHOST_ZONE[5] + per_process_extent[5]; ++n)
  for(unsigned m = PADDING[4] + GHOST_ZONE[4]; m < PADDING[4] + GHOST_ZONE[4] + per_process_extent[4]; ++m)
  for(unsigned l = PADDING[3] + GHOST_ZONE[3]; l < PADDING[3] + GHOST_ZONE[3] + per_process_extent[3]; ++l)
  for(unsigned k = PADDING[2] + GHOST_ZONE[2]; k < PADDING[2] + GHOST_ZONE[2] + per_process_extent[2]; ++k)
  for(unsigned j = PADDING[1] + GHOST_ZONE[1]; j < PADDING[1] + GHOST_ZONE[1] + per_process_extent[1]; ++j)
  #pragma omp simd 
  for(unsigned i = PADDING[0] + GHOST_ZONE[0]; i < PADDING[0] + GHOST_ZONE[0] + per_process_extent[0]; ++i) {
    std::complex<bElem> arr = array_result[n][m][l][k][j][i];
    std::complex<bElem> bri = brick_result[n][m][l][k][j][i];
    if(std::abs(arr - bri) >= 1e-7) {
      std::ostringstream error_stream;
      error_stream << "Mismatch at [" << n << "," << m << "," << l << "," << k << "," << j << "," << i << "]: "
                   << "(array) " << arr << " != " << bri << "(brick)" << std::endl;
      throw std::runtime_error(error_stream.str());
    }
  }

  // free memory
  free(brick_out_ptr);
  free(array_out_ptr);
  free(in_ptr);
  return 0;
}
