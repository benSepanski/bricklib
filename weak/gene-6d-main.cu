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

#include "Array.h"
#include "BrickedArray.h"
#include "MPILayout.h"
#include "bitset.h"
#include "stencils/cudaarray.h"
#include <brickcompare.h>
#include <multiarray.h>

#include <unistd.h>
#include <array-mpi.h>

// useful constants
constexpr unsigned RANK = 6;
constexpr std::array<unsigned, RANK> BRICK_DIM = {2, 16, 2, 2, 1, 1};
constexpr std::array<unsigned, RANK> COEFF_BRICK_DIM = {1, BRICK_DIM[0], BRICK_DIM[2], BRICK_DIM[3], BRICK_DIM[4], BRICK_DIM[5]};
constexpr unsigned NUM_GHOST_ZONES = 1;
constexpr std::array<unsigned, RANK> GHOST_ZONE = {0, 0, 2 * NUM_GHOST_ZONES, 2 * NUM_GHOST_ZONES, 0, 0};
constexpr std::array<unsigned, RANK> PADDING = {0, 0, 2, 2, 0, 0};
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
typedef brick::MPILayout<FieldBrickDimsType, CommIn_kl> GeneMPILayout;

typedef brick::Padding<PADDING[5], PADDING[4], PADDING[3], PADDING[2], PADDING[1], PADDING[0]> Padding6D;
typedef brick::Array<bComplexElem, 6, Padding6D> complexArray6D;
typedef brick::Array<bElem, 6> realArray6D;
typedef brick::BrickedArray<bComplexElem, FieldBrickDimsType , VectorFoldType> BrickedFieldArray;
typedef brick::BrickedArray<bElem, CoeffBrickDimsType, VectorFoldType> BrickedCoeffArray;

// global constants set by CLI
unsigned NUM_EXCHANGES; ///< how many mpi exchanges?

/**
 * @brief build a cartesian communicator
 * 
 * Assumes MPI_Init_thread has already been called.
 * 
 * Prints some useful information about the MPI setup.
 * 
 * @param[in] numProcsPerDim the number of MPI processes to put in each dimension.
 *                                  Product must match the number of MPI processes.
 * @param[in] perProcessExtent extent in each dimension for each individual MPI processes.
 * @return MPI_comm a cartesian communicator built from MPI_COMM_WORLD
 */
MPI_Comm buildCartesianComm(std::array<int, RANK> numProcsPerDim,
                            std::array<int, RANK> perProcessExtent) {
  // get number of MPI processes and my rank
  int size, rank;
  check_MPI(MPI_Comm_size(MPI_COMM_WORLD, &size));
  check_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

  // make sure num_procs_per_dim has product to number of processes
  int prodOfProcsPerDim = std::accumulate(numProcsPerDim.begin(), numProcsPerDim.end(), 1, std::multiplies<size_t>());
  if(prodOfProcsPerDim != size) {
    std::ostringstream error_stream;
    error_stream << "Product of number of processes per dimension is " << prodOfProcsPerDim
                 << " which does not match number of MPI processes (" << size << ")\n";
    throw std::runtime_error(error_stream.str());
  }

  // set up processes on a cartesian communication grid
  std::array<int, RANK> periodic{};
  for(int i = 0; i < RANK; ++i) {
    periodic[i] = true;
  }
  bool allowRankReordering = true;
  MPI_Comm cartesianComm;
  check_MPI(MPI_Cart_create(MPI_COMM_WORLD,
                            RANK,
                            numProcsPerDim.data(),
                            periodic.data(),
                            allowRankReordering,
                            &cartesianComm));
  if(cartesianComm == MPI_COMM_NULL) {
    throw std::runtime_error("Failure in cartesian comm setup");
  }

  // return the communicator
  return cartesianComm;
}

/**
 * @brief times func and prints stats
 * 
 * @param func the func to run
 * @param mpiLayout the MPI layout used
 * @param totElems the number of elements
 */
void timeAndPrintMPIStats(std::function<void(void)> func, GeneMPILayout &mpiLayout, double totElems) {
  // time function
  int warmup = 5;  //<  TODO: read from cmdline
  int cnt = NUM_EXCHANGES * NUM_GHOST_ZONES;
  for(int i = 0; i < warmup; ++i) func();
  packtime = calltime = waittime = movetime = calctime = 0;
  double start = omp_get_wtime(), end;
  for (int i = 0; i < NUM_EXCHANGES; ++i) func();
  end = omp_get_wtime();

  size_t tsize = 0;
  for (auto g: mpiLayout.getBrickDecompPtr()->ghost)
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

    double perf = (double) totElems * 1.0e-9;
    perf = perf / total;
    std::cout << "perf " << perf << " GStencil/s" << std::endl;
    std::cout << std::endl;
  }
}

__device__ __constant__ unsigned arrayExtentWithGZDev[RANK];

/**
 * @brief cuda kernel to compute k-l arakawa derivative (array layout)
 * 
 * Should be invoked 1 thread per array element (including ghosts, but not padding)
 * global thread idx should by x.y.z = K.L.IJMN
 * 
 * @param out_ptr output array with ghost-zones and padding
 * @param in_ptr input array with ghost-zones and padding
 * @param coeff stencil coefficients with ghost-zones
 */
__global__
void semiArakawaArrKernel(brick::Array<bComplexElem, 6> out,
                          brick::Array<bComplexElem, 6> in,
                          realArray6D coeff)
{
  // convenient aliases
  unsigned * extentWithGZ = arrayExtentWithGZDev;

  size_t globalZIdx = threadIdx.z + blockIdx.z * blockDim.z;
  unsigned idx[6];
  idx[0] = globalZIdx % extentWithGZ[0];
  idx[1] = (globalZIdx / extentWithGZ[0]) % extentWithGZ[1];
  idx[4] = (globalZIdx / extentWithGZ[0] / extentWithGZ[1]) % extentWithGZ[4];
  idx[5] = (globalZIdx / extentWithGZ[0] / extentWithGZ[1] / extentWithGZ[4]);
  idx[2] = threadIdx.x + blockIdx.x * blockDim.x;
  idx[3] = threadIdx.y + blockIdx.y * blockDim.y;

  // guard OOB access
  for(unsigned d = 0; d < 6; ++d) {
    if(idx[d] >= extentWithGZ[d]) {
      return;
    }
  }

  auto c = [&coeff, &idx](unsigned s) -> bElem {
    return coeff.get(s, idx[0], idx[2], idx[3], idx[4], idx[5]);
  };
  auto input = [&idx, &in](int shiftK, int shiftL) -> bComplexElem {
    return in.get(idx[0] + PADDING[0], idx[1] + PADDING[1],
                  idx[2] + PADDING[2] + shiftK, idx[3] + PADDING[3] + shiftL,
                  idx[4] + PADDING[4], idx[5] + PADDING[5]);
  };

  out(idx[0] + PADDING[0], idx[1] + PADDING[1], idx[2] + PADDING[2],
      idx[3] + PADDING[3], idx[4] + PADDING[4], idx[5] + PADDING[5])
      = input(0, -1);
//      = c( 0) * input( 0, -2)
//      + c( 1) * input(-1, -1)
//      + c( 2) * input( 0, -1)
//      + c( 3) * input( 1, -1)
//      + c( 4) * input(-2,  0)
//      + c( 5) * input(-1,  0)
//      + c( 6) * input( 0,  0)
//      + c( 7) * input( 1,  0)
//      + c( 8) * input( 2,  0)
//      + c( 9) * input(-1,  1)
//      + c(10) * input( 0,  1)
//      + c(11) * input( 1,  1)
//      + c(12) * input( 0,  2);
}

/**
 * @brief perform semi-arakawa k-l derivative kernel weak-scaling benchmark
 * 
 * Uses array layout
 * 
 * @param[out] out output data (has ghost-zones)
 * @param[in] in input data (has ghost-zones)
 * @param[in] coeffs input coefficients (has ghost-zones)
 * @param[in] mpiLayout the mpi layout
 */
void semiArakawaDistributedArray(complexArray6D out,
                                 complexArray6D in,
                                 realArray6D coeffs,
                                 GeneMPILayout &mpiLayout) {
  // set up MPI types for transfer
  std::unordered_map<uint64_t, MPI_Datatype> stypemap;
  std::unordered_map<uint64_t, MPI_Datatype> rtypemap;
  // get arrays as vectors
  std::vector<long> extentAsVector({in.extent[0], in.extent[1], in.extent[2],
                                    in.extent[3], in.extent[4], in.extent[5]}),
                    paddingAsVector(PADDING.begin(), PADDING.end()),
                    ghostZoneAsVector(GHOST_ZONE.begin(), GHOST_ZONE.end());
  exchangeArrPrepareTypes<RANK, bComplexElem>(stypemap,
                                              rtypemap,
                                              extentAsVector,
                                              paddingAsVector,
                                              ghostZoneAsVector);
  // set up in/out ptrs on device without padding
  std::array<unsigned, RANK> unpaddedExtentWithGZ{};
  for(unsigned d = 0; d < RANK; ++d) {
    unpaddedExtentWithGZ[d] = in.extent[d] + 2 * PADDING[d];
  }
  brick::Array<bComplexElem, 6> unpaddedIn(unpaddedExtentWithGZ, in.getData()),
      unpaddedOut(unpaddedExtentWithGZ, out.getData()),
      unpaddedIn_dev = unpaddedIn.allocateOnDevice(),
      unpaddedOut_dev = unpaddedOut.allocateOnDevice();
  realArray6D coeff_dev = coeffs.allocateOnDevice();
  unpaddedIn.copyToDevice(unpaddedIn_dev);
  coeffs.copyToDevice(coeff_dev);

  // build function to perform computation
  auto inPtr_dev = &unpaddedIn_dev;
  auto outPtr_dev = &unpaddedOut_dev;
  auto arrFunc = [&]() -> void {
    float elapsed;
    cudaEvent_t c_0, c_1;
    cudaCheck(cudaEventCreate(&c_0));
    cudaCheck(cudaEventCreate(&c_1));
#if !defined(CUDA_AWARE) || !defined(USE_TYPES)
    // Copy everything back from device
    double st = omp_get_wtime();
    unpaddedIn.copyFromDevice(*inPtr_dev);
    movetime += omp_get_wtime() - st;
    mpiLayout.exchangeArray(in); // exchange should ignore padding
    st = omp_get_wtime();
    unpaddedIn.copyToDevice(*inPtr_dev);
    movetime += omp_get_wtime() - st;
#else
    static_assert(false, "Not implemented yet");
    exchangeArrTypes<RANK>(in_dev.getData(),
                           bDecomp.comm,
                           bDecomp.rank_map,
                           stypemap,
                           rtypemap);
#endif
    cudaCheck(cudaEventRecord(c_0));
    dim3 block(TILE_SIZE, TILE_SIZE),
          grid((in.extent[2] + block.x - 1) / block.x,
               (in.extent[3] + block.y - 1) / block.y,
               (in.extent[0] * in.extent[1] * in.extent[4] * in.extent[5]) / block.z);
    for (int i = 0; i < NUM_GHOST_ZONES; ++i) {
      semiArakawaArrKernel<< < grid, block>> > (*outPtr_dev, *inPtr_dev, coeff_dev);
      if(i + 1 < NUM_GHOST_ZONES) {
        std::cout << "Swapping!" << std::endl;
        std::swap(outPtr_dev, inPtr_dev);
      }
    }
    cudaCheck(cudaEventRecord(c_1));
    cudaCheck(cudaEventSynchronize(c_1));
    cudaCheck(cudaEventElapsedTime(&elapsed, c_0, c_1));
    calctime += elapsed / 1000.0;
  };

  int rank;
  check_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  if (rank == 0)
    std::cout << "array MPI decomp" << std::endl;
  timeAndPrintMPIStats(arrFunc, mpiLayout, (double) in.numElements);

  // Copy back
  unpaddedOut.copyFromDevice(unpaddedOut_dev);
}

__device__ __constant__ unsigned brick_grid_extent_with_gz_dev[RANK];
/**
 * @brief cuda kernel to compute k-l arakawa derivative (brick layout)
 * 
 * Should be invoked thread-block size =  num brick elements
 * and 1 thread-block per brick.
 * global brick idx should by x.y.z = K.L.IJMN.
 * intra-brick idx should be x.y.z = I.J.KLMN
 * 
 * @param grid brick-grid for field bricks (includes ghost bricks)
 * @param coeffGrid brick-grid for coefficients (includes ghost bricks)
 * @param bOut output brick
 * @param bIn input brick
 * @param bCoeff coefficients
 */
__global__
void semiArakawaBrickKernel(brick::Array<unsigned, RANK, brick::Padding<>, unsigned> grid,
                            brick::Array<unsigned, RANK, brick::Padding<>, unsigned> coeffGrid,
                            FieldBrick_kl bOut,
                            FieldBrick_kl bIn,
                            RealCoeffBrick bCoeff)
{
  // get brick index for field
  unsigned b_k = blockIdx.x;
  unsigned b_l = blockIdx.y;
  unsigned b_ijmn = blockIdx.z;
  unsigned b_i = b_ijmn % grid.extent[0];
  unsigned b_jmn = b_ijmn / grid.extent[0];
  unsigned b_j = b_jmn % grid.extent[1];
  unsigned b_mn = b_jmn / grid.extent[1];
  unsigned b_m = b_mn % grid.extent[4];
  unsigned b_n = b_mn / grid.extent[4];

  // get field and coeff brick indexes
  unsigned fieldBrickIdx = grid(b_i, b_j, b_k, b_l, b_m, b_n);

  // intra-brick indexing
  int i = threadIdx.x;
  int j = threadIdx.y;
  int klmn = threadIdx.z;
  int k = klmn % BRICK_DIM[2];
  int lmn = klmn / BRICK_DIM[2];
  int l = lmn % BRICK_DIM[3];
  int mn = lmn / BRICK_DIM[3];
  int m = mn % BRICK_DIM[4];
  int n = mn / BRICK_DIM[4];

  // check for intra-brick OOB
  assert(i < BRICK_DIM[0] && i < COEFF_BRICK_DIM[1]);
  assert(j < BRICK_DIM[1]);
  assert(k < BRICK_DIM[2] && k < COEFF_BRICK_DIM[2]);
  assert(l < BRICK_DIM[3] && l < COEFF_BRICK_DIM[3]);
  assert(m < BRICK_DIM[4] && m < COEFF_BRICK_DIM[4]);
  assert(n < BRICK_DIM[5] && n < COEFF_BRICK_DIM[5]);

  // compute stencil
  bComplexElem result = 0.0;
  auto in = bIn[fieldBrickIdx];
  auto input = [&](int deltaK, int deltaL) -> bComplexElem {
    return in[n][m][l + deltaL][k + deltaK][j][i];
  };
  auto c = [&](unsigned stencilIdx) -> bElem {
    unsigned coeffBrickIndex = coeffGrid(stencilIdx, b_i, b_k, b_l, b_m, b_n);
    return bCoeff[coeffBrickIndex][n][m][l][k][i][0];
  };
  bOut[fieldBrickIdx][n][m][l][k][j][i] = input(0, -1);
//      = c( 0) * input( 0, -2)
//      + c( 1) * input(-1, -1)
//      + c( 2) * input( 0, -1)
//      + c( 3) * input( 1, -1)
//      + c( 4) * input(-2,  0)
//      + c( 5) * input(-1,  0)
//      + c( 6) * input( 0,  0)
//      + c( 7) * input( 1,  0)
//      + c( 8) * input( 2,  0)
//      + c( 9) * input(-1,  1)
//      + c(10) * input( 0,  1)
//      + c(11) * input( 1,  1)
//      + c(12) * input( 0,  2);
}

/**
 * @brief perform semi-arakawa k-l derivative kernel weak-scaling benchmark
 * 
 * Uses bricks layout
 * 
 * @param[out] out_ptr output data (has ghost-zones and padding)
 * @param[in] in_ptr input data (has ghost-zones and padding)
 * @param[in] coeffs input coefficients (has ghost-zones but no padding)
 * @param[in] mpiLayout the mpi-layout
 */
void semiArakawaDistributedBrick(complexArray6D out,
                                 complexArray6D in,
                                 realArray6D coeffs,
                                 GeneMPILayout &mpiLayout) {
  // set up brick-info and storage on host
  brick::BrickLayout<RANK> fieldLayout = mpiLayout.getBrickLayout();
#ifdef DECOMP_PAGEUNALIGN
  BrickedFieldArray bIn(in.extent), bOut(out.extent);
#else
  // load with mmap
  BrickedFieldArray bInArray(fieldLayout, nullptr);
  BrickedFieldArray bOutArray(fieldLayout, nullptr);
#endif
  // load in input
  bInArray.loadFrom(in);
  std::array<unsigned, RANK> coeffBrickGridExtent{};
  for(unsigned d = 0; d < RANK; ++d) {
    assert(coeffs.extent[d] % COEFF_BRICK_DIM[d] == 0);
    coeffBrickGridExtent[d] = coeffs.extent[d] / COEFF_BRICK_DIM[d];
  }
  brick::BrickLayout<RANK> coeffLayout(coeffBrickGridExtent);
  BrickedCoeffArray bCoeffArray(coeffLayout);
  bCoeffArray.loadFrom(coeffs);

  // set up on device
  bInArray.copyToDevice();
  bCoeffArray.copyToDevice();
  FieldBrick_kl bIn_dev = bInArray.viewBricksOnDevice<CommIn_kl>();
  FieldBrick_kl bOut_dev = bOutArray.viewBricksOnDevice<CommIn_kl>();
  RealCoeffBrick bCoeff_dev = bCoeffArray.viewBricksOnDevice<NoComm>();
  auto fieldIndexInStorage_dev = fieldLayout.indexInStorage.allocateOnDevice();
  fieldLayout.indexInStorage.copyToDevice(fieldIndexInStorage_dev);
  auto coeffIndexInStorage_dev = coeffLayout.indexInStorage.allocateOnDevice();
  coeffLayout.indexInStorage.copyToDevice(coeffIndexInStorage_dev);

#ifndef DECOMP_PAGEUNALIGN
  ExchangeView ev = mpiLayout.buildExchangeView(bInArray);
#endif
  // setup brick function to compute stencil
  auto brickFunc = [&]() -> void {
    float elapsed;
    cudaEvent_t c_0, c_1;
    cudaEventCreate(&c_0);
    cudaEventCreate(&c_1);
#ifndef CUDA_AWARE
  {
    double t_a = omp_get_wtime();
//    mpiLayout.copyBoundaryFromCuda(bInArray);
    bInArray.copyFromDevice(); // TODO: JUST COPY BOUNDARY
    double t_b = omp_get_wtime();
    movetime += t_b - t_a;
  #ifdef DECOMP_PAGEUNALIGN
    b_decomp.exchange(b_storage);
  #else
    ev.exchange();
  #endif
    t_a = omp_get_wtime();
//    mpiLayout.copyGhostToCuda(bInArray);
    bInArray.copyToDevice(); // TODO: JUST COPY GHOSTS
    t_b = omp_get_wtime();
    movetime += t_b - t_a;
  }
#else
  bDecomp.exchange(bStorage_dev);
#endif
    const unsigned * const brickExtentWithGZ = fieldLayout.indexInStorage.extent;
    dim3 cuda_grid_size(brickExtentWithGZ[2],
                        brickExtentWithGZ[3],
                        brickExtentWithGZ[0] * brickExtentWithGZ[1] *
                        brickExtentWithGZ[4] * brickExtentWithGZ[5]),
         cuda_block_size(BRICK_DIM[0], BRICK_DIM[1], FieldBrick_kl::BRICKLEN / BRICK_DIM[0] / BRICK_DIM[1]);
    cudaCheck(cudaEventRecord(c_0));
    for (int i = 0; i < NUM_GHOST_ZONES; ++i) {
      semiArakawaBrickKernel<< < cuda_grid_size, cuda_block_size>> >(
          fieldIndexInStorage_dev, coeffIndexInStorage_dev, bOut_dev,
          bIn_dev, bCoeff_dev
          );
      if(i + 1 < NUM_GHOST_ZONES) {
        std::cout << "Swapping!" << std::endl;
        std::swap(bOut_dev, bIn_dev);
      }
    }
    cudaCheck(cudaEventRecord(c_1));
    cudaCheck(cudaEventSynchronize(c_1));
    cudaCheck(cudaEventElapsedTime(&elapsed, c_0, c_1));
    calctime += elapsed / 1000.0;
  };

  // time function
  int rank;
  check_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  if (rank == 0)
    std::cout << "brick MPI decomp" << std::endl;
  timeAndPrintMPIStats(brickFunc, mpiLayout, (double) in.numElements);

  // Copy back
  bOutArray.copyFromDevice();
  bOutArray.storeTo(out);
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
unsigned parse_args(std::array<int, RANK> *per_process_domain_size,
                    std::array<int, RANK> *num_procs_per_dim,
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
        } else if(tuple.size() != RANK) {
          error_stream << "Expected extent of length " << RANK << ", not " << tuple.size();
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
        else if(tuple.size() != RANK) {
          error_stream << "Expected num procs per dim of length " << RANK << ", not " << tuple.size();
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
  std::array<int, RANK> numProcsPerDim{}, globalExtent{}, perProcessExtent{};
  std::stringstream input_stream;
  for(int i = 1; i < argc; ++i) {
    input_stream << argv[i] << " ";
  }
  NUM_EXCHANGES = parse_args(&perProcessExtent, &numProcsPerDim, input_stream);
  for(int i = 0; i < RANK; ++i) {
    globalExtent[i] = perProcessExtent[i] * numProcsPerDim[i];
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
    size_t totElems = std::accumulate(globalExtent.begin(), globalExtent.end(), 1, std::multiplies<size_t>());
    int io_col_width = 30;
    std::cout << std::setw(io_col_width) << "Pagesize :" << page_size << "\n"
              << std::setw(io_col_width) << "MPI Processes :" << size << "\n"
              << std::setw(io_col_width) << "OpenMP threads :" << numthreads << "\n" 
              << std::setw(io_col_width) << "Domain size (per-process) of :";
    for(int i = 0; i < RANK; ++i) {
      std::cout << std::setw(2) << perProcessExtent[i];
      if(i < RANK - 1) std::cout << " x ";
    }
    std::cout << " for a total of " << totElems << " elements " << std::endl
              << std::setw(io_col_width) << "Ghost Zone :";
    for(int i = 0; i < RANK; ++i) {
      std::cout << std::setw(2) << GHOST_ZONE[i];
      if(i < RANK - 1) std::cout << " x ";
    }
    std::cout << std::endl
              << std::setw(io_col_width) << "Array Padding :";
    for(int i = 0; i < RANK; ++i) {
      std::cout << std::setw(2) << PADDING[i];
      if(i < RANK - 1) std::cout << " x ";
    }
    std::cout << std::endl
              << std::setw(io_col_width) << "MPI Processes :";
    for(int i = 0; i < RANK; ++i) {
      std::cout << std::setw(2) << numProcsPerDim[i];
      if(i < RANK - 1) std::cout << " x ";
    }
    std::cout << " for a total of " << size << " processes" << std::endl
              << std::setw(io_col_width) << "Brick Size :";
    for(int i = 0; i < RANK; ++i) {
      std::cout << std::setw(2) << BRICK_DIM[i];
      if(i < RANK - 1) std::cout << " x ";
    }
    std::cout << "\n"
              << "Iters Between exchanges : " << NUM_GHOST_ZONES << "\n"
              << "Num Exchanges : " << NUM_EXCHANGES << std::endl; 
  }

  // build cartesian communicator and setup MEMFD
  MPI_Comm cartesianComm = buildCartesianComm(numProcsPerDim, perProcessExtent);
  MEMFD::setup_prefix("weak-gene-6d-main", rank);
  // get array/brick extents set up for my MPI process (all include ghost-zones)
  std::array<int, RANK> perProcessExtentWithGZ{},
                       per_process_extent_with_padding{};
  for(int i = 0; i < RANK; ++i) {
    perProcessExtentWithGZ[i] = perProcessExtent[i] + 2 * (int) GHOST_ZONE[i];
    per_process_extent_with_padding[i] = perProcessExtentWithGZ[i] + 2 * (int) PADDING[i];
    if(perProcessExtent[i] % BRICK_DIM[i] != 0) {
      std::ostringstream error_stream;
      error_stream << "Brick-dimension " << i << " (" << BRICK_DIM[i] << ")"
                   << " does not divide per-process extent " << i << " (" << perProcessExtent[i] << ")";
      throw std::runtime_error(error_stream.str());
    }
    if(GHOST_ZONE[i] % BRICK_DIM[i] != 0) {
      std::ostringstream error_stream;
      error_stream << "Brick-dimension " << i << " (" << BRICK_DIM[i] << ")"
                   << " does not divide ghost-zone " << i << " (" << GHOST_ZONE[i] << ")";
      throw std::runtime_error(error_stream.str());
    }
  }
  std::cout << "Building input arrays..." << std::endl;

  cudaCheck(cudaMemcpyToSymbol(arrayExtentWithGZDev, perProcessExtentWithGZ.data(), RANK * sizeof(unsigned)));
  // set up coeffs
  std::array<unsigned, RANK> coeffExtent{}, coeffGhostDepth{};
  coeffExtent[0] = ARAKAWA_STENCIL_SIZE;
  coeffGhostDepth[0] = 0;
  coeffExtent[1] = perProcessExtent[0];
  coeffGhostDepth[0] = GHOST_ZONE[0];
  for(unsigned i = 2; i < RANK; ++i) {
    coeffExtent[i] = perProcessExtent[i];
    coeffGhostDepth[i] = GHOST_ZONE[i];
  }
  std::array<unsigned, RANK> coeffExtentWithGZ{};
  for(unsigned i = 0; i < RANK; ++i) {
    coeffExtentWithGZ[i] = coeffExtent[i] + 2 * coeffGhostDepth[i];
  }

  complexArray6D in{complexArray6D::random(perProcessExtentWithGZ)},
                 array_out{perProcessExtentWithGZ, 0.0},
                 brick_out{perProcessExtentWithGZ, 0.0};

  std::cout << "Input arrays built. Setting up brick decomposition..." << std::endl;
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
  // build brick decomp
  brick::MPILayout<FieldBrickDimsType, CommIn_kl> mpiLayout(
      cartesianComm, perProcessExtent, GHOST_ZONE, skin2d
  );

  std::cout << "Brick decomposition setup complete. Beginning coefficient setup..." << std::endl;
  // initialize my coefficients to random data, and receive coefficients for ghost-zones
  realArray6D coeffs{realArray6D::random(coeffExtentWithGZ)};
  std::cout << "Beginning coefficient exchange" << std::endl;
  brick::MPILayout<CoeffBrickDimsType, CommIn_kl> coeffMpiLayout(
      cartesianComm, coeffExtent, coeffGhostDepth, skin2d
      );
#if defined(USE_TYPES)
  static_assert(false, "exchangeArrTypes with BrickLayout implemented yet");
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
  coeffMpiLayout.exchangeArray(coeffs);
#endif

  std::cout << "Coefficient exchange complete. Beginning array computation" << std::endl;
  // run array computation
  semiArakawaDistributedArray(array_out, in, coeffs, mpiLayout);
  std::cout << "Array computation complete. Beginning bricks computation" << std::endl;
  semiArakawaDistributedBrick(brick_out, in, coeffs, mpiLayout);

  // check for correctness
  // #pragma omp parallel for collapse(5)
  for(unsigned n = GHOST_ZONE[5]; n < GHOST_ZONE[5] + perProcessExtent[5]; ++n) {
    for (unsigned m = GHOST_ZONE[4]; m < GHOST_ZONE[4] + perProcessExtent[4];
         ++m) {
      for (unsigned l = GHOST_ZONE[3]; l < GHOST_ZONE[3] + perProcessExtent[3];
           ++l) {
        for (unsigned k = GHOST_ZONE[2];
             k < GHOST_ZONE[2] + perProcessExtent[2]; ++k) {
          for (unsigned j = GHOST_ZONE[1];
               j < GHOST_ZONE[1] + perProcessExtent[1]; ++j) {
            // #pragma omp simd
            for (unsigned i = GHOST_ZONE[0];
                 i < GHOST_ZONE[0] + perProcessExtent[0]; ++i) {
              std::complex<bElem> arr = array_out(i, j, k, l, m, n);
              std::complex<bElem> bri = brick_out(i, j, k, l, m, n);
              if (std::abs(arr - bri) >= 1e-7) {
                std::ostringstream error_stream;
                error_stream << "Mismatch at [" << n << "," << m << "," << l
                             << "," << k << "," << j << "," << i << "]: "
                             << "(array) " << arr << " != " << bri << "(brick)"
                             << std::endl;
                throw std::runtime_error(error_stream.str());
              }
            }
          }
        }
      }
    }
  }

  check_MPI(MPI_Finalize());
  return 0;
}
