#include "gene-6d.h"
#include "nvToolsExt.h"
#include <iomanip>

// constants for number of times to run functions
unsigned NUM_WARMUP_ITERS = CU_WARMUP;
unsigned NUM_ITERS = CU_ITER;

/**
 * @brief wrap timing function with more convenient output
 * 
 * @param f function to time
 * @return std::string a description of the time in GStencil/s
 */
std::string gene_cutime_func(std::function<void()> f)
{
  std::ostringstream string_stream;
  string_stream << NUM_ELEMENTS / cutime_func(f, NUM_WARMUP_ITERS, NUM_ITERS) / 1000000000;
  return string_stream.str();
}

/**
 * @brief check that a and b are close in values
 * 
 * @param a an array of shape EXTENT_WITH_GHOST_ZONE
 * @param b an array of shape EXTENT_WITH_GHOST_ZONE
 */
void checkClose(complexArray6D a, complexArray6D b, std::string name = "")
{
  std::string errorMsg = "";

  // iterate over non-ghost zone elements;
  #pragma omp parallel for collapse(5)
  for (long n = GHOST_ZONE[5]; n < GHOST_ZONE[5] + EXTENT[5]; n++)
  for (long m = GHOST_ZONE[4]; m < GHOST_ZONE[4] + EXTENT[4]; m++)
  for (long l = GHOST_ZONE[3]; l < GHOST_ZONE[3] + EXTENT[3]; l++)
  for (long k = GHOST_ZONE[2]; k < GHOST_ZONE[2] + EXTENT[2]; k++)
  for (long j = GHOST_ZONE[1]; j < GHOST_ZONE[1] + EXTENT[1]; j++)
  for (long i = GHOST_ZONE[0]; i < GHOST_ZONE[0] + EXTENT[0]; i++)
  {
    std::complex<bElem> z = a(i, j, k, l, m, n),
                        w = b(i, j, k, l, m, n);
    bElem diff = std::abs(z - w);
    if(diff > 1e-6) 
    {
      std::ostringstream errorMsgStream;
      errorMsgStream << name << " result mismatch at [n, m, l, k, j, i] = ["
                    << n << ", " << m << ", " << l << ", " << k << ", " << j << ", " << i << "]: "
                    << z.real() << "+" << z.imag() << "I != "
                    << w.real() << "+" << w.imag() << "I";
      std::string localErrorMsg = errorMsgStream.str();
      #pragma omp critical
      errorMsg = localErrorMsg;
    }
  }
  // throw error if there was one
  if(errorMsg.length() > 0)
  {
    throw std::runtime_error(errorMsg);
  }
}

/**
 * @brief Return a shifted view of the array
 * copied from 
 * https://github.com/wdmapp/gtensor/blob/41cf4fe26625f8d7ba2d0d3886a54ae6415a2017/benchmarks/bench_hypz.cxx#L14-L24
 */
template <int N, typename E>
inline auto stencil(E&& e, std::array<int, N> shift)
{
  static_assert(N <= RANK);

  std::vector<gt::gdesc> slices;
  slices.reserve(N);
  for (int d = 0; d < N; d++) {
    using namespace gt::placeholders;
    slices.push_back(_s(GHOST_ZONE[d] + shift[d], -GHOST_ZONE[d] + shift[d]));
  }

  return gt::view<N>(std::forward<E>(e), slices);
}

/**
 * @brief 1-D stencil fused with multiplication by 1-D array
 * 
 * Based on https://github.com/wdmapp/gtensor/blob/d07000b15d253cdeb44942b52f3d2caf4522faa0/benchmarks/ij_deriv.cxx
 */
void ij_deriv(bool run_bricks, bool run_gtensor) {
  // TODO
}

/**
 * @brief Compute the arakawa derivative using gtensor
 * 
 * @param out[out]
 * @param in[in]
 * @param coeff[in]
 */
void semi_arakawa_gtensor(complexArray6D out,
                          complexArray6D in,
                          realArray6D coeff,
                          complexArray6D out_check)
{
  auto shape6D = gt::shape(EXTENT_WITH_GHOST_ZONE[0], EXTENT_WITH_GHOST_ZONE[1],
                           EXTENT_WITH_GHOST_ZONE[2], EXTENT_WITH_GHOST_ZONE[3],
                           EXTENT_WITH_GHOST_ZONE[4], EXTENT_WITH_GHOST_ZONE[5]);
  auto coeffShape = gt::shape(EXTENT[0], ARAKAWA_STENCIL_SIZE, EXTENT[2], EXTENT[3], EXTENT[4], EXTENT[5]);
  // copy in-arrays to gtensor
  auto gt_in = gt::empty<gt::complex<bElem> >(shape6D);
  auto gt_coeff = gt::empty<bElem>(coeffShape);
  #pragma omp parallel for collapse(5)
  for(long n = 0; n < EXTENT_WITH_GHOST_ZONE[5]; ++n)
  for(long m = 0; m < EXTENT_WITH_GHOST_ZONE[4]; ++m)
  for(long l = 0; l < EXTENT_WITH_GHOST_ZONE[3]; ++l)
  for(long k = 0; k < EXTENT_WITH_GHOST_ZONE[2]; ++k)
  for(long j = 0; j < EXTENT_WITH_GHOST_ZONE[1]; ++j)
  #pragma omp simd
  for(long i = 0; i < EXTENT_WITH_GHOST_ZONE[0]; ++i) {
    gt_in(i, j, k, l, m, n) = in(i, j, k, l, m, n);
  }

#pragma omp parallel for collapse(5)
  for(long n = 0; n < EXTENT[5]; ++n)
  for(long m = 0; m < EXTENT[4]; ++m)
  for(long l = 0; l < EXTENT[3]; ++l)
  for(long k = 0; k < EXTENT[2]; ++k)
  for(int coeff_index = 0; coeff_index < ARAKAWA_STENCIL_SIZE; ++coeff_index)
#pragma omp simd
  for(long i = 0; i < EXTENT[0]; ++i) {
    gt_coeff(i, coeff_index, k, l, m, n) =
      coeff(coeff_index, i + GHOST_ZONE[0], k + GHOST_ZONE[2],
            l + GHOST_ZONE[3], m + GHOST_ZONE[4], n + GHOST_ZONE[5]);
  }

  // copy the in-arrays to device
  auto gt_in_dev = gt::empty_device<gt::complex<bElem> >(shape6D);
  auto gt_coeff_dev = gt::empty_device<bElem>(coeffShape);
  gt::copy(gt_in, gt_in_dev);
  gt::copy(gt_coeff, gt_coeff_dev);

  // declare our out-array
  gtensor6D<gt::space::device> gt_out_dev(shape6D);

  // build a function which computes our stencil
  auto compute_semi_arakawa = [&gt_out_dev, &gt_in_dev, &gt_coeff_dev]() -> void {
    using namespace gt::placeholders;
    auto _si = _s(GHOST_ZONE[0], GHOST_ZONE[0] + EXTENT[0]),
         _sj = _s(GHOST_ZONE[1], GHOST_ZONE[1] + EXTENT[1]),
         _sk = _s(GHOST_ZONE[2], GHOST_ZONE[2] + EXTENT[2]),
         _sl = _s(GHOST_ZONE[3], GHOST_ZONE[3] + EXTENT[3]),
         _sm = _s(GHOST_ZONE[4], GHOST_ZONE[4] + EXTENT[4]),
         _sn = _s(GHOST_ZONE[5], GHOST_ZONE[5] + EXTENT[5]);

    auto coeff = [&](int s) { return gt_coeff_dev.view(_all, s, _newaxis, _all, _all, _all, _all); };
    gt_out_dev.view(_si, _sj, _sk, _sl, _sm, _sn) =
      coeff(0 ) * stencil<RANK>(gt_in_dev, {0, 0, +0, -2, 0, 0}) +
      coeff(1 ) * stencil<RANK>(gt_in_dev, {0, 0, -1, -1, 0, 0}) +
      coeff(2 ) * stencil<RANK>(gt_in_dev, {0, 0, +0, -1, 0, 0}) +
      coeff(3 ) * stencil<RANK>(gt_in_dev, {0, 0, +1, -1, 0, 0}) +
      coeff(4 ) * stencil<RANK>(gt_in_dev, {0, 0, -2, +0, 0, 0}) +
      coeff(5 ) * stencil<RANK>(gt_in_dev, {0, 0, -1, +0, 0, 0}) +
      coeff(6 ) * stencil<RANK>(gt_in_dev, {0, 0, +0, +0, 0, 0}) +
      coeff(7 ) * stencil<RANK>(gt_in_dev, {0, 0, +1, +0, 0, 0}) +
      coeff(8 ) * stencil<RANK>(gt_in_dev, {0, 0, +2, +0, 0, 0}) +
      coeff(9 ) * stencil<RANK>(gt_in_dev, {0, 0, -1, +1, 0, 0}) +
      coeff(10) * stencil<RANK>(gt_in_dev, {0, 0, +0, +1, 0, 0}) +
      coeff(11) * stencil<RANK>(gt_in_dev, {0, 0, +1, +1, 0, 0}) +
      coeff(12) * stencil<RANK>(gt_in_dev, {0, 0, +0, +2, 0, 0});

    // actually compute the result
    gt::synchronize();
  };

  // time the function
  std::cout << "gtensor: " << gene_cutime_func(compute_semi_arakawa);

  // copy output data back to host
  auto gt_out = gt::empty<gt::complex<bElem> >(shape6D);
  gt::copy(gt_out_dev, gt_out);

  // copy data from gtensor back to padded array
//  #pragma omp parallel for collapse(5)
  for(long n = GHOST_ZONE[5]; n < GHOST_ZONE[5] + EXTENT[5]; ++n)
  for(long m = GHOST_ZONE[4]; m < GHOST_ZONE[4] + EXTENT[4]; ++m)
  for(long l = GHOST_ZONE[3]; l < GHOST_ZONE[3] + EXTENT[3]; ++l)
  for(long k = GHOST_ZONE[2]; k < GHOST_ZONE[2] + EXTENT[2]; ++k)
  for(long j = GHOST_ZONE[1]; j < GHOST_ZONE[1] + EXTENT[1]; ++j)
//  #pragma omp simd
  for(long i = GHOST_ZONE[0]; i < GHOST_ZONE[0] + EXTENT[0]; ++i)
  {
    out(i, j, k, l, m, n)
        = reinterpret_cast<bComplexElem&>(gt_out(i, j, k, l, m, n));
  }

  // check for correctness
  checkClose(out, out_check, "gtensor");
  std::cout << " PASSED" << std::endl;
  out.set(0.0);
}

/**
 * @brief the arakawa deriv kernel using hand-written bricks code
 */
void semi_arakawa_bricks(complexArray6D out,
                         complexArray6D in,
                         realArray6D coeff,
                         complexArray6D out_check)
{
  // copy coeff from STENCIL_SIZE x GRID to GRID x STENCIL_SIZE
  std::vector<realArray5D> reordered_coeff;
  std::array<unsigned, 5> coeff5DExtent = {EXTENT_WITH_GHOST_ZONE[0],
                                           EXTENT_WITH_GHOST_ZONE[2],
                                           EXTENT_WITH_GHOST_ZONE[3],
                                           EXTENT_WITH_GHOST_ZONE[4],
                                           EXTENT_WITH_GHOST_ZONE[5]};

  for(unsigned s = 0; s < ARAKAWA_STENCIL_SIZE; ++s) {
    realArray5D c(coeff5DExtent);
    #pragma omp parallel for collapse(4)
    for (long n = 0; n < EXTENT_WITH_GHOST_ZONE[5]; n++) {
      for (long m = 0; m < EXTENT_WITH_GHOST_ZONE[4]; m++) {
        for (long l = 0; l < EXTENT_WITH_GHOST_ZONE[3]; l++) {
          for (long k = 0; k < EXTENT_WITH_GHOST_ZONE[2]; k++) {
            for (long i = 0; i < EXTENT_WITH_GHOST_ZONE[0]; i++) {
              c(i, k, l, m, n) = coeff(s, i, k, l, m, n);
            }
          }
        }
      }
    }
    reordered_coeff.push_back(c);
  }

  // set up brick info on host
  brick::BrickLayout<RANK> fieldLayout(BRICK_GRID_EXTENT_WITH_GZ);
  brick::BrickLayout<RANK-1> coeffLayout(
      {BRICK_GRID_EXTENT_WITH_GZ[0], BRICK_GRID_EXTENT_WITH_GZ[2],
       BRICK_GRID_EXTENT_WITH_GZ[3], BRICK_GRID_EXTENT_WITH_GZ[4],
       BRICK_GRID_EXTENT_WITH_GZ[5]});

  // setup bricks on host
  brick::InterleavedBrickedArrays<BrickDims6D ,
                                  brick::DataTypeVectorFoldPair<bComplexElem>
                                  > inAndOut(fieldLayout, 2);
  brick::BrickedArray<bComplexElem, BrickDims6D>
                bIn = std::get<0>(inAndOut.fields).front(),
                bOut = std::get<0>(inAndOut.fields).back();
  bIn.loadFrom(in);

  brick::InterleavedBrickedArrays<BrickDims5D ,
                                  brick::DataTypeVectorFoldPair<bElem>
                                  > interleavedCoeffBricks(coeffLayout, ARAKAWA_STENCIL_SIZE);
  std::vector<brick::BrickedArray<bElem, BrickDims5D> > bCoeffs = std::get<0>(interleavedCoeffBricks.fields);
  for(unsigned s = 0; s < ARAKAWA_STENCIL_SIZE; ++s) {
    bCoeffs[s].loadFrom(reordered_coeff[s]);
  }

  // get bricks on device
  bIn.copyToDevice();  // Side effect: copies bOut to device
  FieldBrick_kl bIn_dev = bIn.viewBricksOnDevice<CommIn_kl>(),
                bOut_dev = bOut.viewBricksOnDevice<CommIn_kl>();

  // copy coefficient bricks to device
  RealCoeffBrick *bCoeffs_dev = nullptr;
  cudaCheck(cudaMalloc(&bCoeffs_dev, ARAKAWA_STENCIL_SIZE * sizeof(RealCoeffBrick)));
  bCoeffs[0].copyToDevice(); // Copies all interleaved bricks (TODO: Better notation?)
  for(unsigned s = 0; s < ARAKAWA_STENCIL_SIZE; ++s)
  {
    RealCoeffBrick brickToCopy = bCoeffs[s].viewBricksOnDevice<NoComm>();
    cudaCheck(cudaMemcpy(bCoeffs_dev + s,
                            &brickToCopy,
                            sizeof(RealCoeffBrick),
                            cudaMemcpyHostToDevice));
  }

  // set up grid pointers on device
  auto fieldGrid_dev = fieldLayout.indexInStorage.allocateOnDevice();
  fieldLayout.indexInStorage.copyToDevice(fieldGrid_dev);
  auto coeffGrid_dev = coeffLayout.indexInStorage.allocateOnDevice();
  coeffLayout.indexInStorage.copyToDevice(coeffGrid_dev);

  // build function to actually run computation
  auto compute_semi_arakawa = [&bIn_dev,
                               &bOut_dev,
                               &fieldGrid_dev,
                               &coeffGrid_dev,
                               &bCoeffs_dev](const char *iteration_order = "ijkmln") -> void {
    unsigned gridDim_x = BRICK_GRID_EXTENT[iteration_order[0] - 'i'];
    unsigned gridDim_y = BRICK_GRID_EXTENT[iteration_order[1] - 'i'];
    unsigned gridDim_z = NUM_BRICKS / gridDim_x / gridDim_y;
    dim3 block(gridDim_x, gridDim_y, gridDim_z),
        thread(BDIM[0], BDIM[1],  NUM_ELEMENTS_PER_BRICK / BDIM[0] / BDIM[1]);
    semi_arakawa_brick_kernel<< < block, thread >> >(
        fieldGrid_dev,
        coeffGrid_dev,
        bIn_dev,
        bOut_dev,
        bCoeffs_dev);
  };
  // figure out some useful cuda data
  int dev;
  cudaGetDevice(&dev);
  cudaDeviceProp devProp;
  cudaGetDeviceProperties(&devProp, dev);

  unsigned colWidth = 18; 
  std::cout << std::setw(colWidth) << "method"
            << std::setw(colWidth) << "blocks"
            << std::setw(colWidth) << "block-size"
            << std::setw(colWidth) << "BrickGridDimOrder"
            << std::setw(colWidth) << "GStencils/s"
            << std::setw(colWidth) << "Check"
            << std::endl;
  std::vector<const char*> iteration_orders = {"ijklmn", "ikjlmn", "kijlmn", "kiljmn", "klijmn"};

  // time function
  for(const char* iteration_order : iteration_orders)
  {
    auto fntn = [&iteration_order, &compute_semi_arakawa]() -> void {compute_semi_arakawa(iteration_order);};

    copy_grid_iteration_order(iteration_order);
    std::cout << std::setw(colWidth) << "bricks"
              << std::setw(colWidth) << NUM_BRICKS
              << std::setw(colWidth) << NUM_ELEMENTS_PER_BRICK
              << std::setw(colWidth) << iteration_order
              << std::setw(colWidth) << gene_cutime_func(fntn)
              << std::flush;
    bOut.copyFromDevice();
    bOut.storeTo(out);
    // check for correctness
    checkClose(out, out_check, "arakawa bricks");
    std::cout << std::setw(colWidth) << " PASSED" << std::endl;
    // reset array
    out.set(0.0);
    bOut.loadFrom(out);
    bOut.copyToDevice();
  }

  cudaCheck(cudaFree(bCoeffs_dev));
}

/**
 * @brief 2-D stencil with variable coefficients
 * 
 * Based on https://github.com/wdmapp/gtensor/blob/41cf4fe26625f8d7ba2d0d3886a54ae6415a2017/benchmarks/bench_hypz.cxx#L30-L49
 */
void semi_arakawa(bool run_bricks, bool run_gtensor) {
  std::cout << "Setting up semi-arakawa arrays" << std::endl;
  // build in/out arrays
  complexArray6D in{complexArray6D::random(EXTENT_WITH_GHOST_ZONE)},
                 out(EXTENT_WITH_GHOST_ZONE, 0.0);
  // build coefficients needed for stencil computation
  realArray6D coeff{realArray6D::random(
      {ARAKAWA_STENCIL_SIZE, EXTENT_WITH_GHOST_ZONE[0], EXTENT_WITH_GHOST_ZONE[2],
       EXTENT_WITH_GHOST_ZONE[3],     EXTENT_WITH_GHOST_ZONE[4], EXTENT_WITH_GHOST_ZONE[5]})
  };

  // compute stencil on CPU for correctness check
  std::cout << "Computing correctness check" << std::endl;
  brick::Array<bComplexElem, RANK, Padding6D> out_check(EXTENT_WITH_GHOST_ZONE);

  double index = 0;
  for(auto & val : coeff) {
    val = index;
    index = index + 1;
  }

  static_assert(EXTENT[2] % TILE == 0);
  static_assert(EXTENT[3] % TILE == 0);
  #pragma omp parallel for collapse(4)
  for(long n = GHOST_ZONE[5]; n < GHOST_ZONE[5] + EXTENT[5]; ++n)
  for(long m = GHOST_ZONE[4]; m < GHOST_ZONE[4] + EXTENT[4]; ++m)
  for(long tl = GHOST_ZONE[3]; tl < GHOST_ZONE[3] + EXTENT[3]; tl += TILE)
  for(long tk = GHOST_ZONE[2]; tk < GHOST_ZONE[2] + EXTENT[2]; tk += TILE)
  for(long l = tl; l < tl + TILE; l++)
  for(long k = tk; k < tk + TILE; k++)
  for(long j = GHOST_ZONE[1]; j < GHOST_ZONE[1] + EXTENT[1]; ++j)
#pragma omp simd
  for(long i = GHOST_ZONE[0]; i < GHOST_ZONE[0] + EXTENT[0]; ++i)
  {
    out_check(i, j, k, l, m, n) =
        coeff(0 , i, k, l, m, n) * in(i, j, k+0, l-2, m, n) +
        coeff(1 , i, k, l, m, n) * in(i, j, k-1, l-1, m, n) +
        coeff(2 , i, k, l, m, n) * in(i, j, k+0, l-1, m, n) +
        coeff(3 , i, k, l, m, n) * in(i, j, k+1, l-1, m, n) +
        coeff(4 , i, k, l, m, n) * in(i, j, k-2, l+0, m, n) +
        coeff(5 , i, k, l, m, n) * in(i, j, k-1, l+0, m, n) +
        coeff(6 , i, k, l, m, n) * in(i, j, k+0, l+0, m, n) +
        coeff(7 , i, k, l, m, n) * in(i, j, k+1, l+0, m, n) +
        coeff(8 , i, k, l, m, n) * in(i, j, k+2, l+0, m, n) +
        coeff(9 , i, k, l, m, n) * in(i, j, k-1, l+1, m, n) +
        coeff(10, i, k, l, m, n) * in(i, j, k+0, l+1, m, n) +
        coeff(11, i, k, l, m, n) * in(i, j, k+1, l+1, m, n) +
        coeff(12, i, k, l, m, n) * in(i, j, k+0, l+2, m, n);
  }

  // run computations
  std::cout << "Starting semi_arakawa benchmarks" << std::endl;
  if(run_gtensor)
  {
    semi_arakawa_gtensor(out, in, coeff, out_check);
  }
  if(run_bricks)
  {
    semi_arakawa_bricks(out, in, coeff, out_check);
  }

  std::cout << "done" << std::endl;
}