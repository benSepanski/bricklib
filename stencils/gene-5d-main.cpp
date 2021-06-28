#define GTENSOR_DEFAULT_DEVICE_ALLOCATOR(T) gt::device_allocator<T>
#include "gene-5d.h"
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
  string_stream << NUM_ELEMENTS / cutime_func(f, NUM_WARMUP_ITERS, NUM_ITERS) / 1000000000 << " avg GStencils/s";
  return string_stream.str();
}

// convenient typedefs for arrays of pointers
typedef bComplexElem (*complexArray6D)[PADDED_EXTENT_m][PADDED_EXTENT_l][PADDED_EXTENT_k][PADDED_EXTENT_j][PADDED_EXTENT_i];
typedef bComplexElem (*coeffArray5D)[PADDED_EXTENT_m][PADDED_EXTENT_l][PADDED_EXTENT_k][PADDED_EXTENT_i];

/**
 * @brief check that a and b are close in values
 * 
 * @param a an array of shape PADDED_EXTENT
 * @param b an array of shape PADDED_EXTENT
 */
void check_close(complexArray6D a, complexArray6D b, std::string name = "")
{
  std::exception *e = nullptr;

  // iterate over non-ghost zone elements
  #pragma omp parallel for collapse(5) shared(e)
  for (long n = PADDING_n + GHOST_ZONE_n; n < PADDING_n + GHOST_ZONE_n + EXTENT_n; n++)
  for (long m = PADDING_m + GHOST_ZONE_m; m < PADDING_m + GHOST_ZONE_m + EXTENT_m; m++)
  for (long l = PADDING_l + GHOST_ZONE_l; l < PADDING_l + GHOST_ZONE_l + EXTENT_l; l++)
  for (long k = PADDING_k + GHOST_ZONE_k; k < PADDING_k + GHOST_ZONE_k + EXTENT_k; k++)
  for (long j = PADDING_j + GHOST_ZONE_j; j < PADDING_j + GHOST_ZONE_j + EXTENT_j; j++)
  #pragma omp simd
  for (long i = PADDING_i + GHOST_ZONE_i; i < PADDING_i + GHOST_ZONE_i + EXTENT_i; i++)
  {
    std::complex<bElem> z = a[n][m][l][k][j][i],
                        w = b[n][m][l][k][j][i];
    bElem diff = std::abs(z - w);
    if(diff > 1e-6) 
    {
      std::ostringstream errorMsgStream;
      errorMsgStream << name << " result mismatch at [n, m, l, k, j, i] = ["
                    << n << ", " << m << ", " << l << ", " << k << ", " << j << ", " << i << "]: "
                    << z.real() << "+" << z.imag() << "I != "
                    << w.real() << "+" << w.imag() << "I";
      *e = std::runtime_error(errorMsgStream.str());
    }
  }
  // throw error if there was one
  if(e != nullptr)
  {
    throw *e;
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
  static_assert(N <= DIM);
  constexpr int bnd[DIM] = {GHOST_ZONE};

  std::vector<gt::gdesc> slices;
  slices.reserve(N);
  for (int d = 0; d < N; d++) {
    using namespace gt::placeholders;
    slices.push_back(_s(bnd[d] + shift[d], -bnd[d] + shift[d]));
  }

  return gt::view<N>(std::forward<E>(e), slices);
}

template<typename Space>
using gtensor6D = gt::gtensor<gt::complex<bElem>, 6, Space>;
/**
 * @brief Compute the i-j derivative using gtensor
 * 
 * @param out_ptr[out]
 * @param in_ptr[in]
 * @param p1[in]
 * @param p2[in]
 * @param ikj[in]
 * @param i_deriv_coeff[in]
 * 
 * \f$out_ptr := p_1 * \frac{\partial}{\partial x} (in_ptr) + p_2 * 2\pi *i * j *(in_ptr)\f$
 */
void ij_deriv_gtensor(bComplexElem *out_ptr, bComplexElem *in_ptr,
                      bComplexElem *p1, bComplexElem *p2,
                      bComplexElem ikj[PADDED_EXTENT_j], bElem i_deriv_coeff[5])
{
  static_assert(sizeof(bComplexElem) == sizeof(gt::complex<bElem>));
  static_assert(alignof(bComplexElem) == alignof(gt::complex<bElem>));

  auto shape6D = gt::shape(GZ_EXTENT);
  auto shape5D = gt::shape(EXTENT_i, EXTENT_k, EXTENT_l, EXTENT_m, EXTENT_n);
  auto shape_ikj = gt::shape(EXTENT_j);
  // copy in-arrays to gtensor (stripping off the padding)
  auto gt_in = gt::empty<gt::complex<bElem> >(shape6D);
  auto gt_p1 = gt::empty<gt::complex<bElem> >(shape5D);
  auto gt_p2 = gt::empty<gt::complex<bElem> >(shape5D);
  auto gt_ikj = gt::adapt(reinterpret_cast<gt::complex<bElem>*>(ikj + PADDING_j + GHOST_ZONE_j), shape_ikj);
  complexArray6D in_arr = (complexArray6D) in_ptr;
  coeffArray5D p1_arr = (coeffArray5D) p1;
  coeffArray5D p2_arr = (coeffArray5D) p2;
  #pragma omp parallel for collapse(5)
  for(long n = PADDING_n; n < PADDING_n + GZ_EXTENT_n; ++n)
  for(long m = PADDING_m; m < PADDING_m + GZ_EXTENT_m; ++m)
  for(long l = PADDING_l; l < PADDING_l + GZ_EXTENT_l; ++l)
  for(long k = PADDING_k; k < PADDING_k + GZ_EXTENT_k; ++k)
  for(long j = PADDING_j; j < PADDING_j + GZ_EXTENT_j; ++j)
  #pragma omp simd
  for(long i = PADDING_i; i < PADDING_i + GZ_EXTENT_i; ++i)
  {
    gt_in(i - PADDING_i, j - PADDING_j, k - PADDING_k, l - PADDING_l, m - PADDING_m, n - PADDING_n) = in_arr[n][m][l][k][j][i];
    // don't copy ghost-zone into coefficients!
    if(   n < PADDING_n + GHOST_ZONE_n || n >= PADDING_n + GHOST_ZONE_n + EXTENT_n
       || m < PADDING_m + GHOST_ZONE_m || m >= PADDING_m + GHOST_ZONE_m + EXTENT_m
       || l < PADDING_l + GHOST_ZONE_l || l >= PADDING_l + GHOST_ZONE_l + EXTENT_l
       || k < PADDING_k + GHOST_ZONE_k || k >= PADDING_k + GHOST_ZONE_k + EXTENT_k
       || j != 0
       || i < PADDING_i + GHOST_ZONE_i || i >= PADDING_i + GHOST_ZONE_i + EXTENT_i)
    {
      continue;
    }
    gt_p1(i - PADDING_i - GHOST_ZONE_i,
          k - PADDING_k - GHOST_ZONE_k,
          l - PADDING_l - GHOST_ZONE_l,
          m - PADDING_m - GHOST_ZONE_m,
          n - PADDING_n - GHOST_ZONE_n) = p1_arr[n][m][l][k][i];
    gt_p2(i - PADDING_i - GHOST_ZONE_i,
          k - PADDING_k - GHOST_ZONE_k,
          l - PADDING_l - GHOST_ZONE_l,
          m - PADDING_m - GHOST_ZONE_m,
          n - PADDING_n - GHOST_ZONE_n) = p2_arr[n][m][l][k][i];
  }
  auto gt_i_deriv_coeff = gt::adapt(i_deriv_coeff, gt::shape(5));

  // copy the in-arrays to device
  auto gt_in_dev = gt::empty_device<gt::complex<bElem> >(shape6D);
  auto gt_p1_dev = gt::empty_device<gt::complex<bElem> >(shape5D);
  auto gt_p2_dev = gt::empty_device<gt::complex<bElem> >(shape5D);
  auto gt_ikj_dev = gt::empty_device<gt::complex<bElem> >(shape_ikj);
  gt::copy(gt_in, gt_in_dev);
  gt::copy(gt_p1, gt_p1_dev);
  gt::copy(gt_p2, gt_p2_dev);
  gt::copy(gt_ikj, gt_ikj_dev);

  // declare our out-array
  gtensor6D<gt::space::device> gt_out_dev(shape6D);

  // build a function which computes our stencil
  auto compute_ij_deriv = [&gt_out_dev, &gt_in_dev, &gt_p1_dev, &gt_p2_dev, &gt_ikj_dev, &gt_i_deriv_coeff]() -> void {
    using namespace gt::placeholders;
    auto _si = _s(GHOST_ZONE_i, GHOST_ZONE_i + EXTENT_i),
         _sj = _s(GHOST_ZONE_j, GHOST_ZONE_j + EXTENT_j),
         _sk = _s(GHOST_ZONE_k, GHOST_ZONE_k + EXTENT_k),
         _sl = _s(GHOST_ZONE_l, GHOST_ZONE_l + EXTENT_l),
         _sm = _s(GHOST_ZONE_m, GHOST_ZONE_m + EXTENT_m),
         _sn = _s(GHOST_ZONE_n, GHOST_ZONE_n + EXTENT_n);
    gt_out_dev.view(_si, _sj, _sk, _sl, _sm, _sn) =
        gt_p1_dev.view(_all, _newaxis, _all, _all, _all, _all) * (
            gt_i_deriv_coeff(0) * stencil<DIM>(gt_in_dev, {-2, 0, 0, 0, 0, 0}) +
            gt_i_deriv_coeff(1) * stencil<DIM>(gt_in_dev, {-1, 0, 0, 0, 0, 0}) +
            gt_i_deriv_coeff(2) * stencil<DIM>(gt_in_dev, { 0, 0, 0, 0, 0, 0}) +
            gt_i_deriv_coeff(3) * stencil<DIM>(gt_in_dev, {+1, 0, 0, 0, 0, 0}) +
            gt_i_deriv_coeff(4) * stencil<DIM>(gt_in_dev, {+2, 0, 0, 0, 0, 0})
        ) +
        gt_p2_dev.view(_all, _newaxis, _all, _all, _all, _all) *
          gt_ikj_dev.view(_newaxis, _all, _newaxis, _newaxis, _newaxis, _newaxis) *
          gt_in_dev.view(_si, _sj, _sk, _sl, _sm, _sn);

    // actually compute the result
    gt::synchronize();
  };

  // time the function
  std::cout << "gtensor: " << gene_cutime_func(compute_ij_deriv);

  // copy output data back to host
  auto gt_out = gt::empty<gt::complex<bElem> >(shape6D);
  gt::copy(gt_out_dev, gt_out);

  // copy data from gtensor back to padded array
  complexArray6D out_arr = (complexArray6D) out_ptr;
  #pragma omp parallel for collapse(5)
  for(long n = PADDING_n + GHOST_ZONE_n; n < PADDING_n + GHOST_ZONE_n + EXTENT_n; ++n)
  for(long m = PADDING_m + GHOST_ZONE_m; m < PADDING_m + GHOST_ZONE_m + EXTENT_m; ++m)
  for(long l = PADDING_l + GHOST_ZONE_l; l < PADDING_l + GHOST_ZONE_l + EXTENT_l; ++l)
  for(long k = PADDING_k + GHOST_ZONE_k; k < PADDING_k + GHOST_ZONE_k + EXTENT_k; ++k)
  for(long j = PADDING_j + GHOST_ZONE_j; j < PADDING_j + GHOST_ZONE_j + EXTENT_j; ++j)
  #pragma omp simd
  for(long i = PADDING_i + GHOST_ZONE_i; i < PADDING_i + GHOST_ZONE_i + EXTENT_i; ++i)
  {
    out_arr[n][m][l][k][j][i]
      = reinterpret_cast<bComplexElem&>(gt_out(i - PADDING_i, j - PADDING_j, k - PADDING_k, l - PADDING_l, m - PADDING_m, n - PADDING_n));
  }
}

/**
 * @brief the i-j deriv kernel using hand-written bricks code
 */
void ij_deriv_bricks(bComplexElem *out_ptr, bComplexElem *in_ptr,
                     bComplexElem *p1, bComplexElem *p2,
                     bComplexElem ikj[PADDED_EXTENT_j], bElem i_deriv_coeff[5])
{
  // set up brick info and move to device
  unsigned *field_grid_ptr;
  unsigned *coeff_grid_ptr;

  auto fieldBrickInfo = init_grid<DIM>(field_grid_ptr, {GZ_BRICK_EXTENT});
  auto coeffBrickInfo = init_grid<DIM - 1>(coeff_grid_ptr, {GZ_BRICK_EXTENT_i, GZ_BRICK_EXTENT_k, GZ_BRICK_EXTENT_l, GZ_BRICK_EXTENT_m, GZ_BRICK_EXTENT_n});
  unsigned *field_grid_ptr_dev;
  unsigned *coeff_grid_ptr_dev;
  {
    unsigned num_field_bricks = NUM_GZ_BRICKS;
    unsigned num_coeff_bricks = NUM_GZ_BRICKS / GZ_BRICK_EXTENT_j;
    cudaCheck(cudaMalloc(&field_grid_ptr_dev, num_field_bricks * sizeof(unsigned)));
    cudaCheck(cudaMalloc(&coeff_grid_ptr_dev, num_coeff_bricks * sizeof(unsigned)));
    cudaCheck(cudaMemcpy(field_grid_ptr_dev, field_grid_ptr, num_field_bricks * sizeof(unsigned), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(coeff_grid_ptr_dev, coeff_grid_ptr, num_coeff_bricks * sizeof(unsigned), cudaMemcpyHostToDevice));
  }
  BrickInfo<DIM> *fieldBrickInfo_dev;
  BrickInfo<DIM - 1> *coeffBrickInfo_dev;
  BrickInfo<DIM> _fieldBrickInfo_dev = movBrickInfo(fieldBrickInfo, cudaMemcpyHostToDevice);
  BrickInfo<DIM - 1> _coeffBrickInfo_dev = movBrickInfo(coeffBrickInfo, cudaMemcpyHostToDevice);
  {
    cudaCheck(cudaMalloc(&fieldBrickInfo_dev, sizeof(decltype(fieldBrickInfo))));
    cudaCheck(cudaMalloc(&coeffBrickInfo_dev, sizeof(decltype(coeffBrickInfo))));
    cudaCheck(cudaMemcpy(fieldBrickInfo_dev, &_fieldBrickInfo_dev, sizeof(decltype(fieldBrickInfo)), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(coeffBrickInfo_dev, &_coeffBrickInfo_dev, sizeof(decltype(coeffBrickInfo)), cudaMemcpyHostToDevice));
  }

  // setup brick storage on host
  auto fieldBrickStorage = BrickStorage::allocate(fieldBrickInfo.nbricks, 2 * FieldBrick::BRICKSIZE);

  FieldBrick bIn(&fieldBrickInfo, fieldBrickStorage, 0);
  FieldBrick bOut(&fieldBrickInfo, fieldBrickStorage, FieldBrick::BRICKSIZE);

  auto coeffBrickStorage = BrickStorage::allocate(coeffBrickInfo.nbricks, 2 * PreCoeffBrick::BRICKSIZE);
  PreCoeffBrick bP1(&coeffBrickInfo, coeffBrickStorage, 0);
  PreCoeffBrick bP2(&coeffBrickInfo, coeffBrickStorage, PreCoeffBrick::BRICKSIZE);

  copyToBrick<DIM>({GZ_EXTENT}, {PADDING}, {0,0,0,0,0,0}, in_ptr, field_grid_ptr, bIn);

  // double-check our copying process
  iter_grid<DIM>({GZ_EXTENT}, {PADDING}, {0,0,0,0,0,0}, in_ptr, field_grid_ptr, bIn, [](bComplexElem &brick, bComplexElem *arr) -> void {
    if(brick != *arr)
    {
      char errorMsg[1000];
      sprintf(errorMsg, "Bricks input check failure! %f+%f*I != %f+%f*I",
              brick.real(), brick.imag(), arr->real(), arr->imag());
      throw std::runtime_error(errorMsg);
    }
  });

  const std::vector<long> coeffDimList = {GZ_EXTENT_i, GZ_EXTENT_k, GZ_EXTENT_l, GZ_EXTENT_m, GZ_EXTENT_n};
  const std::vector<long> coeffPadding = {PADDING_i, PADDING_k, PADDING_l, PADDING_m, PADDING_n};
  const std::vector<long> coeffGZ = {0,0,0,0,0,0};
  copyToBrick<DIM - 1>(coeffDimList, coeffPadding, coeffGZ, p1, coeff_grid_ptr, bP1);
  copyToBrick<DIM - 1>(coeffDimList, coeffPadding, coeffGZ, p2, coeff_grid_ptr, bP2);

  // move storage to device
  BrickStorage fieldBrickStorage_dev = movBrickStorage(fieldBrickStorage, cudaMemcpyHostToDevice);
  BrickStorage coeffBrickStorage_dev = movBrickStorage(coeffBrickStorage, cudaMemcpyHostToDevice);

  // set up i-k-j and i-deriv coefficients on the device
  bComplexElem *ikj_dev = nullptr;
  bElem *i_deriv_coeff_dev = nullptr;
  {
    unsigned ikj_size = PADDED_EXTENT_j * sizeof(bComplexElem);
    cudaCheck(cudaMalloc(&ikj_dev, ikj_size));
    cudaCheck(cudaMemcpy(ikj_dev, ikj, ikj_size, cudaMemcpyHostToDevice));

    unsigned i_deriv_coeff_size = 5 * sizeof(bElem);
    cudaCheck(cudaMalloc(&i_deriv_coeff_dev, i_deriv_coeff_size));
    cudaCheck(cudaMemcpy(i_deriv_coeff_dev, i_deriv_coeff, i_deriv_coeff_size, cudaMemcpyHostToDevice));
  }

  // build function to actually run computation
  auto compute_ij_deriv = [&fieldBrickInfo_dev,
                           &fieldBrickStorage_dev,
                           &coeffBrickInfo_dev,
                           &coeffBrickStorage_dev,
                           &field_grid_ptr_dev,
                           &coeff_grid_ptr_dev,
                           &ikj_dev,
                           &i_deriv_coeff_dev]() -> void {
    FieldBrick bIn(fieldBrickInfo_dev, fieldBrickStorage_dev, 0);
    FieldBrick bOut(fieldBrickInfo_dev, fieldBrickStorage_dev, FieldBrick::BRICKSIZE);
    PreCoeffBrick bP1(coeffBrickInfo_dev, coeffBrickStorage_dev, 0);
    PreCoeffBrick bP2(coeffBrickInfo_dev, coeffBrickStorage_dev, PreCoeffBrick::BRICKSIZE);
    dim3 block(BRICK_EXTENT_i, BRICK_EXTENT_j, NUM_BRICKS / BRICK_EXTENT_i / BRICK_EXTENT_j),
        thread(BDIM_i, BDIM_j,  NUM_ELEMENTS_PER_BRICK / BDIM_i / BDIM_j);
    ij_deriv_brick_kernel<< < block, thread >> >(
                          (unsigned (*)[GZ_BRICK_EXTENT_m][GZ_BRICK_EXTENT_l][GZ_BRICK_EXTENT_k][GZ_BRICK_EXTENT_j][GZ_BRICK_EXTENT_i]) field_grid_ptr_dev,
                          (unsigned (*)[GZ_BRICK_EXTENT_m][GZ_BRICK_EXTENT_l][GZ_BRICK_EXTENT_k][GZ_BRICK_EXTENT_i]) coeff_grid_ptr_dev,
                          bIn,
                          bOut,
                          bP1,
                          bP2,
                          ikj_dev,
                          i_deriv_coeff_dev);
  };

  // time function
  std::cout << "bricks: " << gene_cutime_func(compute_ij_deriv);

  // copy back to host
  cudaCheck(cudaMemcpy(fieldBrickStorage.dat.get(),
                       fieldBrickStorage_dev.dat.get(),
                       fieldBrickStorage.chunks * fieldBrickStorage.step * sizeof(bElem),
                       cudaMemcpyDeviceToHost));
  cudaDeviceSynchronize();
  copyFromBrick<DIM>({EXTENT}, {PADDING}, {GHOST_ZONE}, out_ptr, field_grid_ptr, bOut);

  // free allocated memory
  cudaCheck(cudaFree(i_deriv_coeff_dev));
  cudaCheck(cudaFree(ikj_dev));
  cudaCheck(cudaFree(_coeffBrickInfo_dev.adj));
  cudaCheck(cudaFree(_fieldBrickInfo_dev.adj));
  cudaCheck(cudaFree(coeffBrickInfo_dev));
  cudaCheck(cudaFree(fieldBrickInfo_dev));
  cudaCheck(cudaFree(coeff_grid_ptr_dev));
  cudaCheck(cudaFree(field_grid_ptr_dev));
  free(coeffBrickInfo.adj);
  free(fieldBrickInfo.adj);
  free(coeff_grid_ptr);
  free(field_grid_ptr);
}

/**
 * @brief 1-D stencil fused with multiplication by 1-D array
 * 
 * Based on https://github.com/wdmapp/gtensor/blob/d07000b15d253cdeb44942b52f3d2caf4522faa0/benchmarks/ij_deriv.cxx
 */
void ij_deriv(bool run_bricks, bool run_gtensor) {
  std::cout << "Setting up i-j deriv arrays" << std::endl;
  // build in/out arrays
  bComplexElem *in_ptr = randomComplexArray({PADDED_EXTENT}),
               *out_ptr = zeroComplexArray({PADDED_EXTENT});
  // build coefficients needed for stencil computation
  bComplexElem *p1 = randomComplexArray({PADDED_EXTENT_i, PADDED_EXTENT_k, PADDED_EXTENT_l, PADDED_EXTENT_m, PADDED_EXTENT_n}),
               *p2 = randomComplexArray({PADDED_EXTENT_i, PADDED_EXTENT_k, PADDED_EXTENT_l, PADDED_EXTENT_m, PADDED_EXTENT_n});
  bComplexElem ikj[PADDED_EXTENT_j];
  for(int j = PADDING_j; j < PADDING_j + EXTENT_j; ++j) ikj[j] = bComplexElem(0, 2 * pi * (j - PADDING_j));
  bElem i_deriv_coeff[5] = {1. / 12., -2. / 3., 0., 2. / 3., -1 / 12.};

  // compute stencil on CPU for correctness check
  std::cout << "Computing correctness check" << std::endl;
  bComplexElem *out_check_ptr = zeroComplexArray({PADDED_EXTENT});
  complexArray6D out_check_arr = (complexArray6D) out_check_ptr;
  complexArray6D in_arr = (complexArray6D) in_ptr;
  coeffArray5D p1_arr = (coeffArray5D) p1;
  coeffArray5D p2_arr = (coeffArray5D) p2;

  #pragma omp parallel for collapse(5)
  for(long n = PADDING_n + GHOST_ZONE_n; n < PADDING_n + GHOST_ZONE_n + EXTENT_n; ++n)
  for(long m = PADDING_m + GHOST_ZONE_m; m < PADDING_m + GHOST_ZONE_m + EXTENT_m; ++m)
  for(long l = PADDING_l + GHOST_ZONE_l; l < PADDING_l + GHOST_ZONE_l + EXTENT_l; ++l)
  for(long k = PADDING_k + GHOST_ZONE_k; k < PADDING_k + GHOST_ZONE_k + EXTENT_k; ++k)
  for(long j = PADDING_j + GHOST_ZONE_j; j < PADDING_j + GHOST_ZONE_j + EXTENT_j; ++j)
  #pragma omp simd
  for(long i = PADDING_i + GHOST_ZONE_i; i < PADDING_i + GHOST_ZONE_i + EXTENT_i; ++i)
  {
    out_check_arr[n][m][l][k][j][i] = p1_arr[n][m][l][k][i] * (
      i_deriv_coeff[0] * in_arr[n][m][l][k][j][i - 2] +
      i_deriv_coeff[1] * in_arr[n][m][l][k][j][i - 1] +
      i_deriv_coeff[2] * in_arr[n][m][l][k][j][i + 0] +
      i_deriv_coeff[3] * in_arr[n][m][l][k][j][i + 1] +
      i_deriv_coeff[4] * in_arr[n][m][l][k][j][i + 2]
    ) + 
    p2_arr[n][m][l][k][i] * ikj[j] * in_arr[n][m][l][k][j][i];
  }

  complexArray6D out_arr = (complexArray6D) out_ptr;

  // run computations
  std::cout << "Starting ij_deriv benchmarks" << std::endl;
  if(run_bricks)
  {
    ij_deriv_bricks(out_ptr, in_ptr, p1, p2, ikj, i_deriv_coeff);
    check_close(out_arr, out_check_arr, "bricks");
    std::cout << " PASSED" << std::endl;
  }

  if(run_gtensor)
  {
    ij_deriv_gtensor(out_ptr, in_ptr, p1, p2, ikj, i_deriv_coeff);
    check_close(out_arr, out_check_arr, "gtensor");
    std::cout << " PASSED" << std::endl;
  }

  std::cout << "done" << std::endl;

  free(out_check_ptr);
  free(p2);
  free(p1);
  free(out_ptr);
  free(in_ptr);
}

/**
 * @brief Compute the arakawa derivative using gtensor
 * 
 * @param out_ptr[out]
 * @param in_ptr[in]
 * @param coeff[in]
 */
void semi_arakawa_gtensor(bComplexElem *out_ptr, bComplexElem *in_ptr, bElem *coeff)
{
  auto shape6D = gt::shape(GZ_EXTENT);
  auto coeffShape = gt::shape(EXTENT_i, ARAKAWA_STENCIL_SIZE, EXTENT_k, EXTENT_l, EXTENT_m, EXTENT_n);
  // copy in-arrays to gtensor (stripping off the padding)
  auto gt_in = gt::empty<gt::complex<bElem> >(shape6D);
  auto gt_coeff = gt::empty<bElem>(coeffShape);
  complexArray6D in_arr = (complexArray6D) in_ptr;
  auto coeff_arr = (bElem (*)[PADDED_EXTENT_m][PADDED_EXTENT_l][PADDED_EXTENT_k][PADDED_EXTENT_i][ARAKAWA_STENCIL_SIZE]) coeff;
  #pragma omp parallel for collapse(5)
  for(long n = PADDING_n; n < PADDING_n + GZ_EXTENT_n; ++n)
  for(long m = PADDING_m; m < PADDING_m + GZ_EXTENT_m; ++m)
  for(long l = PADDING_l; l < PADDING_l + GZ_EXTENT_l; ++l)
  for(long k = PADDING_k; k < PADDING_k + GZ_EXTENT_k; ++k)
  for(long j = PADDING_j; j < PADDING_j + GZ_EXTENT_j; ++j)
  #pragma omp simd
  for(long i = PADDING_i; i < PADDING_i + GZ_EXTENT_i; ++i)
  {
    gt_in(i - PADDING_i, j - PADDING_j, k - PADDING_k, l - PADDING_l, m - PADDING_m, n - PADDING_n) = in_arr[n][m][l][k][j][i];
    // don't copy ghost-zone into coefficients!
    if(   n < PADDING_n + GHOST_ZONE_n || n >= PADDING_n + GHOST_ZONE_n + EXTENT_n
       || m < PADDING_m + GHOST_ZONE_m || m >= PADDING_m + GHOST_ZONE_m + EXTENT_m
       || l < PADDING_l + GHOST_ZONE_l || l >= PADDING_l + GHOST_ZONE_l + EXTENT_l
       || k < PADDING_k + GHOST_ZONE_k || k >= PADDING_k + GHOST_ZONE_k + EXTENT_k
       || j != 0
       || i < PADDING_i + GHOST_ZONE_i || i >= PADDING_i + GHOST_ZONE_i + EXTENT_i)
    {
      continue;
    }
    for(int coeff_index = 0; coeff_index < ARAKAWA_STENCIL_SIZE; ++coeff_index)
    {
      gt_coeff(i - PADDING_i - GHOST_ZONE_i,
              coeff_index,
              k - PADDING_k - GHOST_ZONE_k,
              l - PADDING_l - GHOST_ZONE_l,
              m - PADDING_m - GHOST_ZONE_m,
              n - PADDING_n - GHOST_ZONE_n) = coeff_arr[n][m][l][k][i][coeff_index];
    }
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
    auto _si = _s(GHOST_ZONE_i, GHOST_ZONE_i + EXTENT_i),
         _sj = _s(GHOST_ZONE_j, GHOST_ZONE_j + EXTENT_j),
         _sk = _s(GHOST_ZONE_k, GHOST_ZONE_k + EXTENT_k),
         _sl = _s(GHOST_ZONE_l, GHOST_ZONE_l + EXTENT_l),
         _sm = _s(GHOST_ZONE_m, GHOST_ZONE_m + EXTENT_m),
         _sn = _s(GHOST_ZONE_n, GHOST_ZONE_n + EXTENT_n);

    auto coeff = [&](int s) { return gt_coeff_dev.view(_all, s, _newaxis, _all, _all, _all, _all); };
    gt_out_dev.view(_si, _sj, _sk, _sl, _sm, _sn) =
      coeff(0 ) * stencil<DIM>(gt_in_dev, {0, 0, +0, -2, 0, 0}) +
      coeff(1 ) * stencil<DIM>(gt_in_dev, {0, 0, -1, -1, 0, 0}) +
      coeff(2 ) * stencil<DIM>(gt_in_dev, {0, 0, +0, -1, 0, 0}) +
      coeff(3 ) * stencil<DIM>(gt_in_dev, {0, 0, +1, -1, 0, 0}) +
      coeff(4 ) * stencil<DIM>(gt_in_dev, {0, 0, -2, +0, 0, 0}) +
      coeff(5 ) * stencil<DIM>(gt_in_dev, {0, 0, -1, +0, 0, 0}) +
      coeff(6 ) * stencil<DIM>(gt_in_dev, {0, 0, +0, +0, 0, 0}) +
      coeff(7 ) * stencil<DIM>(gt_in_dev, {0, 0, +1, +0, 0, 0}) +
      coeff(8 ) * stencil<DIM>(gt_in_dev, {0, 0, +2, +0, 0, 0}) +
      coeff(9 ) * stencil<DIM>(gt_in_dev, {0, 0, -1, +1, 0, 0}) +
      coeff(10) * stencil<DIM>(gt_in_dev, {0, 0, +0, +1, 0, 0}) +
      coeff(11) * stencil<DIM>(gt_in_dev, {0, 0, +1, +1, 0, 0}) +
      coeff(12) * stencil<DIM>(gt_in_dev, {0, 0, +0, +2, 0, 0});  

    // actually compute the result
    gt::synchronize();
  };

  // time the function
  std::cout << "gtensor: " << gene_cutime_func(compute_semi_arakawa);

  // copy output data back to host
  auto gt_out = gt::empty<gt::complex<bElem> >(shape6D);
  gt::copy(gt_out_dev, gt_out);

  // copy data from gtensor back to padded array
  complexArray6D out_arr = (complexArray6D) out_ptr;
  #pragma omp parallel for collapse(5)
  for(long n = PADDING_n + GHOST_ZONE_n; n < PADDING_n + GHOST_ZONE_n + EXTENT_n; ++n)
  for(long m = PADDING_m + GHOST_ZONE_m; m < PADDING_m + GHOST_ZONE_m + EXTENT_m; ++m)
  for(long l = PADDING_l + GHOST_ZONE_l; l < PADDING_l + GHOST_ZONE_l + EXTENT_l; ++l)
  for(long k = PADDING_k + GHOST_ZONE_k; k < PADDING_k + GHOST_ZONE_k + EXTENT_k; ++k)
  for(long j = PADDING_j + GHOST_ZONE_j; j < PADDING_j + GHOST_ZONE_j + EXTENT_j; ++j)
  #pragma omp simd
  for(long i = PADDING_i + GHOST_ZONE_i; i < PADDING_i + GHOST_ZONE_i + EXTENT_i; ++i)
  {
    out_arr[n][m][l][k][j][i]
      = reinterpret_cast<bComplexElem&>(gt_out(i - PADDING_i, j - PADDING_j, k - PADDING_k, l - PADDING_l, m - PADDING_m, n - PADDING_n));
  }
}

/**
 * @brief the arakawa deriv kernel using hand-written bricks code
 */
void semi_arakawa_bricks(bComplexElem *out_ptr, bComplexElem *in_ptr, bElem *coeff)
{
  // copy coeff from STENCIL_SIZE x GRID to GRID x STENCIL_SIZE
  bElem *reordered_coeff = zeroArray({PADDED_EXTENT_i, PADDED_EXTENT_k, PADDED_EXTENT_l, PADDED_EXTENT_m, PADDED_EXTENT_n, ARAKAWA_STENCIL_SIZE});
  auto reordered_coeff_arr = (bElem (*)[PADDED_EXTENT_n][PADDED_EXTENT_m][PADDED_EXTENT_l][PADDED_EXTENT_k][PADDED_EXTENT_i]) reordered_coeff;
  auto coeff_arr = (bElem (*)[PADDED_EXTENT_m][PADDED_EXTENT_l][PADDED_EXTENT_k][PADDED_EXTENT_i][ARAKAWA_STENCIL_SIZE]) coeff;
  #pragma omp parallel for collapse(5)
  for(long n = 0; n < PADDED_EXTENT_n; n++)
  for(long m = 0; m < PADDED_EXTENT_m; m++)
  for(long l = 0; l < PADDED_EXTENT_l; l++)
  for(long k = 0; k < PADDED_EXTENT_k; k++)
  for(long i = 0; i < PADDED_EXTENT_i; i++)
  for(unsigned j = 0; j < ARAKAWA_STENCIL_SIZE; ++j)
  reordered_coeff_arr[j][n][m][l][k][i] = coeff_arr[n][m][l][k][i][j];

  // set up brick info and move to device
  unsigned *field_grid_ptr = nullptr;
  unsigned *coeff_grid_ptr = nullptr;

  auto fieldBrickInfo = init_grid<DIM>(field_grid_ptr, {GZ_BRICK_EXTENT});
  auto coeffBrickInfo = init_grid<DIM - 1>(coeff_grid_ptr, {GZ_BRICK_EXTENT_i, GZ_BRICK_EXTENT_k, GZ_BRICK_EXTENT_l, GZ_BRICK_EXTENT_m, GZ_BRICK_EXTENT_n});
  unsigned *field_grid_ptr_dev = nullptr;
  unsigned *coeff_grid_ptr_dev = nullptr;
  {
    unsigned num_field_bricks = NUM_GZ_BRICKS;
    unsigned num_coeff_bricks = NUM_GZ_BRICKS / GZ_BRICK_EXTENT_j;
    cudaCheck(cudaMalloc(&field_grid_ptr_dev, num_field_bricks * sizeof(unsigned)));
    cudaCheck(cudaMalloc(&coeff_grid_ptr_dev, num_coeff_bricks * sizeof(unsigned)));
    cudaCheck(cudaMemcpy(field_grid_ptr_dev, field_grid_ptr, num_field_bricks * sizeof(unsigned), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(coeff_grid_ptr_dev, coeff_grid_ptr, num_coeff_bricks * sizeof(unsigned), cudaMemcpyHostToDevice));
  }
  BrickInfo<DIM> *fieldBrickInfo_dev = nullptr;
  BrickInfo<DIM - 1> *coeffBrickInfo_dev = nullptr;
  BrickInfo<DIM> _fieldBrickInfo_dev = movBrickInfo(fieldBrickInfo, cudaMemcpyHostToDevice);
  BrickInfo<DIM - 1> _coeffBrickInfo_dev = movBrickInfo(coeffBrickInfo, cudaMemcpyHostToDevice);
  {
    cudaCheck(cudaMalloc(&fieldBrickInfo_dev, sizeof(decltype(fieldBrickInfo))));
    cudaCheck(cudaMalloc(&coeffBrickInfo_dev, sizeof(decltype(coeffBrickInfo))));
    cudaCheck(cudaMemcpy(fieldBrickInfo_dev, &_fieldBrickInfo_dev, sizeof(decltype(fieldBrickInfo)), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(coeffBrickInfo_dev, &_coeffBrickInfo_dev, sizeof(decltype(coeffBrickInfo)), cudaMemcpyHostToDevice));
  }

  // setup brick storage on host
  auto fieldBrickStorage = BrickStorage::allocate(fieldBrickInfo.nbricks, 2 * FieldBrick::BRICKSIZE);

  FieldBrick bIn(&fieldBrickInfo, fieldBrickStorage, 0);
  FieldBrick bOut(&fieldBrickInfo, fieldBrickStorage, FieldBrick::BRICKSIZE);

  auto coeffBrickStorage = BrickStorage::allocate(coeffBrickInfo.nbricks, ARAKAWA_STENCIL_SIZE * RealCoeffBrick::BRICKSIZE);
  std::vector<RealCoeffBrick> bCoeffs;
  bCoeffs.reserve(ARAKAWA_STENCIL_SIZE);
  for(unsigned i = 0; i < ARAKAWA_STENCIL_SIZE; ++i)
  {
    bCoeffs.push_back(RealCoeffBrick(&coeffBrickInfo, coeffBrickStorage, i * RealCoeffBrick::BRICKSIZE));
  }

  copyToBrick<DIM>({GZ_EXTENT}, {PADDING}, {0,0,0,0,0,0}, in_ptr, field_grid_ptr, bIn);
  const std::vector<long> coeffDimList = {GZ_EXTENT_i, GZ_EXTENT_k, GZ_EXTENT_l, GZ_EXTENT_m, GZ_EXTENT_n};
  const std::vector<long> coeffPadding = {PADDING_i, PADDING_k, PADDING_l, PADDING_m, PADDING_n};
  const std::vector<long> coeffGZ = {0,0,0,0,0};
  for(unsigned i = 0; i < ARAKAWA_STENCIL_SIZE; ++i)
  {
    bElem *coeff_i_ptr = reordered_coeff + i * NUM_PADDED_ELEMENTS / PADDED_EXTENT_j;
    copyToBrick<DIM - 1>(coeffDimList, coeffPadding, coeffGZ, coeff_i_ptr, coeff_grid_ptr, bCoeffs[i]);
  }

  // move storage to device
  BrickStorage fieldBrickStorage_dev = movBrickStorage(fieldBrickStorage, cudaMemcpyHostToDevice);
  BrickStorage coeffBrickStorage_dev = movBrickStorage(coeffBrickStorage, cudaMemcpyHostToDevice);

  // copy coefficient bricks to device
  RealCoeffBrick *bCoeffs_dev = nullptr;
  cudaCheck(cudaMalloc(&bCoeffs_dev, ARAKAWA_STENCIL_SIZE * sizeof(RealCoeffBrick)));
  for(unsigned i = 0; i < ARAKAWA_STENCIL_SIZE; ++i)
  {
    RealCoeffBrick brickToCopy(coeffBrickInfo_dev, coeffBrickStorage_dev, i * RealCoeffBrick::BRICKSIZE);
    cudaCheck(cudaMemcpy(bCoeffs_dev + i, &brickToCopy, sizeof(RealCoeffBrick), cudaMemcpyHostToDevice));
  }

  // build function to actually run computation
  auto compute_semi_arakawa = [&fieldBrickInfo_dev,
                               &fieldBrickStorage_dev,
                               &coeffBrickInfo_dev,
                               &coeffBrickStorage_dev,
                               &field_grid_ptr_dev,
                               &coeff_grid_ptr_dev,
                               &bCoeffs_dev]() -> void {
    FieldBrick bIn(fieldBrickInfo_dev, fieldBrickStorage_dev, 0);
    FieldBrick bOut(fieldBrickInfo_dev, fieldBrickStorage_dev, FieldBrick::BRICKSIZE);
    dim3 block(BRICK_EXTENT_i, BRICK_EXTENT_j, NUM_BRICKS / BRICK_EXTENT_i / BRICK_EXTENT_j),
        thread(BDIM_i, BDIM_j,  NUM_ELEMENTS_PER_BRICK / BDIM_i / BDIM_j);
    semi_arakawa_brick_kernel<< < block, thread >> >(
                          (unsigned (*)[GZ_BRICK_EXTENT_m][GZ_BRICK_EXTENT_l][GZ_BRICK_EXTENT_k][GZ_BRICK_EXTENT_j][GZ_BRICK_EXTENT_i]) field_grid_ptr_dev,
                          (unsigned (*)[GZ_BRICK_EXTENT_m][GZ_BRICK_EXTENT_l][GZ_BRICK_EXTENT_k][GZ_BRICK_EXTENT_i]) coeff_grid_ptr_dev,
                          bIn,
                          bOut,
                          bCoeffs_dev);
  };

  // time function
  std::cout << "bricks: " << gene_cutime_func(compute_semi_arakawa);

  // copy back to host
  cudaCheck(cudaMemcpy(fieldBrickStorage.dat.get(),
                       fieldBrickStorage_dev.dat.get(),
                       fieldBrickStorage.chunks * fieldBrickStorage.step * sizeof(bElem),
                       cudaMemcpyDeviceToHost));
  cudaDeviceSynchronize();
  copyFromBrick<DIM>({EXTENT}, {PADDING}, {GHOST_ZONE}, out_ptr, field_grid_ptr, bOut);

  // free allocated memory
  cudaCheck(cudaFree(bCoeffs_dev));
  cudaCheck(cudaFree(_coeffBrickInfo_dev.adj));
  cudaCheck(cudaFree(_fieldBrickInfo_dev.adj));
  cudaCheck(cudaFree(coeffBrickInfo_dev));
  cudaCheck(cudaFree(fieldBrickInfo_dev));
  cudaCheck(cudaFree(coeff_grid_ptr_dev));
  cudaCheck(cudaFree(field_grid_ptr_dev));
  free(coeffBrickInfo.adj);
  free(fieldBrickInfo.adj);
  free(coeff_grid_ptr);
  free(field_grid_ptr);
  free(reordered_coeff);
}

/**
 * @brief 2-D stencil with variable coefficients
 * 
 * Based on https://github.com/wdmapp/gtensor/blob/41cf4fe26625f8d7ba2d0d3886a54ae6415a2017/benchmarks/bench_hypz.cxx#L30-L49
 */
void semi_arakawa(bool run_bricks, bool run_gtensor) {
  std::cout << "Setting up semi-arakawa arrays" << std::endl;
  // build in/out arrays
  bComplexElem *in_ptr = randomComplexArray({PADDED_EXTENT}),
               *out_ptr = zeroComplexArray({PADDED_EXTENT});
  // build coefficients needed for stencil computation
  bElem *coeff = randomArray({ARAKAWA_STENCIL_SIZE, PADDED_EXTENT_i, PADDED_EXTENT_k, PADDED_EXTENT_l, PADDED_EXTENT_m, PADDED_EXTENT_n});

  // compute stencil on CPU for correctness check
  std::cout << "Computing correctness check" << std::endl;
  bComplexElem *out_check_ptr = zeroComplexArray({PADDED_EXTENT});
  complexArray6D out_check_arr = (complexArray6D) out_check_ptr;
  complexArray6D in_arr = (complexArray6D) in_ptr;
  auto coeff_arr = (bElem (*)[PADDED_EXTENT_m][PADDED_EXTENT_l][PADDED_EXTENT_k][PADDED_EXTENT_i][ARAKAWA_STENCIL_SIZE]) coeff;

  static_assert(EXTENT_l % TILE == 0);
  static_assert(EXTENT_k % TILE == 0);
  #pragma omp parallel for collapse(4)
  for(long n = PADDING_n + GHOST_ZONE_n; n < PADDING_n + GHOST_ZONE_n + EXTENT_n; ++n)
  for(long m = PADDING_m + GHOST_ZONE_m; m < PADDING_m + GHOST_ZONE_m + EXTENT_m; ++m)
  for(long tl = PADDING_l + GHOST_ZONE_l; tl < PADDING_l + GHOST_ZONE_l + EXTENT_l; tl += TILE)
  for(long tk = PADDING_k + GHOST_ZONE_k; tk < PADDING_k + GHOST_ZONE_k + EXTENT_k; tk += TILE)
  for(long l = tl; l < tl + TILE; l++)
  for(long k = tk; k < tk + TILE; k++)
  for(long j = PADDING_j + GHOST_ZONE_j; j < PADDING_j + GHOST_ZONE_j + EXTENT_j; j++)
  #pragma omp simd
  for(long i = PADDING_i + GHOST_ZONE_i; i < PADDING_i + GHOST_ZONE_i + EXTENT_i; i++)
  {
    out_check_arr[n][m][l][k][j][i] = 
        coeff_arr[n][m][k][l][i][ 0] * in_arr[n][m][l-2][k+0][j][i] +
        coeff_arr[n][m][k][l][i][ 1] * in_arr[n][m][l-1][k-1][j][i] +
        coeff_arr[n][m][k][l][i][ 2] * in_arr[n][m][l-1][k+0][j][i] +
        coeff_arr[n][m][k][l][i][ 3] * in_arr[n][m][l-1][k+1][j][i] +
        coeff_arr[n][m][k][l][i][ 4] * in_arr[n][m][l+0][k-2][j][i] +
        coeff_arr[n][m][k][l][i][ 5] * in_arr[n][m][l+0][k-1][j][i] +
        coeff_arr[n][m][k][l][i][ 6] * in_arr[n][m][l+0][k+0][j][i] +
        coeff_arr[n][m][k][l][i][ 7] * in_arr[n][m][l+0][k+1][j][i] +
        coeff_arr[n][m][k][l][i][ 8] * in_arr[n][m][l+0][k+2][j][i] +
        coeff_arr[n][m][k][l][i][ 9] * in_arr[n][m][l+1][k-1][j][i] +
        coeff_arr[n][m][k][l][i][10] * in_arr[n][m][l+1][k+0][j][i] +
        coeff_arr[n][m][k][l][i][11] * in_arr[n][m][l+1][k+1][j][i] +
        coeff_arr[n][m][k][l][i][12] * in_arr[n][m][l+2][k+0][j][i];
  }

  complexArray6D out_arr = (complexArray6D) out_ptr;

  // run computations
  std::cout << "Starting semi_arakawa benchmarks" << std::endl;
  if(run_bricks)
  {
    semi_arakawa_bricks(out_ptr, in_ptr, coeff);
    check_close(out_arr, out_check_arr, "bricks");
    std::cout << " PASSED" << std::endl;
  }

  if(run_gtensor)
  {
    semi_arakawa_gtensor(out_ptr, in_ptr, coeff);
    check_close(out_arr, out_check_arr, "gtensor");
    std::cout << " PASSED" << std::endl;
  }

  std::cout << "done" << std::endl;

  free(out_check_ptr);
  free(coeff);
  free(out_ptr);
  free(in_ptr);
}

// usage: (Optional) [num iterations] 
//        (Optional) [num warmup iterations] 
//        (Optional) [which kernel (ij or arakawa)] 
//        (Optional) [g for just gtensor, b for just bricks]
int main(int argc, char * const argv[]) {
  #ifndef NDEBUG
  std::cout << "NDEBUG is not defined" << std::endl;
  #else
  std::cout << "NDEBUG is defined" << std::endl;
  #endif
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

  std::cout << "WARM UP:" << NUM_WARMUP_ITERS << std::endl;
  std::cout << "ITERATIONS:" << NUM_ITERS << std::endl;

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
