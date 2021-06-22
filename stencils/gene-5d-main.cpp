#include "gene-5d.h"

/**
 * @brief wrap timing function with more convenient output
 * 
 * @param f function to time
 * @return std::string a description of the time in GStencil/s
 */
std::string gene_cutime_func(std::function<void()> f)
{
  std::ostringstream string_stream;
  string_stream << NUM_ELEMENTS / cutime_func(f) / 1000000000 << " avg GStencils/s";
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
  _TILEFOR6D {
    if(i < 2 * PADDING_i || i > EXTENT_i) continue;
    if(j < 2 * PADDING_j || j > EXTENT_j) continue;
    if(k < 2 * PADDING_k || k > EXTENT_k) continue;
    if(l < 2 * PADDING_l || l > EXTENT_l) continue;
    if(m < 2 * PADDING_m || m > EXTENT_m) continue;
    if(n < 2 * PADDING_n || n > EXTENT_n) continue;
    std::complex<bElem> z = a[n][m][l][k][j][i],
                        w = b[n][m][l][k][j][i];
    bElem diff = std::abs(z - w);
    if(diff > 1e-6) 
    {
      char errorMsg[1000];
      strcpy(errorMsg, name.data());
      sprintf(errorMsg + name.length(), " result mismatch at [n, m, l, k, j, i] = [%d, %d, %d, %d, %d, %d]! %f+%f*I != %f+%f*I",
              n, m, l, k, j, i,
              z.real(), z.imag(), w.real(), w.imag());
      throw std::runtime_error(errorMsg);
    }
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
  constexpr int bnd[DIM] = {PADDING_i, PADDING_j, PADDING_k, PADDING_l, PADDING_m, PADDING_n};

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

  auto shape6D = gt::shape(PADDED_EXTENT);
  auto shape5D = gt::shape(PADDED_EXTENT_i, PADDED_EXTENT_k, PADDED_EXTENT_l, PADDED_EXTENT_m, PADDED_EXTENT_n);
  auto shape_ikj = gt::shape(PADDED_EXTENT_j);
  // adapt in-arrays to gtensor
  auto gt_in = gt::adapt(reinterpret_cast<gt::complex<bElem>*>(in_ptr), shape6D);
  auto gt_p1 = gt::adapt(reinterpret_cast<gt::complex<bElem>*>(p1), shape5D);
  auto gt_p2 = gt::adapt(reinterpret_cast<gt::complex<bElem>*>(p2), shape5D);
  auto gt_ikj = gt::adapt(reinterpret_cast<gt::complex<bElem>*>(ikj), shape_ikj);
  auto gt_i_deriv_coeff = gt::adapt(i_deriv_coeff, gt::shape(5));

  // double check data copied in correctly
  complexArray6D in_arr = (complexArray6D) in_ptr;
  _TILEFOR6D {
    std::complex<bElem> z = gt_in(i, j, k, l, m, n),
                        w = in_arr[n][m][l][k][j][i];
    bElem diff = std::abs(z - w);
    if(diff > 1e-6) 
    {
      char errorMsg[1000];
      sprintf(errorMsg, "gtensor input copy failure at (n, m, l, k, j, i) = (%d, %d, %d, %d, %d, %d)! %f+%f*I != %f+%f*I",
              n, m, l, k, j, i,
              z.real(), z.imag(), w.real(), w.imag());
      throw std::runtime_error(errorMsg);
    }
  }

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
    auto _si = _s(PADDING_i, PADDING_i + EXTENT_i),
         _sj = _s(PADDING_j, PADDING_j + EXTENT_j),
         _sk = _s(PADDING_k, PADDING_k + EXTENT_k),
         _sl = _s(PADDING_l, PADDING_l + EXTENT_l),
         _sm = _s(PADDING_m, PADDING_m + EXTENT_m),
         _sn = _s(PADDING_n, PADDING_n + EXTENT_n);
    gt_out_dev.view(_si, _sj, _sk, _sl, _sm, _sn) =
        gt_p1_dev.view(_si, _newaxis, _sk, _sl, _sm, _sn) * (
            gt_i_deriv_coeff(0) * stencil<DIM>(gt_in_dev, {-2, 0, 0, 0, 0, 0}) +
            gt_i_deriv_coeff(1) * stencil<DIM>(gt_in_dev, {-1, 0, 0, 0, 0, 0}) +
            gt_i_deriv_coeff(2) * stencil<DIM>(gt_in_dev, { 0, 0, 0, 0, 0, 0}) +
            gt_i_deriv_coeff(3) * stencil<DIM>(gt_in_dev, {+1, 0, 0, 0, 0, 0}) +
            gt_i_deriv_coeff(4) * stencil<DIM>(gt_in_dev, {+2, 0, 0, 0, 0, 0})
        ) +
        gt_p2_dev.view(_si, _newaxis, _sk, _sl, _sm, _sn) *
          gt_ikj_dev.view(_newaxis, _sj, _newaxis, _newaxis, _newaxis, _newaxis) *
          gt_in_dev.view(_si, _sj, _sk, _sl, _sm, _sn);

    // actually compute the result
    gt::synchronize();
  };

  // time the function
  std::cout << "gtensor: " << gene_cutime_func(compute_ij_deriv) << std::endl;

  // copy output data back to host
  auto gt_out = gt::adapt(reinterpret_cast<gt::complex<bElem>*>(out_ptr), shape6D);
  gt::copy(gt_out_dev, gt_out);

  // double check data copied out correctly
  complexArray6D out_arr = (complexArray6D) out_ptr;
  _TILEFOR6D {
    std::complex<bElem> z = gt_out(i, j, k, l, m, n),
                        w = out_arr[n][m][l][k][j][i];
    bElem diff = std::abs(z - w);
    if(diff > 1e-6) 
    {
      char errorMsg[1000];
      sprintf(errorMsg, "gtensor result copy failure at (n, m, l, k, j, i) = (%d, %d, %d, %d, %d, %d)! %f+%f*I != %f+%f*I",
              n, m, l, k, j, i,
              z.real(), z.imag(), w.real(), w.imag());
      throw std::runtime_error(errorMsg);
    }
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

  auto fieldBrickInfo = init_grid<DIM>(field_grid_ptr, {BRICK_EXTENT});
  auto coeffBrickInfo = init_grid<DIM - 1>(coeff_grid_ptr, {BRICK_EXTENT_i, BRICK_EXTENT_k, BRICK_EXTENT_l, BRICK_EXTENT_m, BRICK_EXTENT_n});
  unsigned *field_grid_ptr_dev;
  unsigned *coeff_grid_ptr_dev;
  {
    unsigned num_field_bricks = NUM_BRICKS;
    unsigned num_coeff_bricks = NUM_BRICKS / BRICK_EXTENT_j;
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

  copyToBrick<DIM>({GZ_EXTENT}, {PADDING}, {GHOST_ZONE}, in_ptr, field_grid_ptr, bIn);

  // double-check our copying process
  iter_grid<DIM>({GZ_EXTENT}, {PADDING}, {GHOST_ZONE}, in_ptr, field_grid_ptr, bIn, [](bComplexElem &brick, bComplexElem *arr) -> void {
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
  const std::vector<long> coeffGZ = {GHOST_ZONE_i, GHOST_ZONE_k, GHOST_ZONE_l, GHOST_ZONE_m, GHOST_ZONE_n};
  copyToBrick<DIM - 1>(coeffDimList, coeffPadding, coeffGZ, p1, coeff_grid_ptr, bP1);
  copyToBrick<DIM - 1>(coeffDimList, coeffPadding, coeffGZ, p2, coeff_grid_ptr, bP2);

  // move storage to device
  BrickStorage fieldBrickStorage_dev = movBrickStorage(fieldBrickStorage, cudaMemcpyHostToDevice);
  BrickStorage coeffBrickStorage_dev = movBrickStorage(coeffBrickStorage, cudaMemcpyHostToDevice);

  // set up i-k-j and i-deriv coefficients on the device
  bComplexElem *ikj_dev = nullptr;
  bElem *i_deriv_coeff_dev = nullptr;
  {
    // don't copy over padding, if any
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
                           &bIn,
                           &bOut,
                           &bP1,
                           &bP2,
                           &ikj_dev,
                           &i_deriv_coeff_dev]() -> void {
    FieldBrick bIn(fieldBrickInfo_dev, fieldBrickStorage_dev, 0);
    FieldBrick bOut(fieldBrickInfo_dev, fieldBrickStorage_dev, FieldBrick::BRICKSIZE);
    PreCoeffBrick bP1(coeffBrickInfo_dev, coeffBrickStorage_dev, 0);
    PreCoeffBrick bP2(coeffBrickInfo_dev, coeffBrickStorage_dev, PreCoeffBrick::BRICKSIZE);
    dim3 block(BRICK_EXTENT_i, BRICK_EXTENT_j, NUM_BRICKS / BRICK_EXTENT_i / BRICK_EXTENT_j),
        thread(BDIM_i, BDIM_j,  NUM_ELEMENTS_PER_BRICK / BDIM_i / BDIM_j);
    ij_deriv_brick_kernel<< < block, thread >> >(
                          (unsigned (*)[BRICK_EXTENT_m][BRICK_EXTENT_l][BRICK_EXTENT_k][BRICK_EXTENT_j][BRICK_EXTENT_i]) field_grid_ptr_dev,
                          (unsigned (*)[BRICK_EXTENT_m][BRICK_EXTENT_l][BRICK_EXTENT_k][BRICK_EXTENT_i]) coeff_grid_ptr_dev,
                          bIn,
                          bOut,
                          bP1,
                          bP2,
                          ikj_dev,
                          i_deriv_coeff_dev);
  };

  // time function
  std::cout << "bricks: " << gene_cutime_func(compute_ij_deriv) << std::endl;

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
void ij_deriv() {
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
  _TILEFOR6D out_check_arr[n][m][l][k][j][i] = p1_arr[n][m][l][k][i] * (
    i_deriv_coeff[0] * in_arr[n][m][l][k][j][i - 2] +
    i_deriv_coeff[1] * in_arr[n][m][l][k][j][i - 1] +
    i_deriv_coeff[2] * in_arr[n][m][l][k][j][i + 0] +
    i_deriv_coeff[3] * in_arr[n][m][l][k][j][i + 1] +
    i_deriv_coeff[4] * in_arr[n][m][l][k][j][i + 2]
  ) + 
  p2_arr[n][m][l][k][i] * ikj[j] * in_arr[n][m][l][k][j][i];

  complexArray6D out_arr = (complexArray6D) out_ptr;

  // run computations
  std::cout << "Starting ij_deriv benchmarks" << std::endl;
  ij_deriv_bricks(out_ptr, in_ptr, p1, p2, ikj, i_deriv_coeff);
  check_close(out_arr, out_check_arr);

  ij_deriv_gtensor(out_ptr, in_ptr, p1, p2, ikj, i_deriv_coeff);
  check_close(out_arr, out_check_arr);

  std::cout << "done" << std::endl;

  free((void *)p2);
  free((void *)p1);
  free(out_ptr);
  free((void *)in_ptr);
}


int main() {
  std::cout << "WARM UP:" << CU_WARMUP << std::endl;
  std::cout << "ITERATIONS:" << CU_ITER << std::endl;
  ij_deriv();
  return 0;
}