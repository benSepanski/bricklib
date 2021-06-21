#include "gene-5d.h"

// convenient typedefs for arrays of pointers
typedef bComplexElem (*complexArray5D)[PADDED_EXTENT_l][PADDED_EXTENT_k][PADDED_EXTENT_j][PADDED_EXTENT_i];
typedef bComplexElem (*coeffArray4D)[PADDED_EXTENT_l][PADDED_EXTENT_k][PADDED_EXTENT_i];

/**
 * @brief check that a and b are close in values
 * 
 * @param a an array of shape PADDED_EXTENT
 * @param b an array of shape PADDED_EXTENT
 */
void check_close(complexArray5D a, complexArray5D b)
{
  _TILEFOR5D {
    std::complex<bElem> z = a[m][l][k][j][i],
                        w = b[m][l][k][j][i];
    bElem diff = std::abs(z - w);
    if(diff > 1e-6) 
    {
      char errorMsg[1000];
      sprintf(errorMsg, "Result mismatch at (m, l, k, j, i) = (%d, %d, %d, %d, %d)! %f+%f*I != %f+%f*I",
              m, l, k, j, i,
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
  static_assert(N <= 5);
  constexpr int bnd[5] = {PADDING_i, PADDING_j, PADDING_k, PADDING_l, PADDING_m};

  std::vector<gt::gdesc> slices;
  slices.reserve(N);
  for (int d = 0; d < N; d++) {
    using namespace gt::placeholders;
    slices.push_back(_s(bnd[d] + shift[d], -bnd[d] + shift[d]));
  }

  return gt::view<N>(std::forward<E>(e), slices);
}

template<typename Space>
using gtensor5D = gt::gtensor<gt::complex<bElem>, 5, Space>;
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

  auto shape5D = gt::shape(PADDED_EXTENT);
  auto shape4D = gt::shape(PADDED_EXTENT_i, PADDED_EXTENT_k, PADDED_EXTENT_l, PADDED_EXTENT_m);
  auto shape_ikj = gt::shape(PADDED_EXTENT_j);
  // adapt in-arrays to gtensor
  auto gt_in = gt::adapt(reinterpret_cast<gt::complex<bElem>*>(in_ptr), shape5D);
  auto gt_p1 = gt::adapt(reinterpret_cast<gt::complex<bElem>*>(p1), shape4D);
  auto gt_p2 = gt::adapt(reinterpret_cast<gt::complex<bElem>*>(p2), shape4D);
  auto gt_ikj = gt::adapt(reinterpret_cast<gt::complex<bElem>*>(ikj), shape_ikj);
  auto gt_i_deriv_coeff = gt::adapt(i_deriv_coeff, gt::shape(5));

  // double check data copied in correctly
  complexArray5D in_arr = (complexArray5D) in_ptr;
  _TILEFOR5D {
    std::complex<bElem> z = gt_in(i, j, k, l, m),
                        w = in_arr[m][l][k][j][i];
    bElem diff = std::abs(z - w);
    if(diff > 1e-6) 
    {
      char errorMsg[1000];
      sprintf(errorMsg, "Input copy failure at (m, l, k, j, i) = (%d, %d, %d, %d, %d)! %f+%f*I != %f+%f*I",
              m, l, k, j, i,
              z.real(), z.imag(), w.real(), w.imag());
      throw std::runtime_error(errorMsg);
    }
  }

  // copy the in-arrays to device
  auto gt_in_dev = gt::empty_device<gt::complex<bElem> >(shape5D);
  auto gt_p1_dev = gt::empty_device<gt::complex<bElem> >(shape4D);
  auto gt_p2_dev = gt::empty_device<gt::complex<bElem> >(shape4D);
  auto gt_ikj_dev = gt::empty_device<gt::complex<bElem> >(shape_ikj);
  gt::copy(gt_in, gt_in_dev);
  gt::copy(gt_p1, gt_p1_dev);
  gt::copy(gt_p2, gt_p2_dev);
  gt::copy(gt_ikj, gt_ikj_dev);

  // declare our out-array
  gtensor5D<gt::space::device> gt_out_dev(shape5D);

  // build a function which computes our stencil
  auto compute_ij_deriv = [&gt_out_dev, &gt_in_dev, &gt_p1_dev, &gt_p2_dev, &gt_ikj_dev, &gt_i_deriv_coeff]() -> void {
    using namespace gt::placeholders;
    auto _si = _s(PADDING_i, PADDING_i + EXTENT_i),
         _sj = _s(PADDING_j, PADDING_j + EXTENT_j),
         _sk = _s(PADDING_k, PADDING_k + EXTENT_k),
         _sl = _s(PADDING_l, PADDING_l + EXTENT_l),
         _sm = _s(PADDING_m, PADDING_m + EXTENT_m);
    gt_out_dev.view(_si, _sj, _sk, _sl, _sm) =
        gt_p1_dev.view(_si, _newaxis, _sk, _sl, _sm) * (
            gt_i_deriv_coeff(0) * stencil<5>(gt_in_dev, {-2, 0, 0, 0, 0}) +
            gt_i_deriv_coeff(1) * stencil<5>(gt_in_dev, {-1, 0, 0, 0, 0}) +
            gt_i_deriv_coeff(2) * stencil<5>(gt_in_dev, { 0, 0, 0, 0, 0}) +
            gt_i_deriv_coeff(3) * stencil<5>(gt_in_dev, {+1, 0, 0, 0, 0}) +
            gt_i_deriv_coeff(4) * stencil<5>(gt_in_dev, {+2, 0, 0, 0, 0})
        ) +
        gt_p2_dev.view(_si, _newaxis, _sk, _sl, _sm) *
          gt_ikj_dev.view(_newaxis, _sj, _newaxis, _newaxis, _newaxis) *
          gt_in_dev.view(_si, _sj, _sk, _sl, _sm);

    // actually compute the result
    gt::synchronize();
  };

  // time the function
  std::cout << "gtensor: " << 1000 * cutime_func(compute_ij_deriv) << " avg ms/computation" << std::endl;

  // copy output data back to host
  auto gt_out = gt::adapt(reinterpret_cast<gt::complex<bElem>*>(out_ptr), shape5D);
  gt::copy(gt_out_dev, gt_out);

  // double check data copied out correctly
  complexArray5D out_arr = (complexArray5D) out_ptr;
  _TILEFOR5D {
    std::complex<bElem> z = gt_out(i, j, k, l, m),
                        w = out_arr[m][l][k][j][i];
    bElem diff = std::abs(z - w);
    if(diff > 1e-6) 
    {
      char errorMsg[1000];
      sprintf(errorMsg, "Result copy failure at (m, l, k, j, i) = (%d, %d, %d, %d, %d)! %f+%f*I != %f+%f*I",
              m, l, k, j, i,
              z.real(), z.imag(), w.real(), w.imag());
      throw std::runtime_error(errorMsg);
    }
  }
}

/**
 * @brief the i-j deriv kernel using hand-written bricks code, and
 *        only bricking in the i-j-k-l directions
 */
void ij_deriv_bricks4D(bComplexElem *out_ptr, bComplexElem *in_ptr,
                       bComplexElem *p1, bComplexElem *p2,
                       bComplexElem ikj[PADDED_EXTENT_j], bElem i_deriv_coeff[5])
{
  // set brick sizes
  constexpr unsigned BDIM_i = 4;
  constexpr unsigned BDIM_j = 4;
  constexpr unsigned BDIM_k = 4;
  constexpr unsigned BDIM_lm = 4;
  // figure out number of bricks in each direction, collapsing
  // lastexpr dimensions into dim 4
  constexpr unsigned BRICK_EXTENT_i = PADDED_EXTENT_i / BDIM_i;
  constexpr unsigned BRICK_EXTENT_j = PADDED_EXTENT_j / BDIM_j;
  constexpr unsigned BRICK_EXTENT_k = PADDED_EXTENT_k / BDIM_k;
  constexpr unsigned BRICK_EXTENT_lm = PADDED_EXTENT_l * PADDED_EXTENT_m / BDIM_lm;

  // set our brick types
  using VFold = Dim<2, 4, 4>;
  using FieldBrick = Brick<Dim<BDIM_i, BDIM_j, BDIM_k, BDIM_lm>, VFold, true>;
  using PreCoeffBrick = Brick<Dim<BDIM_i, BDIM_k, BDIM_lm>, VFold, true>;

  // set up brick info and move to device
  unsigned *field_grid_ptr;
  unsigned *coeff_grid_ptr;

  auto fieldBrickInfo = init_grid<4>(field_grid_ptr, {BRICK_EXTENT_i, BRICK_EXTENT_j, BRICK_EXTENT_k, BRICK_EXTENT_lm});
  auto coeffBrickInfo = init_grid<3>(coeff_grid_ptr, {BRICK_EXTENT_i, BRICK_EXTENT_k, BRICK_EXTENT_lm});
  unsigned *field_grid_ptr_dev;
  unsigned *coeff_grid_ptr_dev;
  {
    unsigned num_field_bricks = BRICK_EXTENT_i * BRICK_EXTENT_j * BRICK_EXTENT_k * BRICK_EXTENT_lm;
    unsigned num_coeff_bricks = BRICK_EXTENT_i * BRICK_EXTENT_k * BRICK_EXTENT_lm;
    cudaMalloc(&field_grid_ptr_dev, num_field_bricks * sizeof(unsigned));
    cudaMalloc(&coeff_grid_ptr_dev, num_coeff_bricks * sizeof(unsigned));
    cudaMemcpy(field_grid_ptr_dev, field_grid_ptr, num_field_bricks * sizeof(unsigned), cudaMemcpyHostToDevice);
    cudaMemcpy(coeff_grid_ptr_dev, coeff_grid_ptr, num_coeff_bricks * sizeof(unsigned), cudaMemcpyHostToDevice);
  }
  BrickInfo<4> *fieldBrickInfo_dev;
  BrickInfo<3> *coeffBrickInfo_dev;
  BrickInfo<4> _fieldBrickInfo_dev = movBrickInfo(fieldBrickInfo, cudaMemcpyHostToDevice);
  BrickInfo<3> _coeffBrickInfo_dev = movBrickInfo(coeffBrickInfo, cudaMemcpyHostToDevice);
  {
    cudaMalloc(&fieldBrickInfo_dev, sizeof(decltype(fieldBrickInfo)));
    cudaMalloc(&coeffBrickInfo_dev, sizeof(decltype(coeffBrickInfo)));
    cudaMemcpy(fieldBrickInfo_dev, &_fieldBrickInfo_dev, sizeof(decltype(fieldBrickInfo)), cudaMemcpyHostToDevice);
    cudaMemcpy(coeffBrickInfo_dev, &_coeffBrickInfo_dev, sizeof(decltype(coeffBrickInfo)), cudaMemcpyHostToDevice);
  }

  // setup brick storage on host
  auto fieldBrickStorage = BrickStorage::allocate(fieldBrickInfo.nbricks, 2 * FieldBrick::BRICKSIZE);
  FieldBrick bIn(&fieldBrickInfo, fieldBrickStorage, 0);
  FieldBrick bOut(&fieldBrickInfo, fieldBrickStorage, FieldBrick::BRICKSIZE);

  auto coeffBrickStorage = BrickStorage::allocate(coeffBrickInfo.nbricks, 2 * PreCoeffBrick::BRICKSIZE);
  PreCoeffBrick bP1(&coeffBrickInfo, coeffBrickStorage, 0);
  PreCoeffBrick bP2(&coeffBrickInfo, coeffBrickStorage, PreCoeffBrick::BRICKSIZE);

  const std::vector<long> fieldDimList = {EXTENT_i, EXTENT_j, EXTENT_k, EXTENT_l * PADDED_EXTENT_m};
  static_assert(PADDING_m == 0);
  const std::vector<long> fieldPadding = {PADDING_i, PADDING_j, PADDING_k, PADDING_l};
  static_assert(GHOST_ZONE_m == 0);
  const std::vector<long> fieldGZ = {GHOST_ZONE_i, GHOST_ZONE_j, GHOST_ZONE_k, GHOST_ZONE_l};
  copyToBrick<4>(fieldDimList, fieldPadding, fieldGZ, in_ptr, field_grid_ptr, bIn);

  const std::vector<long> coeffDimList = {EXTENT_i, EXTENT_k, EXTENT_l * PADDED_EXTENT_m};
  const std::vector<long> coeffPadding = {PADDING_i, PADDING_k, PADDING_l};
  const std::vector<long> coeffGZ = {GHOST_ZONE_i, GHOST_ZONE_k, GHOST_ZONE_l};
  copyToBrick<3>(coeffDimList, coeffPadding, coeffGZ, p1, coeff_grid_ptr, bP1);
  copyToBrick<3>(coeffDimList, coeffPadding, coeffGZ, p2, coeff_grid_ptr, bP2);

  // set up i-k-j and i-deriv coefficients on the device
  bComplexElem *ikj_dev = nullptr;
  bElem *i_deriv_coeff_dev = nullptr;
  {
    cudaMalloc(&ikj_dev, sizeof(decltype(ikj)));
    cudaMalloc(&i_deriv_coeff_dev, sizeof(decltype(i_deriv_coeff)));
    cudaMemcpy(ikj_dev, ikj, sizeof(decltype(ikj)), cudaMemcpyHostToDevice);
    cudaMemcpy(i_deriv_coeff_dev, i_deriv_coeff, sizeof(decltype(i_deriv_coeff)), cudaMemcpyHostToDevice);
  }

  // TODO: build function to actually run computation

  // TODO: time function

  // TODO: copy back to host

  // free allocated memory
  cudaFree(i_deriv_coeff_dev);
  cudaFree(ikj_dev);
  cudaFree(_coeffBrickInfo_dev.adj);
  cudaFree(_fieldBrickInfo_dev.adj);
  cudaFree(coeffBrickInfo_dev);
  cudaFree(fieldBrickInfo_dev);
  cudaFree(coeff_grid_ptr_dev);
  cudaFree(field_grid_ptr_dev);
  free(coeffBrickInfo.adj);
  free(fieldBrickInfo.adj);
  cudaFree(coeff_grid_ptr);
  cudaFree(field_grid_ptr);
}

/**
 * @brief 1-D stencil fused with multiplication by 1-D array
 * 
 * Based on https://github.com/wdmapp/gtensor/blob/d07000b15d253cdeb44942b52f3d2caf4522faa0/benchmarks/ij_deriv.cxx
 */
void ij_deriv() {
  // build in/out arrays
  bComplexElem *in_ptr = randomComplexArray({PADDED_EXTENT}),
               *out_ptr = zeroComplexArray({PADDED_EXTENT});
  // build coefficients needed for stencil computation
  bComplexElem *p1 = randomComplexArray({PADDED_EXTENT_i, PADDED_EXTENT_k, PADDED_EXTENT_l, PADDED_EXTENT_m}),
               *p2 = randomComplexArray({PADDED_EXTENT_i, PADDED_EXTENT_k, PADDED_EXTENT_l, PADDED_EXTENT_m});
  bComplexElem ikj[EXTENT_j];
  for(int j = PADDING_j; j < PADDING_j + EXTENT_j; ++j) ikj[j] = bComplexElem(0, 2 * pi * (j - PADDING_j));
  bElem i_deriv_coeff[5] = {1. / 12., -2. / 3., 0., 2. / 3., -1 / 12.};

  // compute stencil on CPU for correctness check
  std::cout << "Computing correctness check" << std::endl;
  bComplexElem *out_check_ptr = zeroComplexArray({PADDED_EXTENT});
  complexArray5D out_check_arr = (complexArray5D) out_check_ptr;
  complexArray5D in_arr = (complexArray5D) in_ptr;
  coeffArray4D p1_arr = (coeffArray4D) p1;
  coeffArray4D p2_arr = (coeffArray4D) p2;
  _TILEFOR5D out_check_arr[m][l][k][j][i] = p1_arr[m][l][k][i] * (
    i_deriv_coeff[0] * in_arr[m][l][k][j][i - 2] +
    i_deriv_coeff[1] * in_arr[m][l][k][j][i - 1] +
    i_deriv_coeff[2] * in_arr[m][l][k][j][i + 0] +
    i_deriv_coeff[3] * in_arr[m][l][k][j][i + 1] +
    i_deriv_coeff[4] * in_arr[m][l][k][j][i + 2]
  ) + p2_arr[m][l][k][i] * ikj[j] * in_arr[m][l][k][j][i];

  complexArray5D out_arr = (complexArray5D) out_ptr;

  // run computations
  std::cout << "Starting ij_deriv benchmarks" << std::endl;
  ij_deriv_bricks4D(out_ptr, in_ptr, p1, p2, ikj, i_deriv_coeff);
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