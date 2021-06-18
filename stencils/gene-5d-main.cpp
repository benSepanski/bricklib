#include "gene-5d.h"

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
    if(diff > 1e-6) throw std::runtime_error("Result mismatch!");
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
  auto shape5D = gt::shape(PADDED_EXTENT);
  auto shape4D = gt::shape(PADDED_EXTENT_i, PADDED_EXTENT_k, PADDED_EXTENT_l, PADDED_EXTENT_m);
  auto shape_ikj = gt::shape(PADDED_EXTENT_j);
  // adapt in-arrays to gtensor
  auto gt_in = gt::adapt(reinterpret_cast<gt::complex<bElem>*>(in_ptr), shape5D);
  auto gt_p1 = gt::adapt(reinterpret_cast<gt::complex<bElem>*>(p1), shape4D);
  auto gt_p2 = gt::adapt(reinterpret_cast<gt::complex<bElem>*>(p2), shape4D);
  auto gt_ikj = gt::adapt(reinterpret_cast<gt::complex<bElem>*>(ikj), shape_ikj);
  auto gt_i_deriv_coeff = gt::adapt(i_deriv_coeff, gt::shape(5));

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
}

/**
 * @brief Compute the i-j derivative using a tiled GPU algorithm
 */
// void ij_deriv_arr_tile(bComplexElem *out_ptr, bComplexElem *in_ptr,
//                        bComplexElem *p1, bComplexElem *p2,
//                        bComplexElem ikj[PADDED_EXTENT_j], bElem i_deriv_coeff[5])
// {
//   // convert to pointer-to-arrays
//   complexArray5D out_arr = (complexArr5D) out_ptr;
//   complexArray5D in_arr = (complexArr5D) in_ptr;
//   coeffArray4D p1_arr = (coeffArray4D) p1;
//   coeffArray4D p2_arr = (coeffArray4D) p2;
//   dim3 block(NUM_PADDED_ELEMENTS / PADDED_EXTENT_i / PADDED_EXTENT_j, PADDED_EXTENT_j, PADDED_EXTENT_i),
//       thread(BDIM);
//   d3cond_arr << < block, thread >> > (arr_in, arr_out, coeff_dev);
//   // define function which computes the stencil
//   auto compute_ij_deriv[&out_arr, &in_arr, &p1_arr, &p2_arr, &ikj, &i_deriv_coeff] -> void {
//     _TILEFOR5D out_arr[m][l][k][j][i] = p1_arr[m][l][k][i] * (
//       i_deriv_coeff[0] * in_arr[m][l][k][j][i - 2] +
//       i_deriv_coeff[1] * in_arr[m][l][k][j][i - 1] +
//       i_deriv_coeff[2] * in_arr[m][l][k][j][i + 0] +
//       i_deriv_coeff[3] * in_arr[m][l][k][j][i + 1] +
//       i_deriv_coeff[4] * in_arr[m][l][k][j][i + 2]
//     ) + p2_arr[m][l][k][i] * ikj[j] * in_arr[m][l][k][j][i];
//   }

//   // time the function
// }

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
  std::cout << "ij_deriv" << std::endl;
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