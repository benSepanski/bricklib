//
// Created by Benjamin Sepanski on 12/2/21.
//

#ifndef BRICK_GENE6D_GTENSOR_STENCILS_H
#define BRICK_GENE6D_GTENSOR_STENCILS_H

#include "gene-6d-stencils.h"
#define GTENSOR_DEFAULT_DEVICE_ALLOCATOR(T) gt::device_allocator<T>
#include <gtensor/gtensor.h>

/**
 * @brief Return a shifted view of the array
 * copied from
 * https://github.com/wdmapp/gtensor/blob/41cf4fe26625f8d7ba2d0d3886a54ae6415a2017/benchmarks/bench_hypz.cxx#L14-L24
 */
template <int N, typename E> inline auto stencil(E &&e, std::array<int, N> shift) {
  static_assert(N <= RANK);
  constexpr int bnd[RANK] = {(int) GHOST_ZONE[0], (int) GHOST_ZONE[1], (int) GHOST_ZONE[2],
                             (int) GHOST_ZONE[3], (int) GHOST_ZONE[4], (int) GHOST_ZONE[5]};

  std::vector<gt::gdesc> slices;
  slices.reserve(N);
  for (int d = 0; d < N; d++) {
    using namespace gt::placeholders;
    slices.push_back(_s(bnd[d] + shift[d], -bnd[d] + shift[d]));
  }

  return gt::view<N>(std::forward<E>(e), slices);
}

/**
 * @brief Compute the i-j derivative using gtensor (and calls gt::sychronize)
 *
 * \fout := p_1 * \frac{\partial}{\partial x} (in_ptr) + p_2 * ikj * in \f$
 */
template <typename Space>
void ij_deriv_gtensor(gt::gtensor<gt::complex<bElem>, 6, Space> in,
                      gt::gtensor<gt::complex<bElem>, 6, Space> out,
                      gt::gtensor<gt::complex<bElem>, 5, Space> p1,
                      gt::gtensor<gt::complex<bElem>, 6, Space> p2,
                      gt::gtensor<gt::complex<bElem>, 1, Space> ikj,
                      gt::gtensor<bElem, 1, Space> i_deriv_coeff) {
  using namespace gt::placeholders;
  auto _si = _s(GHOST_ZONE[0] + PADDING[0], -GHOST_ZONE[0] - PADDING[0]),
       _sj = _s(GHOST_ZONE[1] + PADDING[1], -GHOST_ZONE[1] - PADDING[1]),
       _sk = _s(GHOST_ZONE[2] + PADDING[2], -GHOST_ZONE[2] - PADDING[2]),
       _sl = _s(GHOST_ZONE[3] + PADDING[3], -GHOST_ZONE[3] - PADDING[3]),
       _sm = _s(GHOST_ZONE[4] + PADDING[4], -GHOST_ZONE[4] - PADDING[4]),
       _sn = _s(GHOST_ZONE[5] + PADDING[5], -GHOST_ZONE[5] - PADDING[5]);
  out.view(_si, _sj, _sk, _sl, _sm, _sn) =
      p1.view(_all, _newaxis, _all, _all, _all, _all) *
          (i_deriv_coeff(0) * stencil<RANK>(in, {-2, 0, 0, 0, 0, 0}) +
           i_deriv_coeff(1) * stencil<RANK>(in, {-1, 0, 0, 0, 0, 0}) +
           i_deriv_coeff(2) * stencil<RANK>(in, {0, 0, 0, 0, 0, 0}) +
           i_deriv_coeff(3) * stencil<RANK>(in, {+1, 0, 0, 0, 0, 0}) +
           i_deriv_coeff(4) * stencil<RANK>(in, {+2, 0, 0, 0, 0, 0})) +
      p2.view(_all, _newaxis, _all, _all, _all, _all) *
          ikj.view(_newaxis, _all, _newaxis, _newaxis, _newaxis, _newaxis) *
          in.view(_si, _sj, _sk, _sl, _sm, _sn);

  // actually compute the result
  gt::synchronize();
}

/**
 * @brief Return a function that compute the k-l 13-point arakawa stencil
 */
template <typename Space>
auto buildArakawaGTensorKernel(const gt::gtensor<gt::complex<bElem>, 6UL, Space> &in,
                               gt::gtensor<gt::complex<bElem>, 6UL, Space> &out,
                               const gt::gtensor<bElem, 6UL, Space> &arakawaCoeff) {
  using namespace gt::placeholders;
  auto _si = _s(GHOST_ZONE[0] + PADDING[0], in.shape(0) - GHOST_ZONE[0] - PADDING[0]),
       _sj = _s(GHOST_ZONE[1] + PADDING[1], in.shape(1) - GHOST_ZONE[1] - PADDING[1]),
       _sk = _s(GHOST_ZONE[2] + PADDING[2], in.shape(2) - GHOST_ZONE[2] - PADDING[2]),
       _sl = _s(GHOST_ZONE[3] + PADDING[3], in.shape(3) - GHOST_ZONE[3] - PADDING[3]),
       _sm = _s(GHOST_ZONE[4] + PADDING[4], in.shape(4) - GHOST_ZONE[4] - PADDING[4]),
       _sn = _s(GHOST_ZONE[5] + PADDING[5], in.shape(5) - GHOST_ZONE[5] - PADDING[5]);

  auto coeff = [&arakawaCoeff](int s) {
    return arakawaCoeff.view(_all, s, _newaxis, _all, _all, _all, _all);
  };
  auto arakawaComputation = [&out, _si, _sj, _sk, _sl, _sm, _sn, coeff, &in]() -> void {
    out.view(_si, _sj, _sk, _sl, _sm, _sn) = coeff(0) * stencil<RANK>(in, {0, 0, +0, -2, 0, 0}) +
                                             coeff(1) * stencil<RANK>(in, {0, 0, -1, -1, 0, 0}) +
                                             coeff(2) * stencil<RANK>(in, {0, 0, +0, -1, 0, 0}) +
                                             coeff(3) * stencil<RANK>(in, {0, 0, +1, -1, 0, 0}) +
                                             coeff(4) * stencil<RANK>(in, {0, 0, -2, +0, 0, 0}) +
                                             coeff(5) * stencil<RANK>(in, {0, 0, -1, +0, 0, 0}) +
                                             coeff(6) * stencil<RANK>(in, {0, 0, +0, +0, 0, 0}) +
                                             coeff(7) * stencil<RANK>(in, {0, 0, +1, +0, 0, 0}) +
                                             coeff(8) * stencil<RANK>(in, {0, 0, +2, +0, 0, 0}) +
                                             coeff(9) * stencil<RANK>(in, {0, 0, -1, +1, 0, 0}) +
                                             coeff(10) * stencil<RANK>(in, {0, 0, +0, +1, 0, 0}) +
                                             coeff(11) * stencil<RANK>(in, {0, 0, +1, +1, 0, 0}) +
                                             coeff(12) * stencil<RANK>(in, {0, 0, +0, +2, 0, 0});

    // actually compute the result
    gt::synchronize();
  };
  return arakawaComputation;
}

#endif // BRICK_GENE6D_GTENSOR_STENCILS_H
