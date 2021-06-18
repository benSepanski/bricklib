#include "gene-5d.h"

template<typename Space>
using gtensor5D = gt::gtensor<gt::complex<bElem>, 5, Space>;

void ij_deriv_gtensor(bComplexElem *in_ptr, bComplexElem *out_ptr)
{
  using namespace gt::placeholders;

  auto shape = gt::shape(PADDED_EXTENT);
  // copy in-array to gtensor
  auto gt_in = gt::empty<gt::complex<bElem> >(shape);
  complexArray5D in_arr = (complexArray5D) in_ptr;
  _TILEFOR5D { gt_in(i, j, k, l, m) = reinterpret_cast<const gt::complex<bElem>&>(in_arr[m][l][k][j][i]); }

  // copy the in-array to device
  auto gt_in_dev = gt::empty_device<gt::complex<bElem> >(shape);
  gt::copy(gt_in, gt_in_dev);

  // compute our stencil
  gtensor5D<gt::space::device> gt_out_dev(shape);
  gt_out_dev.view(_all, _all, _all, _all, _all) = 2.0 * gt_in_dev.view(_all, _all, _all, _all, _all);
  gt::synchronize();

  // copy output data back to host
  auto gt_out = gt::empty_like(gt_in);
  gt::copy(gt_out_dev, gt_out);

  // copy host to out_ptr
  complexArray5D out_arr = (complexArray5D) out_ptr;
  _TILEFOR5D out_arr[m][l][k][j][i] = reinterpret_cast<const bComplexElem&>(gt_out(i, j, k, l, m));
}

/**
 * @brief 1-D stencil fused with multiplication by 1-D array
 * 
 * Based on https://github.com/wdmapp/gtensor/blob/d07000b15d253cdeb44942b52f3d2caf4522faa0/benchmarks/ij_deriv.cxx
 */
void ij_deriv() {
    // build in/out arrays
    bComplexElem *in_ptr = randomComplexArray({PADDED_EXTENT});
    bComplexElem *out_ptr = zeroComplexArray({PADDED_EXTENT});

    // run computations
    std::cout << "ij_deriv" << std::endl;
    ij_deriv_gtensor(in_ptr, out_ptr);
    std::cout << "done" << std::endl;

    free(out_ptr);
    free(in_ptr);
}

int main() {
  // std::random_device r;
  // std::mt19937_64 mt(r());
  // std::uniform_real_distribution<bElem> u(0, 1);
  ij_deriv();
  return 0;
}