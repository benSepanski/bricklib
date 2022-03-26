//
// Created by Benjamin Sepanski on 3/25/22.
//

#ifndef BRICK_MPI_GTENSOR_CUH
#define BRICK_MPI_GTENSOR_CUH


#ifndef CUDA_EXCHANGE_H
#define CUDA_EXCHANGE_H

#ifndef GTENSOR_DEFAULT_DEVICE_ALLOCATOR
#define GTENSOR_DEFAULT_DEVICE_ALLOCATOR(T) gt::device_allocator<T>
#endif
#include <gtensor/gtensor.h>
#include "elementTypes.h"

// Set GENE preprocessor macro for cuda aware
#ifdef CUDA_AWARE
#define HAVE_GPU_AWARE_MPI 1
#endif

// Instead of including cuda_gene.h, use the following:
using real_t = bElem;
using complex_t = bComplexElem;

// Instead of including rdef.h, use the following:
#if bElem==float
#define MPI_REAL_TYPE MPI_DOUBLE_PRECISION
#define MPI_COMPLEX_TYPE MPI_DOUBLE_COMPLEX
#elif bElem==double
#define MPI_REAL_TYPE MPI_SINGLE
#define MPI_COMPLEX_TYPE MPI_COMPLEX
#else
#error "Unexpected bElem type"
#endif

// COPIED FROM src/cuda_exchange.h in cuda_under_the_hood branch of GENE-dev on Mar 25, 2022
// (commit 0c881a6112a595b9822a10c1b2df154cc6d58f41).
//
// All modifications are marked with "MODIFIED"
//#include "cuda_gene.h" MODIFIED

#include <mpi.h>

//#include "redef.h" MODIFIED

namespace {// anonymous namespace  (MODIFIED)
using namespace gt::placeholders;

// (MODIFIED) : deleted mpi_datatype() and exchange_x_cuda()

// ======================================================================
// exchange_z_cuda

template <typename E_u, typename E_pb_phase_fac>
inline void exchange_z_cuda(E_u& u, int bnd, MPI_Comm comm,
                            const E_pb_phase_fac& pb_phase_fac)
{
  static_assert(gt::expr_dimension<E_u>() == 6, "u must be 6-d");
  static_assert(gt::expr_dimension<E_pb_phase_fac>() == 3,
                "pb_phase_fac must be 3-d");

  if (bnd == 0) {
    return;
  }

  int dims[6], periods[6], coords[6];
  MPI_Cart_get(comm, 6, dims, periods, coords);
  __attribute__((unused)) int rank_z = coords[3];  // (MODIFIED WITH __attribute__((unused)))
  int n_procs_z = dims[3];

  // doesn't handle fewer inner points than boundary points
  assert(u.shape(2) >= 3 * bnd);

  if (n_procs_z > 1) {
#if 0
    int rank_source, rank_dest;
    auto sbuf = gt::eval(u.view(_all, _all, _s(-2 * bnd, -bnd)));
    gt::synchronize();
    auto rbuf = gt::empty_like(sbuf);
    MPI_Cart_shift(comm, 3, 1, &rank_source, &rank_dest);
    MPI_Sendrecv(sbuf.data().get(), sbuf.size(), MPI_COMPLEX_TYPE, rank_dest,
                 123, rbuf.data().get(), rbuf.size(), MPI_COMPLEX_TYPE,
                 rank_source, 123, comm, MPI_STATUS_IGNORE);
    if (rank_source >= 0) {
      u.view(_all, _all, _s(_, bnd)) = rbuf;
      gt::synchronize();
    }

    sbuf = u.view(_all, _all, _s(bnd, 2 * bnd));
    gt::synchronize();
    MPI_Cart_shift(comm, 3, -1, &rank_source, &rank_dest);
    MPI_Sendrecv(sbuf.data().get(), sbuf.size(), MPI_COMPLEX_TYPE, rank_dest,
                 123, rbuf.data().get(), rbuf.size(), MPI_COMPLEX_TYPE,
                 rank_source, 123, comm, MPI_STATUS_IGNORE);
    if (rank_source >= 0) {
      u.view(_all, _all, _s(-bnd, _)) = rbuf;
      gt::synchronize();
    }
#else
    MPI_Request reqs[4] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL,
                           MPI_REQUEST_NULL};

    auto sbuf_p = gt::eval(u.view(_all, _all, _s(-2 * bnd, -bnd)));
    auto sbuf_m = gt::eval(u.view(_all, _all, _s(bnd, 2 * bnd)));
    auto rbuf_p = gt::empty_like(sbuf_p);
    auto rbuf_m = gt::empty_like(sbuf_m);
#ifndef HAVE_GPU_AWARE_MPI
    gt::gtensor<complex_t, 6> h_sbuf_p(sbuf_p.shape());
    gt::gtensor<complex_t, 6> h_sbuf_m(sbuf_m.shape());
    gt::gtensor<complex_t, 6> h_rbuf_p(rbuf_p.shape());
    gt::gtensor<complex_t, 6> h_rbuf_m(rbuf_m.shape());
#endif

    int rank_source_p, rank_dest_p;
    int rank_source_m, rank_dest_m;
    MPI_Cart_shift(comm, 3, 1, &rank_source_p, &rank_dest_p);
    MPI_Cart_shift(comm, 3, -1, &rank_source_m, &rank_dest_m);

    if (rank_source_p >= 0) {
#ifdef HAVE_GPU_AWARE_MPI
      MPI_Irecv(rbuf_p.data().get(), rbuf_p.size(), MPI_COMPLEX_TYPE,
                rank_source_p, 123, comm, &reqs[0]);
#else
      MPI_Irecv(h_rbuf_p.data(), h_rbuf_p.size(), MPI_COMPLEX_TYPE,
                rank_source_p, 123, comm, &reqs[0]);
#endif
    }
    if (rank_source_m >= 0) {
#ifdef HAVE_GPU_AWARE_MPI
      MPI_Irecv(rbuf_m.data().get(), rbuf_m.size(), MPI_COMPLEX_TYPE,
                rank_source_m, 456, comm, &reqs[2]);
#else
      MPI_Irecv(h_rbuf_m.data(), h_rbuf_m.size(), MPI_COMPLEX_TYPE,
                rank_source_m, 456, comm, &reqs[2]);
#endif
    }

    gt::synchronize();

    if (rank_dest_p >= 0) {
#ifdef HAVE_GPU_AWARE_MPI
      MPI_Isend(sbuf_p.data().get(), sbuf_p.size(), MPI_COMPLEX_TYPE,
                rank_dest_p, 123, comm, &reqs[1]);
#else
      copy(sbuf_p, h_sbuf_p);
      MPI_Isend(h_sbuf_p.data(), h_sbuf_p.size(), MPI_COMPLEX_TYPE, rank_dest_p,
                123, comm, &reqs[1]);
#endif
    }
    if (rank_dest_m >= 0) {
#ifdef HAVE_GPU_AWARE_MPI
      MPI_Isend(sbuf_m.data().get(), sbuf_m.size(), MPI_COMPLEX_TYPE,
                rank_dest_m, 456, comm, &reqs[3]);
#else
      copy(sbuf_m, h_sbuf_m);
      MPI_Isend(h_sbuf_m.data(), h_sbuf_m.size(), MPI_COMPLEX_TYPE, rank_dest_m,
                456, comm, &reqs[3]);
#endif
    }

    MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

    if (rank_source_p >= 0) {
#ifndef HAVE_GPU_AWARE_MPI
      copy(h_rbuf_p, rbuf_p);
#endif
      u.view(_all, _all, _s(_, bnd)) = rbuf_p;
    }
    if (rank_source_m >= 0) {
#ifndef HAVE_GPU_AWARE_MPI
      copy(h_rbuf_m, rbuf_m);
#endif
      u.view(_all, _all, _s(-bnd, _)) = rbuf_m;
    }
#endif
  } else {
    // periodic ghost points on single proc
    u.view(_all, _all, _s(_, bnd)) = u.view(_all, _all, _s(-2 * bnd, -bnd));
    u.view(_all, _all, _s(-bnd, _)) = u.view(_all, _all, _s(bnd, 2 * bnd));
    gt::synchronize();
  }

  // Here we apply the physical boundary conditions as post-process of the
  // exchange. It is only applied to the outer processes.
  // MODIFIED: BRICKS CURRENTLY DOES NOT APPLY THE BCs
  // TODO: APPLY BCs AFTER EACH EXCHANGE
//  if (rank_z == 0) {
//    u.view(_all, _all, _s(_, bnd)) =
//        u.view(_all, _all, _s(_, bnd)) *
//        pb_phase_fac.view(_all, _all, 1, _newaxis, _newaxis, _newaxis, _newaxis);
//  }
//
//  if (rank_z == n_procs_z - 1) {
//    u.view(_all, _all, _s(-bnd, _)) =
//        u.view(_all, _all, _s(-bnd, _)) *
//        pb_phase_fac.view(_all, _all, 0, _newaxis, _newaxis, _newaxis, _newaxis);
//  }

  gt::synchronize();
}

#endif
} // end anonymous namespace (MODIFIED)

// END OF COPY

void exchange_v_cuda_6d(complex_t* _u, const int* u_shape, int bndl,
                        int bndu, MPI_Comm comm, int rank_v,
                        int n_procs_v);

// Expose exchange function to viewer
template<typename PhaseFacArrayType>
void geneExchangeZV(gt::gtensor_span<gt::complex<bElem>, 6, gt::space::device> &toExchange,
                    MPI_Comm &comm,
                    unsigned bnd, PhaseFacArrayType &pbPhaseFac,
                    int rank_v,
                    int n_procs_v) {
  // See https://gitlab.mpcdf.mpg.de/GENE/gene-dev/-/blob/0c881a6112a595b9822a10c1b2df154cc6d58f41/src/f_computer.F90#L484-486
  // for an example
  exchange_z_cuda(toExchange, bnd, comm, pbPhaseFac);
  exchange_v_cuda_6d(reinterpret_cast<complex_t*>(toExchange.data().get()),
                     toExchange.shape().data(),
                     bnd, bnd, comm, rank_v, n_procs_v);
}

#endif // BRICK_MPI_GTENSOR_CUH
