// COPIED FROM src/cuda_exchange.cu in cuda_under_the_hood branch of GENE-dev on Mar 25, 2022
// (commit 0c881a6112a595b9822a10c1b2df154cc6d58f41).
//
// All modifications are marked with "MODIFIED"
//#include "cuda_exchange.h" (MODIFIED)

//#include "cuda_gene.h" (MODIFIED)
#include "mpi-gtensor.cuh" // (MODIFIED)
//#include <cuda_runtime.h>
//#include <gtensor/gtensor.h> (MODIFIED)
//#include <mpi.h>             (MODIFIED)
//#ifdef GTENSOR_DEVICE_CUDA   (MODIFIED)
//#include <nvToolsExt.h>      (MODIFIED)
//#endif                       (MODIFIED)

// last include, so we don't accidentally redef inside vendor headers
//#include "redef.h"  (MODIFIED)

using namespace gt::placeholders;

// (MODIFIED): Deleted exchange_x_cuda_general(...)

// ======================================================================
// exchange_z_cuda_3d

// (MODIFIED: Don't need to use fortran-to-C conversion)
extern "C" void exchange_z_cuda_3d(complex_t* _u, const int* u_shape, int bndl,
//                                   int bndu, int comm_f, int rank_z,
                                   int bndu, MPI_Comm comm, int rank_z,
                                   int n_procs_z, complex_t* _pb_phase_fac,
                                   const int* pb_phase_fac_shape)
{
  auto u3 = gt::adapt_device<3>(_u, u_shape);
  auto u = u3.view(_all, _all, _all, _newaxis, _newaxis, _newaxis);
  auto pb_phase_fac = gt::adapt_device<3>(_pb_phase_fac, pb_phase_fac_shape);
//  MPI_Comm comm = MPI_Comm_f2c(comm_f); (MODIFIED: Don't need to use fortran-to-C conversion)

  bndl /= u.shape(0) * u.shape(1);
  bndu /= u.shape(0) * u.shape(1);
  assert(bndl == bndu);

  exchange_z_cuda(u, bndl, comm, pb_phase_fac);
}

// ======================================================================
// exchange_z_cuda_4d

// (MODIFIED: Don't need to use fortran-to-C conversion)
extern "C" void exchange_z_cuda_4d(complex_t* _u, const int* u_shape, int bndl,
//                                   int bndu, int comm_f, int rank_z,
                                   int bndu, MPI_Comm comm, int rank_z,
                                   int n_procs_z, complex_t* _pb_phase_fac,
                                   const int* pb_phase_fac_shape)
{
  auto u4 = gt::adapt_device<4>(_u, u_shape);
  auto u = u4.view(_all, _all, _all, _all, _newaxis, _newaxis);
  auto pb_phase_fac = gt::adapt_device<3>(_pb_phase_fac, pb_phase_fac_shape);
//  MPI_Comm comm = MPI_Comm_f2c(comm_f); (MODIFIED: Don't need to use fortran-to-C conversion)

  bndl /= u.shape(0) * u.shape(1);
  bndu /= u.shape(0) * u.shape(1);
  assert(bndl == bndu);

  exchange_z_cuda(u, bndl, comm, pb_phase_fac);
}

// ======================================================================
// exchange_z_cuda_5d

// (MODIFIED: Don't need to use fortran-to-C conversion)
extern "C" void exchange_z_cuda_5d(complex_t* _u, const int* u_shape, int bndl,
//                                   int bndu, int comm_f, int rank_z,
                                   int bndu, MPI_Comm comm, int rank_z,
                                   int n_procs_z, complex_t* _pb_phase_fac,
                                   const int* pb_phase_fac_shape)
{
  auto u5 = gt::adapt_device<5>(_u, u_shape);
  auto u = u5.view(_all, _all, _all, _all, _all, _newaxis);
  auto pb_phase_fac = gt::adapt_device<3>(_pb_phase_fac, pb_phase_fac_shape);
//  MPI_Comm comm = MPI_Comm_f2c(comm_f); (MODIFIED: Don't need to use fortran-to-C conversion)

  bndl /= u.shape(0) * u.shape(1);
  bndu /= u.shape(0) * u.shape(1);
  assert(bndl == bndu);

  exchange_z_cuda(u, bndl, comm, pb_phase_fac);
}

// ======================================================================
// exchange_z_cuda_6d

// (MODIFIED: Don't need to use fortran-to-C conversion)
extern "C" void exchange_z_cuda_6d(complex_t* _u, const int* u_shape, int bndl,
//                                   int bndu, int comm_f, int rank_z,
                                   int bndu, MPI_Comm comm, int rank_z,
                                   int n_procs_z, complex_t* _pb_phase_fac,
                                   const int* pb_phase_fac_shape)
{
  auto u = gt::adapt_device<6>(_u, u_shape);
  auto pb_phase_fac = gt::adapt_device<3>(_pb_phase_fac, pb_phase_fac_shape);
//  MPI_Comm comm = MPI_Comm_f2c(comm_f); (MODIFIED: Don't need to use fortran-to-C conversion)

  bndl /= u.shape(0) * u.shape(1);
  bndu /= u.shape(0) * u.shape(1);
  assert(bndl == bndu);

  exchange_z_cuda(u, bndl, comm, pb_phase_fac);
}

// (MODIFIED: Don't need to use fortran-to-C conversion)
// (MODIFIED: Deleted extern "C")
void exchange_v_cuda_6d(complex_t* _u, const int* u_shape, int bndl,
//                                   int bndu, int comm_f, int rank_v,
                                   int bndu, MPI_Comm comm, int rank_v,
                                   int n_procs_v)
{
  if (bndl == 0 && bndu == 0) {
    return;
  }

  // C_PERFON(__FUNCTION__,strlen(__FUNCTION__));
  auto u = gt::adapt_device<6>(_u, u_shape);
//  MPI_Comm comm = MPI_Comm_f2c(comm_f); (MODIFIED: Don't need to use fortran-to-C conversion)
  bndl /= u.shape(0) * u.shape(1) * u.shape(2);
  bndu /= u.shape(0) * u.shape(1) * u.shape(2);
  assert(bndl == bndu);
  int bnd = bndl;

  if (n_procs_v > 1) {
#if 0
    int rank_source, rank_dest;
    auto sbuf = gt::eval(u.view(_all, _all, _all, _s(-2 * bnd, -bnd)));
    auto rbuf = gt::empty_like(sbuf);
    MPI_Cart_shift(comm, 2, 1, &rank_source, &rank_dest);
    gt::synchronize();
    MPI_Sendrecv(sbuf.data().get(), sbuf.size(), MPI_COMPLEX_TYPE, rank_dest,
                 123, rbuf.data().get(), rbuf.size(), MPI_COMPLEX_TYPE,
                 rank_source, 123, comm, MPI_STATUS_IGNORE);
    if (rank_source >= 0) {
      u.view(_all, _all, _all, _s(_, bnd)) = rbuf;
      gt::synchronize();
    }

    sbuf = u.view(_all, _all, _all, _s(bnd, 2 * bnd));
    MPI_Cart_shift(comm, 2, -1, &rank_source, &rank_dest);
    gt::synchronize();
    MPI_Sendrecv(sbuf.data().get(), sbuf.size(), MPI_COMPLEX_TYPE, rank_dest,
                 123, rbuf.data().get(), rbuf.size(), MPI_COMPLEX_TYPE,
                 rank_source, 123, comm, MPI_STATUS_IGNORE);
    if (rank_source >= 0) {
      u.view(_all, _all, _all, _s(-bnd, _)) = rbuf;
      gt::synchronize();
    }
#else
    MPI_Request reqs[4] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL,
                           MPI_REQUEST_NULL};

    auto sbuf_p = gt::eval(u.view(_all, _all, _all, _s(-2 * bnd, -bnd)));
    auto sbuf_m = gt::eval(u.view(_all, _all, _all, _s(bnd, 2 * bnd)));
    auto rbuf_p = gt::empty_like(sbuf_p);
    auto rbuf_m = gt::empty_like(sbuf_m);

    int rank_source_p, rank_dest_p;
    int rank_source_m, rank_dest_m;
    MPI_Cart_shift(comm, 2, 1, &rank_source_p, &rank_dest_p);
    MPI_Cart_shift(comm, 2, -1, &rank_source_m, &rank_dest_m);

#ifndef HAVE_GPU_AWARE_MPI
    gt::gtensor<complex_t, 6> h_rbuf_p(rbuf_p.shape());
    gt::gtensor<complex_t, 6> h_rbuf_m(rbuf_m.shape());
    gt::gtensor<complex_t, 6> h_sbuf_p(sbuf_p.shape());
    gt::gtensor<complex_t, 6> h_sbuf_m(sbuf_m.shape());
#endif
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
#ifndef HAVE_GPU_AWARE_MPI
    copy(h_rbuf_p, rbuf_p);
    copy(h_rbuf_m, rbuf_m);
#endif

    if (rank_source_p >= 0) {
      u.view(_all, _all, _all, _s(_, bnd)) = rbuf_p;
    }
    if (rank_source_m >= 0) {
      u.view(_all, _all, _all, _s(-bnd, _)) = rbuf_m;
    }
#endif
  }

  // Here we would apply the physical boundary conditions as post-process of the
  // exchange. It is only applied to the outer processes.

  gt::synchronize();
  // C_PERFOFF;
}

// (MODIFIED: Don't need to use fortran-to-C conversion)
extern "C" void exchange_mu_cuda_6d(complex_t* _u, const int* u_shape, int bndl,
//                                    int bndu, int comm_f, int rank_w,
                                    int bndu, MPI_Comm comm, int rank_w,
                                    int n_procs_w)
{
  if (bndl == 0 && bndu == 0) {
    return;
  }

  // C_PERFON(__FUNCTION__,strlen(__FUNCTION__));
  auto u = gt::adapt_device<6>(_u, u_shape);
//  MPI_Comm comm = MPI_Comm_f2c(comm_f);
  bndl /= u.shape(0) * u.shape(1) * u.shape(2) * u.shape(3);
  bndu /= u.shape(0) * u.shape(1) * u.shape(2) * u.shape(3);
  assert(bndl == bndu);
  int bnd = bndl;

  if (n_procs_w > 1) {
#if 0
    int rank_source, rank_dest;
    auto sbuf = gt::eval(u.view(_all, _all, _all, _all, _s(-2 * bnd, -bnd)));
    auto rbuf = gt::empty_like(sbuf);
    MPI_Cart_shift(comm, 1, 1, &rank_source, &rank_dest);
    gt::synchronize();
    MPI_Sendrecv(sbuf.data().get(), sbuf.size(), MPI_COMPLEX_TYPE, rank_dest,
                 123, rbuf.data().get(), rbuf.size(), MPI_COMPLEX_TYPE,
                 rank_source, 123, comm, MPI_STATUS_IGNORE);
    if (rank_source >= 0) {
      u.view(_all, _all, _all, _all, _s(_, bnd)) = rbuf;
      gt::synchronize();
    }

    sbuf = u.view(_all, _all, _all, _all, _s(bnd, 2 * bnd));
    MPI_Cart_shift(comm, 1, -1, &rank_source, &rank_dest);
    gt::synchronize();
    MPI_Sendrecv(sbuf.data().get(), sbuf.size(), MPI_COMPLEX_TYPE, rank_dest,
                 123, rbuf.data().get(), rbuf.size(), MPI_COMPLEX_TYPE,
                 rank_source, 123, comm, MPI_STATUS_IGNORE);
    if (rank_source >= 0) {
      u.view(_all, _all, _all, _all, _s(-bnd, _)) = rbuf;
      gt::synchronize();
    }
#else
    MPI_Request reqs[4] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL,
                           MPI_REQUEST_NULL};

    auto sbuf_p = gt::eval(u.view(_all, _all, _all, _all, _s(-2 * bnd, -bnd)));
    auto sbuf_m = gt::eval(u.view(_all, _all, _all, _all, _s(bnd, 2 * bnd)));
    auto rbuf_p = gt::empty_like(sbuf_p);
    auto rbuf_m = gt::empty_like(sbuf_m);

    int rank_source_p, rank_dest_p;
    int rank_source_m, rank_dest_m;
    MPI_Cart_shift(comm, 1, 1, &rank_source_p, &rank_dest_p);
    MPI_Cart_shift(comm, 1, -1, &rank_source_m, &rank_dest_m);

#ifndef HAVE_GPU_AWARE_MPI
    gt::gtensor<complex_t, 6> h_rbuf_p(rbuf_p.shape());
    gt::gtensor<complex_t, 6> h_rbuf_m(rbuf_m.shape());
    gt::gtensor<complex_t, 6> h_sbuf_p(sbuf_p.shape());
    gt::gtensor<complex_t, 6> h_sbuf_m(sbuf_m.shape());
#endif
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
#ifndef HAVE_GPU_AWARE_MPI
    copy(h_rbuf_p, rbuf_p);
    copy(h_rbuf_m, rbuf_m);
#endif

    if (rank_source_p >= 0) {
      u.view(_all, _all, _all, _all, _s(_, bnd)) = rbuf_p;
    }
    if (rank_source_m >= 0) {
      u.view(_all, _all, _all, _all, _s(-bnd, _)) = rbuf_m;
    }
#endif
  }

  // Here we would apply the physical boundary conditions as post-process of the
  // exchange. It is only applied to the outer processes.

  gt::synchronize();
  // C_PERFOFF;
}