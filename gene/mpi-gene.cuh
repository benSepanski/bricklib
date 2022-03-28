//
// Created by Benjamin Sepanski on 3/25/22.
//

#ifndef BRICK_MPI_GENE_CUH
#define BRICK_MPI_GENE_CUH

#ifndef CUDA_EXCHANGE_H
#define CUDA_EXCHANGE_H

#ifndef GTENSOR_DEFAULT_DEVICE_ALLOCATOR
#define GTENSOR_DEFAULT_DEVICE_ALLOCATOR(T) gt::device_allocator<T>
#endif
#include <gtensor/gtensor.h>
#include "elementTypes.h"
#include "MPIHandle.h"

// Set GENE preprocessor macro for cuda aware
#ifdef CUDA_AWARE
#define HAVE_GPU_AWARE_MPI 1
#endif

// Instead of including cuda_gene.h, use the following:
using real_t = bElem;
using complex_t = gt::complex<bElem>;

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

// COPIED and modified FROM src/cuda_exchange.h in cuda_under_the_hood branch of GENE-dev on Mar 25, 2022
// (commit 0c881a6112a595b9822a10c1b2df154cc6d58f41).
//
//#include "cuda_gene.h"

#include <mpi.h>

//#include "redef.h"

namespace {// anonymous namespace
using namespace gt::placeholders;

// ======================================================================
// exchange_z_cuda

struct GeneExchangeBuffers {
private:
  static gt::shape_type<6> getBufferShape(gt::shape_type<6> shape, int bnd, int axis) {
    if(axis < 0 || axis >= 6) {
      throw std::runtime_error("axis must be in [0,5]");
    }
    shape[axis] = bnd;
    return shape;
  }

public:
  gt::gtensor<complex_t, 6, gt::space::device> sbuf_p,sbuf_m, rbuf_p, rbuf_m;
#ifndef HAVE_GPU_AWARE_MPI
  gt::gtensor<complex_t, 6> h_sbuf_p;
  gt::gtensor<complex_t, 6> h_sbuf_m;
  gt::gtensor<complex_t, 6> h_rbuf_p;
  gt::gtensor<complex_t, 6> h_rbuf_m;
#endif

  GeneExchangeBuffers(gt::shape_type<6> shape, int bnd, int axis)
  : sbuf_p{gt::empty_device<complex_t>(getBufferShape(shape, bnd, axis))}
  , sbuf_m{gt::empty_like(sbuf_p)}
  , rbuf_p{gt::empty_like(sbuf_p)}
  , rbuf_m{gt::empty_like(sbuf_p)}
#ifndef HAVE_GPU_AWARE_MPI
  , h_sbuf_p{sbuf_p.shape()}
  , h_sbuf_m{sbuf_m.shape()}
  , h_rbuf_p{rbuf_p.shape()}
  , h_rbuf_m{rbuf_m.shape()}
#endif
  {}
};
typedef gt::gtensor_span<gt::complex<bElem>, 6, gt::space::device> gtensorComplex6D;
typedef std::function<void(gtensorComplex6D &, GeneExchangeBuffers&)> GTensorExchange;

template <typename E_pb_phase_fac>
inline GTensorExchange exchange_z_cuda(gt::shape_type<6> shape, int bnd, MPI_Comm comm,
                            const E_pb_phase_fac& pb_phase_fac)
{
  static_assert(gt::expr_dimension<E_pb_phase_fac>() == 3,
                "pb_phase_fac must be 3-d");

  if (bnd == 0) {
    throw std::runtime_error("exchange_z_cuda: bnd must be non-zero");
  }

  int dims[6], periods[6], coords[6];
  MPI_Cart_get(comm, 6, dims, periods, coords);
  __attribute__((unused)) int rank_z = coords[3];
  int n_procs_z = dims[3];

  // doesn't handle fewer inner points than boundary points
  assert(shape[2] >= 3 * bnd);

  int rank_source_p, rank_dest_p;
  int rank_source_m, rank_dest_m;

  if (n_procs_z > 1) {
    MPI_Cart_shift(comm, 3, 1, &rank_source_p, &rank_dest_p);
    MPI_Cart_shift(comm, 3, -1, &rank_source_m, &rank_dest_m);
  }

  auto doZExchange = [n_procs_z, rank_source_p, comm, rank_source_m, rank_dest_m, bnd, rank_dest_p](gtensorComplex6D &toExchange, GeneExchangeBuffers &buf) -> void {
    double st = omp_get_wtime(), ed;
    if(n_procs_z > 1) {
      buf.sbuf_p = toExchange.view(_all, _all, _s(-2 * bnd, -bnd));
      buf.sbuf_m = toExchange.view(_all, _all, _s(bnd, 2 * bnd));
      // instrument timing
      ed = omp_get_wtime();
      packtime += ed - st;
      st = ed;

      MPI_Request reqs[4] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL,
                             MPI_REQUEST_NULL};
      if (rank_source_p >= 0) {
#ifdef HAVE_GPU_AWARE_MPI
        MPI_Irecv(buf.rbuf_p.data().get(), buf.rbuf_p.size(), MPI_COMPLEX_TYPE,
                  rank_source_p, 123, comm, &reqs[0]);
#else
        MPI_Irecv(buf.h_rbuf_p.data(), buf.h_rbuf_p.size(), MPI_COMPLEX_TYPE,
                  rank_source_p, 123, comm, &reqs[0]);
#endif
      }
      if (rank_source_m >= 0) {
#ifdef HAVE_GPU_AWARE_MPI
        MPI_Irecv(buf.rbuf_m.data().get(), buf.rbuf_m.size(), MPI_COMPLEX_TYPE,
                  rank_source_m, 456, comm, &reqs[2]);
#else
        MPI_Irecv(buf.h_rbuf_m.data(), buf.h_rbuf_m.size(), MPI_COMPLEX_TYPE,
                  rank_source_m, 456, comm, &reqs[2]);
#endif
      }
      // instrument timing
      ed = omp_get_wtime();
      calltime += ed - st;
      st = ed;

      gt::synchronize();

      // instrument timing
      ed = omp_get_wtime();
      packtime += ed - st;
      st = ed;

      if (rank_dest_p >= 0) {
#ifdef HAVE_GPU_AWARE_MPI
        MPI_Isend(buf.sbuf_p.data().get(), buf.sbuf_p.size(), MPI_COMPLEX_TYPE,
                  rank_dest_p, 123, comm, &reqs[1]);
#else
        copy(buf.sbuf_p, buf.h_sbuf_p);
        // instrument timing
        ed = omp_get_wtime();
        movetime += ed - st;
        st = ed;

        MPI_Isend(buf.h_sbuf_p.data(), buf.h_sbuf_p.size(), MPI_COMPLEX_TYPE, rank_dest_p,
                  123, comm, &reqs[1]);
#endif
      }
      // instrument timing
      ed = omp_get_wtime();
      calltime += ed - st;
      st = ed;

      if (rank_dest_m >= 0) {
#ifdef HAVE_GPU_AWARE_MPI
        MPI_Isend(buf.sbuf_m.data().get(), buf.sbuf_m.size(), MPI_COMPLEX_TYPE,
                  rank_dest_m, 456, comm, &reqs[3]);
#else
        copy(buf.sbuf_m, buf.h_sbuf_m);
        // instrument timing
        ed = omp_get_wtime();
        movetime += ed - st;
        st = ed;

        MPI_Isend(buf.h_sbuf_m.data(), buf.h_sbuf_m.size(), MPI_COMPLEX_TYPE, rank_dest_m,
                  456, comm, &reqs[3]);
#endif
      }
      // instrument timing
      ed = omp_get_wtime();
      calltime += ed - st;
      st = ed;

      check_MPI(MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE));

      // instrument timing
      ed = omp_get_wtime();
      waittime += ed - st;
      st = ed;

      if (rank_source_p >= 0) {
#ifndef HAVE_GPU_AWARE_MPI
        copy(buf.h_rbuf_p,buf.rbuf_p);
        // instrument timing
        ed = omp_get_wtime();
        movetime += ed - st;
        st = ed;
#endif
        toExchange.view(_all, _all, _s(_, bnd)) = buf.rbuf_p;
      }
      if (rank_source_m >= 0) {
#ifndef HAVE_GPU_AWARE_MPI
        copy(buf.h_rbuf_m, buf.rbuf_m);
        // instrument timing
        ed = omp_get_wtime();
        movetime += ed - st;
        st = ed;
#endif
        toExchange.view(_all, _all, _s(-bnd, _)) = buf.rbuf_m;
      }
#endif
    } else {
      // periodic ghost points on single proc
      toExchange.view(_all, _all, _s(_, bnd)) = toExchange.view(_all, _all, _s(-2 * bnd, -bnd));
      toExchange.view(_all, _all, _s(-bnd, _)) = toExchange.view(_all, _all, _s(bnd, 2 * bnd));
      gt::synchronize();
    }

    // Here we apply the physical boundary conditions as post-process of the
    // exchange. It is only applied to the outer processes.
    // BRICKS CURRENTLY DOES NOT APPLY THE BCs
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

    // instrument timing
    ed = omp_get_wtime();
    packtime += ed - st;
  };

  return doZExchange;
}

} // end anonymous namespace

// END OF COPY

GTensorExchange build_exchange_v_cuda_6d(gt::shape_type<6> shape, int bndl, int bndu,
                                         MPI_Comm comm, int rank_v, int n_procs_v);

// Expose exchange function to viewer
template<typename PhaseFacArrayType>
auto buildGeneExchangeZV(gt::shape_type<6> &shape,
                         MPI_Comm &comm,
                         unsigned bnd, PhaseFacArrayType &pbPhaseFac) {
  int rankV, numProcsV;
  std::array<int, 6> mpiCoords{}, mpiSize{}, mpiPeriodic{};
  check_MPI(MPI_Cart_get(comm, 6, mpiSize.data(), mpiPeriodic.data(), mpiCoords.data()));
  int vAxis = 2; // MPI Stores axes in reverse order {0,1,2,3(v),4,5} -> {5,4,3,2(v),1,0}
  rankV = mpiCoords[vAxis];
  numProcsV = mpiSize[vAxis];

  // See https://gitlab.mpcdf.mpg.de/GENE/gene-dev/-/blob/0c881a6112a595b9822a10c1b2df154cc6d58f41/src/f_computer.F90#L484-486
  // for an example
  auto exchangeZ = exchange_z_cuda(shape, bnd, comm, pbPhaseFac);
  auto exchangeV = build_exchange_v_cuda_6d(shape, bnd, bnd, comm, rankV, numProcsV);
  return [=](gt::gtensor_span<gt::complex<bElem>, 6, gt::space::device> &toExchange, GeneExchangeBuffers &bufZ, GeneExchangeBuffers &bufV) -> void {
    exchangeZ(toExchange, bufZ);
    exchangeV(toExchange, bufV);
  };
}

#endif // BRICK_MPI_GENE_CUH
