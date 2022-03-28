// COPIED FROM src/cuda_exchange.cu in cuda_under_the_hood branch of GENE-dev on Mar 25, 2022
// (commit 0c881a6112a595b9822a10c1b2df154cc6d58f41).
//
// All modifications are marked with
//#include "cuda_exchamnge.h"

//#include "cuda_gene.h"
#include "mpi-gene.cuh"
//#include <cuda_runtime.h>
//#include <gtensor/gtensor.h>
//#include <mpi.h>
//#ifdef GTENSOR_DEVICE_CUDA
//#include <nvToolsExt.h>
//#endif

// last include, so we don't accidentally redef inside vendor headers
//#include "redef.h"

using namespace gt::placeholders;


GTensorExchange build_exchange_v_cuda_6d(gt::shape_type<6> shape, int bndl,
//                                   int bndu, int comm_f, int rank_v,
                                       int bndu, MPI_Comm comm, int rank_v,
                                       int n_procs_v)
{
  if (bndl == 0 && bndu == 0) {
    throw std::runtime_error("Expected bndl != 0 or bndu != 0");
  }

  // C_PERFON(__FUNCTION__,strlen(__FUNCTION__));
//  MPI_Comm comm = MPI_Comm_f2c(comm_f);
//  bndl /= shape[0] * shape[1] * shape[2];
//  bndu /= shape[0] * shape[1] * shape[2];
  assert(bndl == bndu);
  int bnd = bndl;

  int rank_source_p, rank_dest_p;
  int rank_source_m, rank_dest_m;
  MPI_Cart_shift(comm, 2, 1, &rank_source_p, &rank_dest_p);
  MPI_Cart_shift(comm, 2, -1, &rank_source_m, &rank_dest_m);

  auto exchangeV = [rank_source_p, n_procs_v, bnd, comm, rank_source_m, rank_dest_p, rank_dest_m](gtensorComplex6D &toExchange, GeneExchangeBuffers &buf) -> void {
    double st = omp_get_wtime(), ed;
    if (n_procs_v > 1) {
      buf.sbuf_p.view() = toExchange.view(_all, _all, _all, _s(-2 * bnd, -bnd));
      buf.sbuf_m.view() = toExchange.view(_all, _all, _all, _s(bnd, 2 * bnd));

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

#ifndef HAVE_GPU_AWARE_MPI
      copy(buf.h_rbuf_p, buf.rbuf_p);
      copy(buf.h_rbuf_m, buf.rbuf_m);

      // instrument timing
      ed = omp_get_wtime();
      movetime += ed - st;
      st = ed;
#endif

      if (rank_source_p >= 0) {
        toExchange.view(_all, _all, _all, _s(_, bnd)) = buf.rbuf_p;
      }
      if (rank_source_m >= 0) {
        toExchange.view(_all, _all, _all, _s(-bnd, _)) = buf.rbuf_m;
      }
    }

    // Here we would apply the physical boundary conditions as post-process of the
    // exchange. It is only applied to the outer processes.

    gt::synchronize();

    // instrument timing
    ed = omp_get_wtime();
    packtime += ed - st;

    // C_PERFOFF;
  };

  return exchangeV;
}
