//
// Created by Ben_Sepanski on 11/11/2021.
//

#include "MPIEnvironment.h"

void MPIEnvironment::SetUp() {
  int provided;
  int argc = 0;
  char **argv;
  // setup MPI environment
  check_MPI(MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided));
  if (provided != MPI_THREAD_SERIALIZED) {
    check_MPI(MPI_Finalize());
    ASSERT_EQ(provided, MPI_THREAD_SERIALIZED);
  }
}

void MPIEnvironment::TearDown() {
  check_MPI(MPI_Finalize());
}