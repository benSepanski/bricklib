//
// Created by Ben_Sepanski on 11/11/2021.
//

#include "MPIEnvironment.h"

MPIEnvironment::MPIEnvironment(int argc, char** argv)
    : argc{argc}, argv{argv} {}

void MPIEnvironment::SetUp() {
  int provided;
  // setup MPI environment
  check_MPI(MPI_Init_thread(&this->argc, &this->argv, MPI_THREAD_SERIALIZED, &provided));
  if (provided != MPI_THREAD_SERIALIZED) {
    check_MPI(MPI_Finalize());
    ASSERT_EQ(provided, MPI_THREAD_SERIALIZED);
  }
}

void MPIEnvironment::TearDown() {
  check_MPI(MPI_Finalize());
}