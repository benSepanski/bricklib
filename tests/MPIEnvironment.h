//
// Created by Ben_Sepanski on 11/11/2021.
//

#ifndef BRICK_MPIENVIRONMENT_H
#define BRICK_MPIENVIRONMENT_H

#include "MPILayout.h"
#include <gtest/gtest.h>

/**
 * Used to initialize/teardown MPI for googletests
 */
class MPIEnvironment : public ::testing::Environment {
public:
  void SetUp() override ;
  ~MPIEnvironment() override = default;
  void TearDown() override;
};

#endif // BRICK_MPIENVIRONMENT_H
