//
// Created by Tuowen Zhao on 12/5/18.
//

#ifndef BRICK_STENCILS_CU_H
#define BRICK_STENCILS_CU_H

#include <brick-cuda.h>
#include "stencils.h"

#define CU_WARMUP 5
#define CU_ITER 100

template<typename T>
double cutime_func(T func, unsigned cu_warmup = CU_WARMUP, unsigned cu_iter = CU_ITER) {
  for(int i = 0; i < cu_warmup; ++i) func(); // Warm up
  cudaEvent_t start, stop;
  float elapsed = 0.0;
  cudaCheck(cudaDeviceSynchronize());
  cudaCheck(cudaEventCreate(&start));
  cudaCheck(cudaEventCreate(&stop));
  cudaCheck(cudaEventRecord(start));
  for (int i = 0; i < cu_iter; ++i)
    func();
  cudaCheck(cudaEventRecord(stop));
  cudaCheck(cudaEventSynchronize(stop));
  cudaCheck(cudaEventElapsedTime(&elapsed, start, stop));
  return elapsed / cu_iter / 1000;
}

void d3pt7cu();

void d3pt7complexcu();

void d3condcu();

#endif //BRICK_STENCILS_CU_H
