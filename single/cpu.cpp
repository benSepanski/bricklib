#include <iostream>
#include <memory>
#include <omp.h>
#include <random>
#include "brick.h"
#include "stencils/stencils.h"

bElem *coeff;
bComplexElem *zCoeff;

int main() {
  coeff = (bElem *) malloc(129 * sizeof(bElem));
  zCoeff = (bComplexElem *) malloc(129 * sizeof(bComplexElem));
  std::random_device r;
  std::mt19937_64 mt(r());
  std::uniform_real_distribution<bElem> u(0, 1);

  for (int i = 0; i < 129; ++i)
  {
    coeff[i] = u(mt);
    zCoeff[i] = bComplexElem(u(mt), u(mt));
  }

  copy();
  d3pt7();
  d3pt7complex();
  d3pt27();
  d3cond();
  std::cout << "result match" << std::endl;
  return 0;
}
