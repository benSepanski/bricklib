//
// Created by Benjamin Sepanski on 12/2/21.
//

#ifndef BRICK_BRICK_STENCILS_H
#define BRICK_BRICK_STENCILS_H

#include <iomanip>
#include <iostream>
#include <functional>
#include <cstring>
#include <utility>

#include <cassert>

#include "Array.h"
#include "BrickedArray.h"
#include "IndexSpace.h"
#include "array-mpi.h"
#include "brick-cuda.h"
#include "brickcompare.h"
#include "bricksetup.h"
#include "multiarray.h"

// useful constants
constexpr unsigned RANK = 6;
#ifndef GENE6D_BRICK_DIM
#define GENE6D_BRICK_DIM 2,32,2,2,1,1
#endif
constexpr std::array<unsigned, RANK> BRICK_DIM = {GENE6D_BRICK_DIM};
constexpr std::array<unsigned, RANK> BRICK_VECTOR_DIM = {1,1,1,1,1,1};
constexpr std::array<unsigned, RANK> PCOEFF_BRICK_DIM = {BRICK_DIM[0], BRICK_DIM[2], BRICK_DIM[3], BRICK_DIM[4], BRICK_DIM[5]};
constexpr std::array<unsigned, RANK> ARAKAWA_COEFF_BRICK_DIM = {
    1, BRICK_DIM[0], BRICK_DIM[2], BRICK_DIM[3], BRICK_DIM[4], BRICK_DIM[5]};
constexpr std::array<unsigned, RANK> PADDING = {0,0,2,2,0,0};
constexpr unsigned TILE_SIZE = 8;
constexpr unsigned ARAKAWA_STENCIL_SIZE = 13;
constexpr unsigned NUM_ELEMENTS_PER_FIELD_BRICK =
    BRICK_DIM[0] * BRICK_DIM[1] * BRICK_DIM[2] * BRICK_DIM[3] * BRICK_DIM[4] * BRICK_DIM[5];

// useful types
typedef Dim<BRICK_DIM[5], BRICK_DIM[4], BRICK_DIM[3], BRICK_DIM[2], BRICK_DIM[1], BRICK_DIM[0]>
    FieldBrickDimsType;
typedef Dim<PCOEFF_BRICK_DIM[4], PCOEFF_BRICK_DIM[3], PCOEFF_BRICK_DIM[2], PCOEFF_BRICK_DIM[1], PCOEFF_BRICK_DIM[0]>
    PreCoeffBrickDimsType;
typedef Dim<ARAKAWA_COEFF_BRICK_DIM[5], ARAKAWA_COEFF_BRICK_DIM[4], ARAKAWA_COEFF_BRICK_DIM[3],
            ARAKAWA_COEFF_BRICK_DIM[2], ARAKAWA_COEFF_BRICK_DIM[1], ARAKAWA_COEFF_BRICK_DIM[0]>
    ArakawaCoeffBrickDimsType;
typedef Dim<BRICK_VECTOR_DIM[5], BRICK_VECTOR_DIM[4], BRICK_VECTOR_DIM[3], BRICK_VECTOR_DIM[2],
            BRICK_VECTOR_DIM[1], BRICK_VECTOR_DIM[0]>
    VectorFoldType;
typedef CommDims<false, false, false, false, false, true> CommIn_i;
typedef CommDims<false, false, true, true, false, false> CommIn_kl;
typedef CommDims<false, false, false, false, false, false> NoComm;
typedef Brick<FieldBrickDimsType, VectorFoldType, true, CommIn_i> FieldBrick_i;
typedef Brick<FieldBrickDimsType, VectorFoldType, true, CommIn_kl> FieldBrick_kl;
typedef Brick<PreCoeffBrickDimsType, Dim<1>, true, NoComm> PreCoeffBrick;
typedef Brick<ArakawaCoeffBrickDimsType, VectorFoldType, false, NoComm> ArakawaCoeffBrick;

typedef brick::Padding<PADDING[5], PADDING[4], PADDING[3], PADDING[2], PADDING[1], PADDING[0]>
    Padding_kl6D;
typedef brick::Padding<PADDING[5], PADDING[4], PADDING[3], PADDING[2], PADDING[0]>
    Padding_kl5D;
typedef brick::Array<bComplexElem, 6, Padding_kl6D> complexArray6D;
typedef brick::Array<bComplexElem, 5, Padding_kl5D> complexArray5D;
typedef brick::Array<bComplexElem, 1, brick::Padding<PADDING[1]> > complexArray1D_J;
typedef brick::Array<bElem, 6> realArray6D;
typedef brick::BrickedArray<bComplexElem, FieldBrickDimsType, VectorFoldType> BrickedFieldArray;
typedef brick::BrickedArray<bComplexElem, PreCoeffBrickDimsType, Dim<1> >BrickedPCoeffArray;
typedef brick::BrickedArray<bElem, ArakawaCoeffBrickDimsType, VectorFoldType>
    BrickedArakawaCoeffArray;

// some CUDA constants
constexpr unsigned WARP_SIZE = 32;
#if __CUDA_ARCH__ <= 370 || __CUDA_ARCH__ == 750 || __CUDA_ARCH__ == 860
constexpr unsigned MAX_BLOCKS_PER_SM = 16;
#else
constexpr unsigned MAX_BLOCKS_PER_SM = 32;
#endif
#if __CUDA_ARCH__ <= 720 || __CUDA_ARCH__ == 800
constexpr unsigned MAX_WARPS_PER_SM = 64;
#elif __CUDA_ARCH__ == 750
constexpr unsigned MAX_WARPS_PER_SM = 32;
#elif __CUDA__ == 860
constexpr unsigned MAX_WARPS_PER_SM = 48;
#else
#error Unexpected compute capability #__CUDA_ARCH__
#endif

enum BricksArakawaKernelType {
  SIMPLE_KLIJMN,
  OPT_IJKLMN,
  OPT_IKJLMN,
  OPT_IKLJMN,
  OPT_KIJLMN,
  OPT_KILJMN,
  OPT_KLIJMN
};

/**
 * return kernelType as a string
 * @param kernelType the kernel type
 * @return the string of the axis order
 */
inline std::string toString(BricksArakawaKernelType kernelType) {
  switch(kernelType) {
  case SIMPLE_KLIJMN:
  case OPT_KLIJMN:
    return "klijmn";
  case OPT_IJKLMN:
    return "ijklmn";
  case OPT_IKJLMN:
    return "ikjlmn";
  case OPT_IKLJMN:
    return "ikljmn";
  case OPT_KIJLMN:
    return "kijlmn";
  case OPT_KILJMN:
    return "kiljmn";
  default:
    throw std::runtime_error("Unrecognized kernelType");
  }
}

/**
 * Easy printing for arakawa kernel type
 */
inline std::ostream& operator<<(std::ostream& out, BricksArakawaKernelType kernelType) {
  switch(kernelType) {
  case SIMPLE_KLIJMN:
    out << "Simple KLIJMN";
    break;
  case OPT_KLIJMN:
    out << "Optimized klijmn";
    break;
  case OPT_IJKLMN:
    out << "Optimized ijklmn";
    break;
  case OPT_IKJLMN:
    out << "Optimized ikjlmn";
    break;
  case OPT_IKLJMN:
    out << "Optimized ikljmn";
    break;
  case OPT_KIJLMN:
    out << "Optimized kijlmn";
    break;
  case OPT_KILJMN:
    out << "Optimized kiljmn";
    break;
  default:
    throw std::runtime_error("Unrecognized kernelType");
  }
  return out;
}

/**
 * Build and return a function which, given bricks (bIn_dev, bOut_dev)
 * on the device, computes the ij-deriv stencil on bIn_dev and stores the
 * result into bOut_dev
 *
 * This function copies the i_deriv_coeffs to a fixed, global constant memory
 * location on the device, so the produced kernel may become incorrect if
 * this function is called multiple times
 *
 * @param fieldLayout the layout of field bricks
 * @param p1 the coefficients
 * @param p2 the coefficients
 * @param ikj i * 2pi * fourier Mode
 * @param i_deriv_coeffs the five coefficients for the five-point stencil
 *
 * @return the function which invokes the kernel
 */
typedef std::function<void(FieldBrick_i,FieldBrick_i)> ijDerivBrickKernelType;
ijDerivBrickKernelType buildBricksIJDerivBrickKernel(brick::BrickLayout<RANK> fieldLayout,
                                                     BrickedPCoeffArray &p1,
                                                     BrickedPCoeffArray &p2,
                                                     const complexArray1D_J &ikj,
                                                     bElem i_deriv_coeffs[5]);


/**
 * Build and return a function which, given bricks (bIn_dev, bOut_dev)
 * on the device, computes the arakawa stencil on bIn_dev and stores the
 * result into bOut_dev
 *
 * @param fieldLayout the layout of field bricks
 * @param bCoeff the coefficients to use
 * @param kernelType which kernel to use
 *
 * @return the function which invokes the kernel
 */
typedef std::function<void(FieldBrick_kl,FieldBrick_kl)> ArakawaBrickKernelType;
ArakawaBrickKernelType buildBricksArakawaKernel(brick::BrickLayout<RANK> fieldLayout, BrickedArakawaCoeffArray bCoeff, BricksArakawaKernelType kernelType);

/**
 * Thrown an error if any pair of elements from arr1 and arr2 are
 * not within the relative tolerance (and also not within the absolute tolerance)
 *
 * @param1 arr1 the first array
 * @param2 arr2 the second array
 * @param rtol the maximum relative difference between array elements
 * @param atol the maximum absolute difference between array elements
 */
template<typename ArrType1, typename ArrType2>
void checkClose(ArrType1 arr1, ArrType2 arr2, std::array<unsigned, RANK> ghostZone, double rtol = 1.0e-6, double atol = 1.0e-20) {
  std::array<brick::Interval<unsigned>, RANK> bounds{};
  static_assert(ArrType1::RANK == RANK, "Mismatch in rank of ArrType1");
  static_assert(ArrType2::RANK == RANK, "Mismatch in rank of ArrType2");
  for(unsigned d = 0; d < RANK; ++d) {
    if(arr1.extent[d] != arr2.extent[d]) {
      throw std::runtime_error("Extents of arr1 and arr2 don't match");
    }
    bounds[d].low = ghostZone[d];
    bounds[d].high = arr1.extent[d] - ghostZone[d];
  }
  brick::IndexSpace<RANK> indexSpace(bounds);
  for(brick::Index<RANK, unsigned> idx : indexSpace) {
    auto val1 = arr1(idx.i(), idx.j(), idx.k(), idx.l(), idx.m(), idx.n());
    auto val2 = arr2(idx.i(), idx.j(), idx.k(), idx.l(), idx.m(), idx.n());
    auto diff = std::abs((std::complex<bElem>) val1 - (std::complex<bElem>) val2);
    auto norm1 = std::abs((std::complex<bElem>) val1);
    auto norm2 = std::abs((std::complex<bElem>) val2);
    if(diff / (norm1 + norm2) * 2 >= rtol && diff >= atol) {
      std::ostringstream errStream;
      errStream << "Mismatch at " << idx << ": "
        << "val1 = " << val1 << " != " << val2 << " = val2" << std::endl;
      throw std::runtime_error(errStream.str());
    }
  }
}

#endif // BRICK_BRICK_STENCILS_H
