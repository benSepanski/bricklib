/**
 * @file
 * @brief Interface to code generator
 */

#ifndef BRICK_VECSCATTER_H
#define BRICK_VECSCATTER_H

#include <complex>
#if defined(__CUDACC__)
#include <cuComplex.h>
#endif

/**
 * @brief Basic datatype for all brick elements
 */
#ifndef bElem
#define bElem double
#endif

// typedef std::complex<bElem> bComplexElem;
// Use the default, C-style CUDA complex types
#if defined(__CUDACC__)
    #if bElem==double
        typedef cuDoubleComplex bCuComplexElem;
        #define CUCADD(a, b) cuCadd(a, b)
        #define CUCMUL(a, b) cuCmul(a, b)
        #define CUCREAL(a) cuCreal(a)
        #define CUCIMAG(a) cuCimag(a)
    #elif bElem==float
        typedef cuFloatComplex bCuComplexElem;
        #define CUCADD(a, b) cuCaddf(a, b)
        #define CUCMUL(a, b) cuCmulf(a, b)
        #define CUCREAL(a) cuCrealf(a)
        #define CUCIMAG(a) cuCimagf(a)
    #else
        #error "Expected bElem to be double or float"
    #endif
#endif

typedef std::complex<bElem> stdComplex;

struct bComplexElem : public stdComplex
{
    using stdComplex::stdComplex;
    using stdComplex::operator=;

#if defined(__CUDACC__)
    // Automatic conversion to/assignment from cuda types
    #if bElem==double
        __host__ __device__ inline
        operator cuDoubleComplex() const{return *((cuDoubleComplex*) this);}

    #elif bElem==float
        __host__ __device__ inline
        operator cuFloatComplex() const{return *((cuFloatComplex*) this);}
    #else
    #error "Expected bElem to be double or float"
    #endif

    __host__ __device__ inline
    bComplexElem operator*(const bComplexElem &that)
    {
        #ifdef __CUDA_ARCH__
        return CUCMUL(*this, that);
        #else
        return (*this) * that;
        #endif
    }

    __host__ __device__ inline
    bComplexElem operator+(const bComplexElem &that)
    {
        #ifdef __CUDA_ARCH__
        return CUCADD(*this, that);
        #else
        return (*this) + that;
        #endif
    }

    __host__ __device__ inline
    bComplexElem &operator=(const bCuComplexElem &that)
    {
        *this = *((bComplexElem*)&that);
        return *this;
    }

    __host__ __device__ inline
    bComplexElem(const bCuComplexElem &that)
    {
        reinterpret_cast<bElem(&)[2]>(*this)[0] = CUCREAL(that);
        reinterpret_cast<bElem(&)[2]>(*this)[1] = CUCIMAG(that);
    }
#endif
};

#define VS_STRING(...) #__VA_ARGS__
#define VS_TOSTR(...) VS_STRING(__VA_ARGS__)

#define _SELECTMACRO(_v0, _v1, _v2, _v3, _v4, _v6, NAME, ...) NAME

/**
 * @brief Inject stencil code for tiling here
 *
 * Versions of this call with different number of variables are defined in #_tile4(file, vec, vsdim, titer) and
 * #_tile5(file, vec, vsdim, titer, stri).
 */
#define tile(...) _SELECTMACRO(__VA_ARGS__, 0, _tile5, _tile4)(__VA_ARGS__)
/**
 * @brief 4 argument variation of #tile(...)
 * @param file Path to the python stencil expression file as a string.
 * @param vec Vectorization ISA, available choices see \ref md_docs_vectorization
 * @param vsdim Tile dimensions
 * @param titer Tiling iteration index
 *
 * If file is given as a relative path, it is specified as relative to the current file.
 *
 * Use from #tile(...), for example:
 * @code{.cpp}
 * tile("7pt.py", "AVX2", (8, 8, 8), ("tk", "tj", "ti"));
 * @endcode
 */
#define _tile4(file, vec, vsdim, titer) do { _Pragma(VS_TOSTR(vecscatter Scatter Tile(__FILE__, __LINE__, file, VS_TOSTR(bElem), vec, tile_iter=titer, dim=vsdim))) } while (false)
/**
 * @brief 5 argument variation of #tile(...)
 * @param stri Stride of shift during greedy algorithm
 *
 * For other parameters see #_tile4(file, vec, vsdim, titer)
 *
 * Use from #tile(...), for example the following will only select shift of multiples of 4 on i-dimension:
 * @code{.cpp}
 * tile("7pt.py", "AVX2", (8, 8, 8), ("tk", "tj", "ti"), (1, 1, 4));
 * @endcode
 */
#define _tile5(file, vec, vsdim, titer, stri) do { _Pragma(VS_TOSTR(vecscatter Scatter Tile(__FILE__, __LINE__, file, VS_TOSTR(bElem), vec, tile_iter=titer, dim=vsdim, stride=stri))) } while (false)

/**
 * @brief Inject stencil code for brick datalayout
 *
 * Versions of this call with different number of variables are defined in #_brick5(file, vec, vsdim, vsfold, brickIdx)
 * and #_brick6(file, vec, vsdim, vsfold, brickIdx, stri)
 */
#define brick(...) _SELECTMACRO(__VA_ARGS__, _brick6, _brick5)(__VA_ARGS__)
/**
 * @brief 5 arguments version of #brick(...)
 * @param file Path to the python stencil expression file as a string.
 * @param vec Vectorization ISA, available choices see \ref md_docs_vectorization
 * @param vsdim Brick dimensions
 * @param vsfold (folded-)Vector dimensions
 * @param brickIdx Index of the brick to compute
 *
 * If file is given as a relative path, it is specified as relative to the current file.
 *
 * Use from #brick(...), for example:
 * @code{.cpp}
 * tile("7pt.py", "AVX2", (8, 8, 8), (2, 2), b);
 * @endcode
 */
#define _brick5(file, vec, vsdim, vsfold, brickIdx) do { _Pragma(VS_TOSTR(vecscatter Scatter Brick(__FILE__, __LINE__, file, VS_TOSTR(bElem), vec, bidx=VS_TOSTR(brickIdx), dim=vsdim, fold=vsfold))) } while (false)
/**
 * @brief 5 arguments version of #brick(...)
 * @param stri Stride of shift during greedy algorithm
 *
 * For other parameters see #_brick5(file, vec, vsdim, vsfold, brickIdx)
 *
 * Use from #brick(...), for example:
 * @code{.cpp}
 * tile("7pt.py", "AVX2", (8, 8, 8), (2, 2), b, (1, 1, 2));
 * @endcode
 */
#define _brick6(file, vec, vsdim, vsfold, brickIdx, stri) do { _Pragma(VS_TOSTR(vecscatter Scatter Brick(__FILE__, __LINE__, file, VS_TOSTR(bElem), vec, bidx=VS_TOSTR(brickIdx), dim=vsdim, fold=vsfold, stride=stri))) } while (false)

#endif //BRICK_VECSCATTER_H
