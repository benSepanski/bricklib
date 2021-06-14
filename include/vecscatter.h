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

// as in brick.h,
/// Overloaded attributes for potentially GPU-usable functions (in place of __host__ __device__ etc.)
#if defined(__CUDACC__) || defined(__HIP__)
#define FORCUDA __host__ __device__
#else
#define FORCUDA
#endif

// Inside CUDA kernels, use the default, C-style CUDA complex types
#if defined(__CUDACC__)
    #if bElem==double
        typedef cuDoubleComplex bCuComplexElem;
        #define CUCADD(z, w) cuCadd(z, w)
        #define CUCMUL(z, w) cuCmul(z, w)
        #define CUCDIV(z, w) cuCdiv(z, w)
        #define CUCREAL(z) cuCreal(z)
        #define CUCIMAG(z) cuCimag(z)
    #elif bElem==float
        typedef cuFloatComplex bCuComplexElem;
        #define CUCADD(z, w) cuCaddf(z, w)
        #define CUCMUL(z, w) cuCmulf(z, w)
        #define CUCDIV(z, w) cuCdivf(z, w)
        #define CUCREAL(z) cuCrealf(z)
        #define CUCIMAG(z) cuCimagf(z)
    #else
        #error "Expected bElem to be double or float"
    #endif
    #define FROMCUDA(z) reinterpet_cast<bComplexElem>(z)
#endif

#if bElem==double
    #define AS_ARRAY(z) reinterpret_cast<double(&)[2]>(z)
    #define AS_CONST_ARRAY(z) reinterpret_cast<const double(&)[2]>(z)
#elif bElem==float
    #define AS_ARRAY(z) reinterpret_cast<float(&)[2]>(z)
    #define AS_CONST_ARRAY(z) reinterpret_cast<const float(&)[2]>(z)
#else
    #error "expected bElem to be float or double"
#endif

typedef std::complex<bElem> stdComplex;

struct bComplexElem : public stdComplex
{
    // inherit parent constructors for host-side
    using stdComplex::stdComplex;

    // allow host-side implicit conversion from std
    bComplexElem(const stdComplex &that) : stdComplex(that) { }

    // redefine constructors needed on the device-side
    FORCUDA
    bComplexElem() : stdComplex() { }

    FORCUDA
    bComplexElem(const bComplexElem &that) : stdComplex()
    {
        AS_ARRAY(*this)[0] = AS_CONST_ARRAY(that)[0];
        AS_ARRAY(*this)[0] = AS_CONST_ARRAY(that)[0];
    }

// Cuda compatability
#if defined(__CUDACC__)
    // Conversion to/from cuda complex
    #if bElem==double
    __host__ __device__ inline
    operator cuDoubleComplex() const
    {
        return *((cuDoubleComplex*) this);
    }
    #elif bElem==float
    __host__ __device__ inline
    operator cuFloatComplex() const
    {
        return *((cuFloatComplex*) this);
    }
    #endif

    __host__ __device__ inline
    bComplexElem(const bCuComplexElem &that)
    {
        AS_ARRAY(*this)[0] = CUCREAL(that);
        AS_ARRAY(*this)[1] = CUCIMAG(that);
    }

    // Overloaded arithmetic operators
    __host__ __device__ inline
    bComplexElem &operator+=(const bComplexElem &that)
    {
        #ifdef __CUDA_ARCH__
        AS_ARRAY(*this)[0] += CUCREAL(that);
        AS_ARRAY(*this)[1] += CUCIMAG(that);
        return *this;
        #else
        return static_cast<bComplexElem &>(stdComplex::operator+=(static_cast<const stdComplex&>(that)));
        #endif
    }

    __host__ __device__ inline
    bComplexElem &operator-=(const bComplexElem &that)
    {
        #ifdef __CUDA_ARCH__
        AS_ARRAY(*this)[0] -= CUCREAL(that);
        AS_ARRAY(*this)[1] -= CUCIMAG(that);
        return *this;
        #else
        return static_cast<bComplexElem &>(stdComplex::operator-=(static_cast<const stdComplex&>(that)));
        #endif
    }

    __host__ __device__ inline
    bComplexElem &operator*=(const bComplexElem &that)
    {
        #ifdef __CUDA_ARCH__
        *this = CUCMUL(*this, that);
        return *this;
        #else
        return static_cast<bComplexElem &>(stdComplex::operator*=(static_cast<const stdComplex&>(that)));
        #endif
    }
    
    __host__ __device__ inline
    bComplexElem &operator/=(const bComplexElem &that)
    {
        #ifdef __CUDA_ARCH__
        *this = CUCDIV(*this, that);
        return *this;
        #else
        return static_cast<bComplexElem &>(stdComplex::operator/=(static_cast<const stdComplex&>(that)));
        #endif
    }
#endif
};

// overloaded arithmetic operators
#if defined(__CUDACC__)
__host__ __device__ inline
bComplexElem operator*(const bComplexElem &a, const bComplexElem &b)
{
    #ifdef __CUDA_ARCH__
    return CUCMUL(a, b);
    #else
    return std::operator*(a, b);
    #endif
}

__host__ __device__ inline
bComplexElem operator+(const bComplexElem &a, const bComplexElem &b)
{
    #ifdef __CUDA_ARCH__
    return CUCADD(a, b);
    #else
    return std::operator+(a, b);
    #endif
}
#endif

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
