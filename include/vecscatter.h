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
    #if bElem==double
        typedef cuDoubleComplex bCuComplexElem ;
    #else
        typedef cuFloatComplex bCuComplexElem;
    #endif
#else
    #define FORCUDA
#endif

// std::complex doesn't work in kernels,
// thrust::complex/C-style CUDA complex types can only be linked
// against if using nvcc.
//
// Since all use the same layout (an array of 2 float/doubles,
// real first) it's more convenient for us to just build our own
// compatible struct that works on device and host code.

struct bComplexElem
{
    bElem value[2];

    // default constructor: random values
    FORCUDA
    bComplexElem() { }
    // basic constructors
    FORCUDA
    bComplexElem(const bElem &real, const bElem &imag) : value{real, imag} { }
    FORCUDA
    bComplexElem(const bElem &real) : value{real, 0.0} { }
    FORCUDA
    bComplexElem(const int &real) : value{(bElem) real, 0.0} { }
    FORCUDA
    bComplexElem(const long &real) : value{(bElem) real, 0.0} { }
    FORCUDA
    bComplexElem(const bComplexElem &that) : value{that.real(), that.imag()} { }

    // conversion from std/cuda
    bComplexElem(const std::complex<bElem> &that) 
    {
        value[0] = reinterpret_cast<const bElem(&)[2]>(that)[0];
        value[1] = reinterpret_cast<const bElem(&)[2]>(that)[1];
    }

    #if defined(__CUDACC__) || defined(__HIP__)
    FORCUDA inline
    bComplexElem(bCuComplexElem &that) 
    {
        value[0] = reinterpret_cast<const bElem(&)[2]>(that)[0];
        value[1] = reinterpret_cast<const bElem(&)[2]>(that)[1];
    }
    #endif

    // conversion to std/cuda
    operator std::complex<bElem>() const
    {
        return reinterpret_cast<const std::complex<bElem>&>(*this);
    }

    #if defined(__CUDACC__) || defined(__HIP__)
    #if bElem==double
        FORCUDA inline
        operator cuDoubleComplex() const 
        {
            return reinterpret_cast<const bCuComplexElem&>(*this);
        }
    #else
        FORCUDA inline
        operator cuFloatComplex() const 
        {
            return reinterpret_cast<const bCuComplexElem&>(*this);
        }
    #endif
    #endif

    // assignment
    FORCUDA inline
    bComplexElem &operator=(const bComplexElem &that)
    {
        this->value[0] = that.real();
        this->value[1] = that.imag();
        return *this;
    }

    // basic complex operations
    FORCUDA inline
    const bElem &real() const
    {
        return value[0];
    }

    FORCUDA inline
    const bElem &imag() const
    {
        return value[1];
    }

    // declare friends for unary/binary operations
    FORCUDA inline
    friend bComplexElem operator+(const bComplexElem &, const bComplexElem &);
    FORCUDA inline
    friend bComplexElem operator-(const bComplexElem &, const bComplexElem &);
    FORCUDA inline
    friend bComplexElem operator*(const bComplexElem &, const bComplexElem &);
    FORCUDA inline
    friend bComplexElem operator+(const bComplexElem &);
    FORCUDA inline
    friend bComplexElem operator-(const bComplexElem &);

    // complex arithmetic updates
    FORCUDA inline
    bComplexElem &operator+=(const bComplexElem &that)
    {
        *this = *this + that;
        return *this;
    }

    FORCUDA inline
    bComplexElem &operator-=(const bComplexElem &that)
    {
        *this = *this - that;
        return *this;
    }

    FORCUDA inline
    bComplexElem &operator*=(const bComplexElem &that)
    {
        *this = *this * that;
        return *this;
    }

    // comparison
    FORCUDA inline
    bool operator==(const bComplexElem &that)
    {
        return this->real() == that.real()  && this->imag() == that.imag();
    }

    FORCUDA inline
    bool operator!=(const bComplexElem &that)
    {
        return !(*this == that);
    }
};

// complex arithmetic binary operators
FORCUDA
bComplexElem operator+(const bComplexElem &z, const bComplexElem &w)
{
    return bComplexElem(z.real() + w.real(), z.imag() + w.imag());
}

FORCUDA
bComplexElem operator-(const bComplexElem &z, const bComplexElem &w)
{
    return bComplexElem(z.real() - w.real(), z.imag() - w.imag());
}

FORCUDA
bComplexElem operator*(const bComplexElem &z, const bComplexElem &w)
{
    return bComplexElem(z.real() * w.real() - z.imag() * w.imag(),
                        z.imag() * w.real() + z.real() * w.imag());
}

// complex arithmetic unary operators
FORCUDA
bComplexElem operator+(const bComplexElem &z)
{
    return z;
}

FORCUDA
bComplexElem operator-(const bComplexElem &z)
{
    return bComplexElem(-z.real(), -z.imag());
}

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
