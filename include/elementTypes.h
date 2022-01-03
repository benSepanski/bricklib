/**
 * @file
 * @brief brick element types (mostly the complex element type)
 */
#ifndef BRICK_ELEMENT_TYPES_H
#define BRICK_ELEMENT_TYPES_H

#include <complex>
#include <iostream>
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

struct alignas(sizeof(bElem[2])) bComplexElem
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
    bComplexElem(const unsigned int &real) : value{(bElem) real, 0.0} { }
    FORCUDA
    bComplexElem(const long &real) : value{(bElem) real, 0.0} { }

    // conversion from std/cuda
    bComplexElem(const std::complex<bElem> &that)
    {
        *this = reinterpret_cast<const bComplexElem &>(that);
    }

    #if defined(__CUDACC__) || defined(__HIP__)
    FORCUDA inline
    bComplexElem(bCuComplexElem &that) 
    {
        *this = reinterpret_cast<const bComplexElem &>(that);
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
    friend bComplexElem operator+(const bElem &, const bComplexElem &);
    FORCUDA inline
    friend bComplexElem operator+(const bComplexElem &, const bElem &);
    FORCUDA inline
    friend bComplexElem operator-(const bComplexElem &, const bComplexElem &);
    FORCUDA inline
    friend bComplexElem operator-(const bElem &, const bComplexElem &);
    FORCUDA inline
    friend bComplexElem operator-(const bComplexElem &, const bElem &);
    FORCUDA inline
    friend bComplexElem operator*(const bElem &, const bComplexElem &);
    FORCUDA inline
    friend bComplexElem operator*(const bComplexElem &, const bElem &);
    FORCUDA inline
    friend bComplexElem operator/(const bComplexElem &, const bComplexElem &);
    FORCUDA inline
    friend bComplexElem operator/(const bComplexElem &, const bElem &);
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
    bComplexElem &operator+=(const bElem &that)
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
    bComplexElem &operator-=(const bElem &that)
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

    FORCUDA inline
    bComplexElem &operator*=(const bElem &that)
    {
        *this = *this * that;
        return *this;
    }

    FORCUDA inline
    bComplexElem &operator/=(const bComplexElem &that)
    {
        *this = *this / that;
        return *this;
    }

    FORCUDA inline
    bComplexElem &operator/=(const bElem &that)
    {
        *this = *this / that;
        return *this;
    }

    // comparison
    FORCUDA inline
    bool operator==(const bComplexElem &that) const
    {
        return this->real() == that.real()  && this->imag() == that.imag();
    }

    FORCUDA inline
    bool operator!=(const bComplexElem &that) const
    {
        return !(*this == that);
    }
};

// complex addition (and specializations for real values)
bComplexElem operator+(const bComplexElem &z, const bComplexElem &w)
{
    return bComplexElem(z.real() + w.real(), z.imag() + w.imag());
}

bComplexElem operator+(const bElem &r, const bComplexElem &w)
{
    return bComplexElem(r + w.real(), w.imag());
}

bComplexElem operator+(const bComplexElem &z, const bElem &r)
{
    return r + z;
}

// complex subtraction (and specializations for real values)
bComplexElem operator-(const bComplexElem &z, const bComplexElem &w)
{
    return bComplexElem(z.real() - w.real(), z.imag() - w.imag());
}

bComplexElem operator-(const bElem &r, const bComplexElem &w)
{
    return bComplexElem(r - w.real(), - w.imag());
}

bComplexElem operator-(const bComplexElem &z, const bElem &r)
{
    return bComplexElem(z.real() - r, z.imag());
}

// complex multiplication (and specializations for real values)
bComplexElem operator*(const bComplexElem &z, const bComplexElem &w)
{
    return bComplexElem(z.real() * w.real() - z.imag() * w.imag(),
                        z.imag() * w.real() + z.real() * w.imag());
}

bComplexElem operator*(const bElem &r, const bComplexElem &w)
{
    return bComplexElem(r * w.real(), r * w.imag());
}

bComplexElem operator*(const bComplexElem &z, const bElem &r)
{
    return r * z;
}

// copied and modified from http://pixel.ecn.purdue.edu:8080/purpl/WSJ/projects/DirectionalStippling/include/cuComplex.h
bComplexElem operator/(const bComplexElem &z, const bComplexElem &w)
{
    bElem s = ((bElem)abs((double) w.real())) + 
              ((bElem)abs((double) w.imag()));
    bElem oos = 1.0f / s;
    bElem ars = z.real() * oos;
    bElem ais = z.imag() * oos;
    bElem brs = w.real() * oos;
    bElem bis = w.imag() * oos;
    s = (brs * brs) + (bis * bis);
    oos = 1.0f / s;
    bComplexElem quot(((ars * brs) + (ais * bis)) * oos,
                      ((ais * brs) - (ars * bis)) * oos);
    return quot;
}

bComplexElem operator/(const bComplexElem &z, const bElem &r)
{
    return bComplexElem(z.real() / r, z.imag() / r);
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

// pretty-printing
inline
std::ostream &operator<<(std::ostream &out, const bComplexElem& z)
{
    out << z.real() << "+" << z.imag() << "*I";
    return out;
}

#endif // BRICK_ELEMENT_TYPES_H

