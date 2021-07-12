/**
 * @file
 * @brief Main header for bricks
 */

#ifndef BRICK_H
#define BRICK_H

#include <iostream>
#include <cassert>
#include <stdlib.h>
#include <type_traits>
#include <memory>
#include "vecscatter.h"

/// BrickStorage allocation alignment
#define ALIGN 2048

/// Overloaded attributes for potentially GPU-usable functions (in place of __host__ __device__ etc.)
#if defined(__CUDACC__) || defined(__HIP__)
#define FORCUDA __host__ __device__
#else
#define FORCUDA
#endif

/**
 * @defgroup static_power Statically compute exponentials
 * @{
 */

/// Compute \f$base^{exp}\f$ @ref static_power
template<unsigned base, unsigned exp>
struct static_power {
  static constexpr unsigned value = base * static_power<base, exp - 1>::value;
};

/// Return 1 @ref static_power
template<unsigned base>
struct static_power<base, 0> {
  static constexpr unsigned value = 1;
};
/**@}*/

/**
 * @brief Initializing and holding the storage of bricks
 *
 * It requires knowing how many bricks to store before allocating.
 *
 * Built-in allocators are host-only.
 */
struct BrickStorage {
  /// Pointer holding brick data
  std::shared_ptr<bElem> dat;
  /**
   * @brief Number of chunks
   *
   * A chunk can contain multiple bricks from different sub-fields. Forming structure-of-array.
   */
  long chunks;
  /// Size of a chunk in number of real elements (each complex elements counts as 2 real elements)
  size_t step;
  /// MMAP data structure when using mmap as allocator
  void *mmap_info = nullptr;

  /// Allocation using *alloc
  static BrickStorage allocate(long chunks, size_t step) {
    BrickStorage b;
    b.chunks = chunks;
    b.step = step;
    b.dat = std::shared_ptr<bElem>((bElem*)aligned_alloc(ALIGN, chunks * step * sizeof(bElem)), free);
    return b;
  }

  /// mmap allocator using default (new) file
  static BrickStorage mmap_alloc(long chunks, long step);

  /// mmap allocator using specified file starting from certain offset
  static BrickStorage mmap_alloc(long chunks, long step, void *mmap_fd, size_t offset);
};

// TODO : docs
template<bool ... CommInDim>
struct CommDims
{
  public:
    constexpr static bool communicatesInDim(unsigned dim)
    {
      unsigned length = sizeof...(CommInDim);
      if(dim < length)
      {
        constexpr unsigned doesCommunicateInDim[sizeof...(CommInDim)] = { CommInDim... };
        return doesCommunicateInDim[sizeof...(CommInDim) - 1 - dim];
      }
      return true;
    }

    constexpr static unsigned numCommunicatingDims(unsigned numDims)
    {
      unsigned numCommunicatingDims = 0;
      for(unsigned dim = 0; dim < numDims; ++dim)
      {
        numCommunicatingDims += CommDims<CommInDim...>::communicatesInDim(dim);
      }
      return numCommunicatingDims;
    }
};

template<unsigned dims, typename CommunicatingDims = CommDims<>>
struct BrickInfo;

/**
 * @brief Metadata related to bricks
 * @tparam dims
 *
 * It stores the adjacency list used by the computation. One of this data structure can be shared among multiple bricks.
 * In fact, for computation to succeed, it will require having the same adjacencies for all participating bricks.
 *
 * Each index of the adjacency list will indicate the memory location in the BrickStorage.
 *
 * Metadata can be used to allocate storage with minimal effort. It is recommended to build the metadata before creating
 * the storage.
 */
template<unsigned dims, bool ... CommInDim>
struct BrickInfo<dims, CommDims<CommInDim...> > {
  /// Type describing which dimensions are communicating
  typedef CommDims<CommInDim...> myCommDims;
  /// Adjacency list type
  typedef unsigned (*adjlist)[static_power<3, myCommDims::numCommunicatingDims(dims)>::value];
  /// Adjacency list
  adjlist adj;
  /// Number of bricks in this list
  unsigned nbricks;

  /**
   * @brief Creating an empty metadata consisting of the specified number of bricks
   * @param nbricks number of bricks
   */
  explicit BrickInfo(unsigned nbricks) : nbricks(nbricks) {
    adj = (adjlist) malloc(nbricks * static_power<3, myCommDims::numCommunicatingDims(dims)>::value * sizeof(unsigned));
  }

  /// Allocate a new brick storage BrickStorage::allocate()
  BrickStorage allocate(long step) {
    return BrickStorage::allocate(nbricks, step);
  }

  /// Allocate a new brick storage BrickStorage::mmap_alloc(long, long)
  BrickStorage mmap_alloc(long step) {
    return BrickStorage::mmap_alloc(nbricks, step);
  }

  /// Allocate a new brick storage BrickStorage::mmap_alloc(long, long, void*, size_t)
  BrickStorage mmap_alloc(long step, void *mmap_fd, size_t offset) {
    return BrickStorage::mmap_alloc(nbricks, step, mmap_fd, offset);
  }
};

/// Empty template to specify an n-D list
template<unsigned ... Ds>
struct Dim {
};

/**
 * @defgroup cal_size Calculate the product of n numbers in a template
 * @{
 */
/**
 * @brief Generic base template for @ref cal_size
 * @tparam xs A list of numbers
 */
template<unsigned ... xs>
struct cal_size;

/**
 * @brief return x when only one number left @ref cal_size
 * @tparam x
 */
template<unsigned x>
struct cal_size<x> {
  static constexpr unsigned value = x;
};

/**
 * @brief Head of the list multiply by result from the rest of list @ref cal_size
 * @tparam x CAR
 * @tparam xs CDR
 */
template<unsigned x, unsigned ... xs>
struct cal_size<x, xs...> {
  static constexpr unsigned value = x * cal_size<xs ...>::value;
};
/**@}*/

/**
 * @defgroup cal_offs Calculating the offset within the adjacency list
 * @{
 */
/**
 * @brief Generic base template for @ref cal_offs
 * @tparam offs Numbers within [0,2]
 */
template<unsigned ... offs>
struct cal_offs;

/**
 * @brief Return offset when only one offset left @ref cal_offs
 * @tparam off
 */
template<unsigned off>
struct cal_offs<1, off> {
  static constexpr unsigned value = off;
};

/**
 * @brief Compute the offset @ref cal_offs
 * @tparam dim Current dimension
 * @tparam off CAR
 * @tparam offs CDR
 */
template<unsigned dim, unsigned off, unsigned ...offs>
struct cal_offs<dim, off, offs...> {
  static constexpr unsigned value = off * static_power<3, dim - 1>::value + cal_offs<dim - 1, offs...>::value;
};
/**@}*/

/**
 * @defgroup _BrickAccessor Accessing brick elements using []
 *
 * It can be fully unrolled and offers very little overhead. However, vectorization is tricky without using codegen.
 *
 * For example, the following code produces types:
 * @code{.cpp}
 * Brick<Dim<8,8,8>, Dim<2,4>> bIn(&bInfo, bStorage, 0);
 * // bIn[0]: _BrickAccessor<bElem, Dim<8,8,8>, Dim<2,4>, void>
 * // bIn[0][1]: _BrickAccessor<bElem, Dim<8,8>, Dim<2,4>, bool>
 * // bIn[0][1][1][1]: bElem
 * @endcode
 *
 * @{
 */

/// Generic base template for @ref _BrickAccessor
template<typename...>
struct _BrickAccessor;

/// Last dimension @ref _BrickAccessor
template<typename T,
    unsigned D,
    unsigned F>
struct _BrickAccessor<T, Dim<D>, Dim<F>, bool> {
  T *par;         ///< parent Brick data structure reference
  typedef typename T::elemType elemType; ///< type of elements in the brick
  // True iff there is communication (between bricks) in this dimension
  static constexpr bool commInDim = T::myCommDims::communicatesInDim(0); 

  unsigned b;     ///< Reference (center) brick
  unsigned pos;   ///< Accumulative position within adjacency list
  unsigned nvec;  ///< Which vector
  unsigned wvec;  ///< Position within a vector

  FORCUDA
  _BrickAccessor(T *par, unsigned b, unsigned pos, unsigned nvec, unsigned wvec) :
      par(par), b(b), pos(pos), nvec(nvec), wvec(wvec) {
  }

  FORCUDA
  inline elemType &operator[](unsigned i) {
    // change pos
    unsigned dir = i + D;
    unsigned d = commInDim ? pos * 3 + dir / D
                           : pos;
    // new vec position
    unsigned l = dir % D;
    unsigned w = wvec * F + l % F;
    unsigned n = nvec * (D / F) + l / F;
    unsigned offset = n * par->VECLEN + w;

    return par->dat[par->bInfo->adj[b][d] * par->step + offset];
  }
};

/**
 * @brief When the number of Brick dimensions and Fold dimensions are the same @ref _BrickAccessor
 * @tparam T Element type
 * @tparam D CAR of brick dimension
 * @tparam BDims CDR of brick dimension
 * @tparam F CAR of vector folds
 * @tparam Folds CDR of vector folds
 */
template<typename T,
    unsigned D,
    unsigned F,
    unsigned ... BDims,
    unsigned ... Folds>
struct _BrickAccessor<T, Dim<D, BDims...>, Dim<F, Folds...>, bool> {
  T *par;         ///< parent Brick data structure reference
  typedef typename T::elemType elemType; ///< type of elements in the brick
  // True iff there is communication (between bricks) in this dimension
  static constexpr bool commInDim = T::myCommDims::communicatesInDim(sizeof...(BDims)); 

  unsigned b;     ///< Reference (center) brick
  unsigned pos;   ///< Accumulative position within adjacency list
  unsigned nvec;  ///< Which vector
  unsigned wvec;  ///< Position within a vector

  FORCUDA
  _BrickAccessor(T *par, unsigned b, unsigned pos, unsigned nvec, unsigned wvec) :
      par(par), b(b), pos(pos), nvec(nvec), wvec(wvec) {
  }

  FORCUDA
  inline _BrickAccessor<T, Dim<BDims...>, Dim<Folds...>, bool> operator[](unsigned i) {
    // change pos
    unsigned dir = i + D;
    unsigned d = commInDim ? pos * 3 + dir / D
                           : pos;
    // new vec position
    unsigned l = dir % D;
    unsigned w = wvec * F + l % F;
    unsigned n = nvec * (D / F) + l / F;
    return _BrickAccessor<T, Dim<BDims...>, Dim<Folds...>, bool>(par, b, d, n, w);
  }
};

/**
 * @brief When the number of Brick dimensions and Fold dimensions are not the same \f$1 + BDims > Folds\f$ @ref _BrickAccessor
 * @tparam T Element type
 * @tparam D CAR of brick dimension
 * @tparam BDims CDR of brick dimension
 * @tparam F CAR of vector folds
 * @tparam Folds CDR of vector folds
 */
template<typename T,
    unsigned D,
    unsigned ... BDims,
    unsigned ... Folds>
struct _BrickAccessor<T, Dim<D, BDims...>, Dim<Folds...>, void> {
  T *par;         ///< parent Brick data structure reference
  typedef typename T::elemType elemType; ///< type of elements in the brick
  // True iff there is communication (between bricks) in this dimension
  static constexpr bool commInDim = T::myCommDims::communicatesInDim(sizeof...(BDims)); 

  unsigned b;     ///< Reference (center) brick
  unsigned pos;   ///< Accumulative position within adjacency list
  unsigned nvec;  ///< Which vector
  unsigned wvec;  ///< Position within a vector

  FORCUDA
  _BrickAccessor(T *par, unsigned b, unsigned pos, unsigned nvec, unsigned wvec) :
      par(par), b(b), pos(pos), nvec(nvec), wvec(wvec) {
  }

  FORCUDA
  inline _BrickAccessor<T, Dim<BDims...>, Dim<Folds...>,
      typename std::conditional<sizeof...(BDims) == sizeof...(Folds), bool, void>::type>
  operator[](unsigned i) {
    // change pos
    unsigned dir = i + D;
    unsigned d = commInDim ? pos * 3 + dir / D
                           : pos;
    // new vec position
    unsigned l = dir % D;
    unsigned w = wvec;
    unsigned n = nvec * D + l;
    return _BrickAccessor<T, Dim<BDims...>, Dim<Folds...>,
        typename std::conditional<sizeof...(BDims) == sizeof...(Folds), bool, void>::type>(par, b, d, n, w);
  }
};
/**@}*/

/**
 * @defgroup Brick Brick data structure
 *
 * See <a href="structBrick_3_01Dim_3_01BDims_8_8_8_01_4_00_01Dim_3_01Folds_8_8_8_01_4_01_4.html">Brick< Dim< BDims... >, Dim< Folds... > ></a>
 *
 * @{
 */

/// Generic base template, see <a href="structBrick_3_01Dim_3_01BDims_8_8_8_01_4_00_01Dim_3_01Folds_8_8_8_01_4_01_4.html">Brick< Dim< BDims... >, Dim< Folds... > ></a>
template<typename BrickDims, typename VectorFold, bool isComplex = false, typename CommunicatingDims = CommDims<>>
struct Brick;

namespace 
{
  /**
   * @brief generic template for specialization
   * 
   * @tparam bdims the brick dimensions
   * @tparam folds the vector folds
   */
  template<typename bdims, typename folds, bool bdimsAndFoldsSameLength>
  struct _folds_divide_bdims;

  /**
   * @brief base case
   */
  template<>
  struct _folds_divide_bdims<Dim<>, Dim<>, true>
  {
    static constexpr bool value = true;
  };

  /**
   * @brief len(vfolds) != len(bdims) case
   */
  template<unsigned LastBDim, unsigned ... BDims, unsigned ... Folds>
  struct _folds_divide_bdims<Dim<LastBDim, BDims...>,
                             Dim<Folds...>,
                             false
                             >
  {
    static constexpr bool value = _folds_divide_bdims<Dim<BDims...>, Dim<Folds...>, sizeof...(BDims) == sizeof...(Folds)>::value;
  };

  /**
   * @brief len(vfolds) == len(bdims) case
   */
  template<unsigned LastBDim, unsigned LastVFold, unsigned ... BDims, unsigned ... Folds>
  struct _folds_divide_bdims<Dim<LastBDim, BDims...>,
                             Dim<LastVFold, Folds...>,
                             true
                             >
  {
    static constexpr bool value = (LastBDim % LastVFold == 0) && _folds_divide_bdims<Dim<BDims...>, Dim<Folds...>, true>::value;
  };
}

/**
 * @brief Brick data structure
 * @tparam isComplex (default false) true if the elements are complex.
 *                   In this case, note that complex bricks are represented
 *                   as an "array-of-structs", i.e. the real-part and imaginary
 *                   part of each complex element are adjacent in memory.
 * @tparam BDims The brick dimensions
 * @tparam Folds The fold dimensions
 *
 * Some example usage:
 * @code{.cpp}
 * Brick<Dim<8,8,8>, Dim<2,4>> bIn(&bInfo, bStorage, 0); // 8x8x8 bricks with 2x4 folding
 * bIn[1][0][0][0] = 2; // Setting the first element for the brick at index 1 to 2
 * @endcode
 */
template<
    bool isComplex,
    unsigned ... BDims,
    unsigned ... Folds,
    bool ... CommInDim>
struct Brick<Dim<BDims...>, Dim<Folds...>, isComplex, CommDims<CommInDim...> > {
  // make sure vector fold dimensions divide block dimensions
  static_assert(_folds_divide_bdims<Dim<BDims...>, Dim<Folds...>, sizeof...(BDims) == sizeof...(Folds)>::value,
                "Vector folds do not divide corresponding brick dimensions.");

  typedef Brick<Dim<BDims...>, Dim<Folds...>, isComplex, CommDims<CommInDim...> > mytype;    ///< Shorthand for this struct's type
  typedef CommDims<CommInDim...> myCommDims;    ///< Shorthand for information about which dimensions communicate
  typedef BrickInfo<sizeof...(BDims), myCommDims> myBrickInfo;        ///< Shorthand for type of the metadata
  typedef typename std::conditional<isComplex, bComplexElem, bElem>::type elemType; ///< the type of elements in this brick
  typedef typename std::conditional<isComplex, std::complex<bElem>, bElem>::type stdElemType; ///< STL-compatible type of elements

  static constexpr unsigned VECLEN = cal_size<Folds...>::value;     ///< Vector length shorthand
  static constexpr unsigned BRICKSIZE = cal_size<BDims...>::value * (isComplex ? 2 : 1);  ///< Brick size shorthand
  static constexpr bool complex = isComplex;  ///< True iff the elements of this brick are complex

  myBrickInfo *bInfo;        ///< Pointer to (possibly shared) metadata
  size_t step;             ///< Spacing between bricks in unit of elemType
  elemType *dat;                ///< Offsetted memory (BrickStorage)
  BrickStorage bStorage;

  /// Indexing operator returns: @ref _BrickAccessor
  FORCUDA
  inline _BrickAccessor<mytype, Dim<BDims...>, Dim<Folds...>,
      typename std::conditional<sizeof...(BDims) == sizeof...(Folds), bool, void>::type> operator[](unsigned b) {
    return _BrickAccessor<mytype, Dim<BDims...>, Dim<Folds...>,
        typename std::conditional<sizeof...(BDims) == sizeof...(Folds), bool, void>::type>(this, b, 0, 0, 0);
  }

  /// Return the adjacency list of brick *b*
  template<unsigned ... Offsets>
  FORCUDA
  inline elemType *neighbor(unsigned b) {
    constexpr unsigned numCommDims = CommDims<CommInDim...>::numCommunicatingDims(sizeof...(BDims));
    unsigned off = cal_offs<numCommDims, Offsets...>::value;
    return &dat[bInfo->adj[b][off] * step];
  }

  /**
   * @brief Initialize a brick data structure
   * @param bInfo Pointer to metadata
   * @param bStorage Brick storage (memory region)
   * @param offset Offset within the brick storage in number of real elements
   *               (i.e. each complex element counts as 2),
   *                eg. is a multiple of 512 for 8x8x8 real bricks,
   *                       a multiple of 1024 for 8x8x8 complex bricks
   */
  Brick(myBrickInfo *bInfo, const BrickStorage &brickStorage, unsigned offset) : bInfo(bInfo) {
    bStorage = brickStorage;
    dat = reinterpret_cast<elemType*>(bStorage.dat.get() + offset);
    assert(bStorage.step % 2 == 0);
    step = (unsigned) bStorage.step / (isComplex ? 2 : 1);
  }
};
/**@}*/

#endif //BRICK_H
