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
  static constexpr unsigned value = static_power<base, exp / 2>::value
                                  * static_power<base, exp - (exp / 2)>::value;
};

/// Return base @ref static_power
template<unsigned base>
struct static_power<base, 1> {
  static constexpr unsigned value = base;
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
struct CommDims;

// TODO : docs
template<>
struct CommDims<>
{
  constexpr static bool communicatesInDim(unsigned dim) {return true;}
  constexpr static unsigned numCommunicatingDims(unsigned numDims) {return numDims;}
};

// TODO : docs
template<bool CommInLastDim, bool ... CommInDim>
struct CommDims<CommInLastDim, CommInDim...>
{
  private:
    constexpr static unsigned length = sizeof...(CommInDim) + 1;

  public:
    // TODO: DOC
    constexpr static bool communicatesInDim(unsigned dim)
    {
      return (dim < length)
        ? (dim == length - 1 ? CommInLastDim : CommDims<CommInDim...>::communicatesInDim(dim))
        : true;
    }

    // TODO: DOC
    constexpr static unsigned numCommunicatingDims(unsigned numDims) {
      return (numDims >= length ? communicatesInDim(length - 1) : 0)
             + CommDims<CommInDim...>::numCommunicatingDims(numDims >= length ? numDims - 1 : numDims);
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
  private:
    static constexpr unsigned dims[sizeof...(Ds)] = { Ds... };

  public:
  // get *d*th entry (from right), e.g. Dim<1,2,3>::template get<0>() == 3
  template<unsigned d>
  static constexpr FORCUDA unsigned get()
  {
    static_assert(d < sizeof...(Ds), "d out of range");
    return dims[sizeof...(Ds) - 1 - d];
  }

  // get, or return default if out of range. In range case:
  template<unsigned d>
  static constexpr FORCUDA
  typename std::enable_if<d < sizeof...(Ds), unsigned>::type
  getOrDefault(unsigned default_value)
  {
    return get<d>();
  }

  // get, or return default if out of range. Out of range case:
  template<unsigned d>
  static constexpr FORCUDA
  typename std::enable_if<d >= sizeof...(Ds), unsigned>::type
  getOrDefault(unsigned default_value)
  {
    return default_value;
  }

  // get product of first *d* entries (from the right)
  // e.g. Dim<1,2,3>::product(0) == 1, Dim<1,2,3>::product(1) == 3,
  //      Dim<1,2,3>::product(2) == 6
  template<unsigned d>
  static constexpr FORCUDA
  typename std::enable_if<d != 0, unsigned>::type product() ///< don't use this implementation if d == 0
  {
    static_assert(d <= sizeof...(Ds), "d out of range");
    return dims[sizeof...(Ds) - 1 - (d-1)] * product<d-1>();
  }

  // explicit specialization to avoid annoying compiler warning
  template<unsigned d>
  static constexpr FORCUDA
  typename std::enable_if<d == 0, unsigned>::type product() {return 1;}
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

// TODO:DOCS
// declaration for indexing
template<typename ... T>
struct BrickIndex;

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

// TODO: Docs
// some utility classes for BrickIndex
namespace  // anonymous namespace
{
  // take static conjunciton of bools
  template<bool ... Bs> struct Conjunction; 
  template<> struct Conjunction<> { static constexpr bool value = true; };
  template<bool b, bool ... Bs> struct Conjunction<b, Bs...> { static constexpr bool value = b && Conjunction<Bs...>::value; };

  // Leverage SFINAE. void if all keys are true, substitution failure otherwise.
  template <bool ... keys>
  using voidIfMatch = typename std::enable_if<Conjunction<keys...>::value>::type;

  // value is true iff all types in parameter pack match T
  template<typename Target, typename ... T>
  struct CanConvertTo 
  { 
    static constexpr bool value = std::is_convertible<typename std::common_type<T...>::type, Target>::value;
  };

  template<typename Target>
  struct CanConvertTo<Target> { static constexpr bool value = true; };
} // end anonymous space

// TODO: Docs
template<
    bool isComplex,
    unsigned ... BDims,
    unsigned ... Folds,
    bool ... CommInDim>
struct BrickIndex<Brick<Dim<BDims...>, Dim<Folds...>, isComplex, CommDims<CommInDim...> > >
{
  // usefeul types
  typedef Brick<Dim<BDims...>, Dim<Folds...>, isComplex, CommDims<CommInDim...> > myBrickType;
  typedef typename myBrickType::elemType elemType;
  typedef CommDims<CommInDim...> myCommDims;
  typedef Dim<BDims...> myBDims;
  typedef Dim<Folds...> myFolds;

  unsigned indexInNbrList; ///< ternary representation: left=0, middle=1, right=2
  int indexOfVec; ///< Index of vector in brick (signed for intermediate computations)
  int indexInVec; ///< Index inside vector (signed for intermediate computations)

  // TODO: doc
  template<typename ... IndexType>
  FORCUDA inline
  BrickIndex(IndexType ... indices)
  {
    static_assert(CanConvertTo<int, IndexType...>::value, "indices must be integer type");
    static_assert(sizeof...(IndexType) == sizeof...(BDims), "Number of indices must match number of brick dimensions");

    constexpr unsigned NUM_COMM_DIMS = myCommDims::numCommunicatingDims(sizeof...(BDims));
    // // initialize \sum_{i=0}^{d-1}3**i == (3^d - 1) / 2
    this->indexInNbrList = (static_power<3, NUM_COMM_DIMS>::value - 1) / 2;
    this->indexOfVec = 0;
    this->indexInVec = 0;
    this->shift(indices...);
  }

  // TODO: doc
  template<unsigned ... dimsToShift, typename ... IndexType>
  FORCUDA inline
  void shiftInDims(IndexType ... shifts)
  {
    static_assert(sizeof...(dimsToShift) <= sizeof...(BDims), "use shift() to shift in all dimensions");
    static_assert(CanConvertTo<int, IndexType...>::value, "shifts must be of integral type");
    shift_some_dims_recurse<dimsToShift...>(shifts...);
  }

  // TODO: doc
  template <typename ... IndexType>
  FORCUDA inline
  voidIfMatch<sizeof...(IndexType) == sizeof...(BDims), CanConvertTo<int, IndexType...>::value> // void if matches
  shift(IndexType ... shifts)
  {
    shift_all_dims_recurse(shifts...);
  }

  // TODO: Doc
  FORCUDA inline
  unsigned getIndexInBrick() const
  {
    return indexOfVec * myBrickType::VECLEN + indexInVec;
  }

  private:
    // TODO: Docs
    template<unsigned ... dimsToShift>
    FORCUDA inline
    void shift_some_dims_recurse() {}

    // TODO: Docs
    template<unsigned nextDim, unsigned ... dimsToShift, typename ... IndexType>
    FORCUDA inline
    void shift_some_dims_recurse(int shift, IndexType ... remainingShifts)
    {
      static_assert(CanConvertTo<int, IndexType...>::value, "shifts must be of integral type.");
      static_assert(sizeof...(dimsToShift) == sizeof...(IndexType), "Number of shifts must match number of dims to shift");
      performShiftInDim<nextDim>(shift);
      shift_some_dims_recurse<dimsToShift...>(remainingShifts...);
    }

    // TODO: Docs
    FORCUDA inline
    void shift_all_dims_recurse() {}

    // TODO: Docs
    template<typename ... IndexType>
    FORCUDA inline
    voidIfMatch<CanConvertTo<int, IndexType...>::value> // all index types must be implicitly convertible to int
    shift_all_dims_recurse(int shift, IndexType ... remainingShifts) 
    {
      this->performShiftInDim<sizeof...(IndexType)>(shift);
      shift_all_dims_recurse(remainingShifts...);
    }

    // todo: doc, FOLD case
    template<unsigned d>
    FORCUDA inline
    voidIfMatch<myFolds::template getOrDefault<d>(1) != 1>
    performShiftInDim(int shift)
    {
      static_assert(d < sizeof...(BDims), "Shifting dimension must be less than the number of dimensions");
      constexpr int VECTOR_FOLD = myFolds::template getOrDefault<d>(1);
      constexpr int foldIndex = std::min(d, (unsigned) sizeof...(Folds));
      constexpr int STRIDE_IN_VECTOR = myFolds::template product<foldIndex>();
      constexpr int BRICK_DIM = myBDims::template get<d>();
      constexpr int STRIDE_IN_BRICK = myBDims::template product<d>();
      static_assert(STRIDE_IN_BRICK % STRIDE_IN_VECTOR == 0);
      constexpr int STRIDE_OVER_VECTORS = STRIDE_IN_BRICK / STRIDE_IN_VECTOR;
      static_assert(BRICK_DIM % VECTOR_FOLD == 0);
      constexpr int DIM_OVER_VECTORS = BRICK_DIM / VECTOR_FOLD;
      // take max to avoid large template instantiation of static_power
      constexpr unsigned exp = myCommDims::numCommunicatingDims(d+1) > 0 
                              ? myCommDims::numCommunicatingDims(d+1) - 1
                              : 0;
      constexpr int NEIGHBOR_WEIGHT = myCommDims::communicatesInDim(d) ? static_power<3, exp>::value : 0;

      // get *d*th component of index inside/outside vector and shift it
      int indexInVec_d = (indexInVec / STRIDE_IN_VECTOR) % VECTOR_FOLD;
      int shiftedIndexInVec_d = indexInVec_d + shift;
      // handle case where we shifted outside the vector
      // (ASSUME shifted no more than 3 bricks (L neighbor to R neighbor))
      int vectorShift = (shiftedIndexInVec_d + 3 * BRICK_DIM) / VECTOR_FOLD - 3 * (BRICK_DIM / VECTOR_FOLD);
      shiftedIndexInVec_d = (shiftedIndexInVec_d + 3 * BRICK_DIM) % VECTOR_FOLD;
      // add back into indexInVec
      indexInVec += (shiftedIndexInVec_d - indexInVec_d) * STRIDE_IN_VECTOR;

      // get *d*th component of index of vector and shift it
      int indexOfVec_d = (indexOfVec / STRIDE_OVER_VECTORS) % DIM_OVER_VECTORS;
      int shiftedIndexOfVec_d = indexOfVec_d + vectorShift;
      // Handle possibility that we shifted into a neighboring brick
      // (ASSUME shifted no more than 3 bricks (L neighbor to R neighbor))
      int brickShift = (shiftedIndexOfVec_d + 3 * DIM_OVER_VECTORS) / DIM_OVER_VECTORS - 3;
      shiftedIndexOfVec_d = (shiftedIndexOfVec_d + 3 * DIM_OVER_VECTORS) % DIM_OVER_VECTORS;
      // add back into indexOfVec
      indexOfVec += (shiftedIndexOfVec_d - indexOfVec_d) * STRIDE_OVER_VECTORS;

      // adjust brick (assumes stays within L neighbor, center, R neighbor)
      indexInNbrList += brickShift * NEIGHBOR_WEIGHT;
    }

    // todo: doc, NO FOLD case
    template<unsigned d>
    FORCUDA inline
    voidIfMatch<myFolds::template getOrDefault<d>(1) == 1>
    performShiftInDim(int shift)
    {
      static_assert(d < sizeof...(BDims), "Shifting dimension must be less than the number of dimensions");
      // compute stride
      constexpr int foldIndex = std::min(d, (unsigned) sizeof...(Folds));
      constexpr int STRIDE_IN_VECTOR = myFolds::template product<foldIndex>();
      constexpr int STRIDE_IN_BRICK = myBDims::template product<d>();
      static_assert(STRIDE_IN_BRICK % STRIDE_IN_VECTOR == 0);
      constexpr int STRIDE = STRIDE_IN_BRICK / STRIDE_IN_VECTOR;
      // extent in this dimension
      constexpr int BRICK_DIM = myBDims::template get<d>();
      // take max to avoid large template instantiation of static_power
      constexpr unsigned exp = myCommDims::numCommunicatingDims(d+1) > 0 
                              ? myCommDims::numCommunicatingDims(d+1) - 1
                              : 0;
      constexpr int NEIGHBOR_WEIGHT = myCommDims::communicatesInDim(d) ? static_power<3, exp>::value : 0;

      // get *d*th component of index of vector and shift it
      int indexOfVec_d = (indexOfVec / STRIDE) % BRICK_DIM;
      int shiftedIndexOfVec_d = indexOfVec_d + shift;
      // Handle possibility that we shifted into a neighboring brick
      // (ASSUME shifted no more than 3 bricks (L neighbor to R neighbor))
      int brickShift = (shiftedIndexOfVec_d + 3 * BRICK_DIM) / BRICK_DIM - 3;
      shiftedIndexOfVec_d = (shiftedIndexOfVec_d + 3 * BRICK_DIM) % BRICK_DIM;
      // add back into indexOfVec
      indexOfVec += (shiftedIndexOfVec_d - indexOfVec_d) * STRIDE;

      // adjust brick (assumes stays within L neighbor, center, R neighbor)
      indexInNbrList += brickShift * NEIGHBOR_WEIGHT;
    }
};

#endif // BRICK_H