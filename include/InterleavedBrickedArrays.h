//
// Created by Ben_Sepanski on 11/8/2021.
//

#ifndef BRICK_INTERLEAVEDBRICKEDARRAYS_H
#define BRICK_INTERLEAVEDBRICKEDARRAYS_H

#include "BrickedArray.h"

namespace brick {

template<typename BrickDims,
          typename ... >
struct InterleavedBrickedArrays;

template<typename DataType, typename VectorFold = Dim<1> >
struct DataTypeVectorFoldPair {
  typedef DataType dataType;
  typedef VectorFold vectorFold;
};

template<unsigned ... BDims,
          typename ... VectorFolds,
          typename ... DataTypes>
struct InterleavedBrickedArrays<Dim<BDims...>,
                                DataTypeVectorFoldPair<DataTypes, VectorFolds>...>
{
  /// constexpr/typedefs
private:
  // private constexprs
  static constexpr unsigned NUM_ELEMENTS_PER_BRICK =
      templateutils::reduce(templateutils::multiply<unsigned>, 1, BDims...);
public:
  // public constexpr
  static constexpr unsigned Rank = sizeof...(BDims);
  /// Members
public:
  std::tuple<
      std::vector<BrickedArray<DataTypes, Dim<BDims...>, VectorFolds> >...
      > fields;
  /// static methods
private:
  // private static methods
  /**
       * Base case for get offsets
   */
  template<typename ...>
  static void computeOffsets(std::vector<size_t> &offsets) {  }

  /**
       * Compute the offsets into brick storage for the provided number
       * of interleaved fields of each data type.
   */
  template<typename HeadDataType, typename ... TailDataTypes, typename ... T>
  static void computeOffsets(std::vector<size_t> &offsets, unsigned headCount, T ... tailCounts) {
    static_assert(sizeof...(T) == sizeof...(TailDataTypes),
                  "Mismatch in number of arguments");
    static_assert(sizeof(HeadDataType) % sizeof(bElem) == 0,
                  "sizeof(bElem) must divide sizeof(DataType)");
    assert(!offsets.empty());
    size_t lastOffset = offsets.back();
    for(unsigned i = 0; i < headCount; ++i) {
      lastOffset += NUM_ELEMENTS_PER_BRICK * sizeof(HeadDataType) / sizeof(bElem);
      offsets.push_back(lastOffset);
    }
    computeOffsets<TailDataTypes...>(offsets, tailCounts...);
  }

  /// Methods
private:
  // private methods
  /**
       * Base case
       * @see initializeBrickedArrays
   */
  template<typename ...>
  void initializeBrickedArrays(const BrickLayout<Rank> &,
                               const brick::ManagedBrickStorage &,
                               std::vector<size_t>::const_iterator ) { }

  /**
       * initialize the bricked arrays
       * @tparam HeadDataTypeVectorFoldPair data type/vector fold pair of current array type
       * @tparam Tail remaining data type/vector fold pairs
       * @tparam T parameter pack of counts
       * @param layout the brick layout
       * @param storage the storage to use
       * @param offset the iterator over offsets
       * @param headCount number of fields of the head type to make
       * @param tailCounts number of fields of the tail types to make
   */
  template<typename HeadDataTypeVectorFoldPair, typename ... Tail, typename ... T>
  void initializeBrickedArrays(const BrickLayout<Rank> &layout,
                               const brick::ManagedBrickStorage &storage,
                               std::vector<size_t>::const_iterator offset,
                               unsigned headCount, T ... tailCounts) {
    typedef typename HeadDataTypeVectorFoldPair::dataType d;
    typedef typename HeadDataTypeVectorFoldPair::vectorFold vf;
    typedef BrickedArray<d, Dim<BDims...>, vf> BrickedArrayType;
    for(unsigned i = 0; i < headCount; ++i) {
      std::get<sizeof...(DataTypes) - 1 - sizeof...(Tail)>(fields).push_back(BrickedArrayType(layout, storage, *(offset++)));
    }
    initializeBrickedArrays<Tail...>(layout, storage, offset, tailCounts...);
  }

public:
  // public methods
  /**
       *
       * @tparam T the types of the count arguments
       * @param layout the layout to use
       * @param fieldCount the number of fields of each data type/vector fold
       *                   pair to build
   */
  template<typename ... T>
  explicit InterleavedBrickedArrays(BrickLayout<Rank> layout, T ... fieldCount) {
    static_assert(sizeof...(T) == sizeof...(DataTypes),
                  "Must provide field counts for each DataType");
    std::vector<size_t> offsets = {0};
    computeOffsets<DataTypes...>(offsets, fieldCount...);
    size_t step = offsets.back();
    offsets.pop_back();
    brick::ManagedBrickStorage storage(layout.size(), step);
    initializeBrickedArrays<DataTypeVectorFoldPair<DataTypes, VectorFolds>...>(layout, storage, offsets.cbegin(), fieldCount...);
  }

  // TODO: Write MMAP version of constructor
};
} // end namespace brick

#endif // BRICK_INTERLEAVEDBRICKEDARRAYS_H
