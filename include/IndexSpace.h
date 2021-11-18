//
// Created by Ben_Sepanski on 11/15/2021.
//

#ifndef BRICK_INDEXSPACE_H
#define BRICK_INDEXSPACE_H

#include <tuple>

#include "template-utils.h"

namespace brick {

/**
 * A Rank-dimensional index
 * @tparam Rank the rank of the space
 * @tparam IndexType the type to use for indices
 */
template <unsigned Rank, typename IndexType> class Index {
private:
  static_assert(Rank > 0, "Rank must be positive");
  IndexType indices[Rank] = {};

  /**
   * Generic template for access to an axis known at compile-time
   * @tparam Axis the axis
   * @tparam AxisIsInBounds true iff the axis is inbounds
   * @see CompileTimeAccessor
   */
  template<unsigned Axis, bool AxisIsInBounds = Axis < Rank>
  struct CompileTimeAccessor;

  /**
   * Axis in-bounds case
   * @see CompileTimeAccessor
   */
  template<unsigned Axis>
  struct CompileTimeAccessor<Axis, true> {
    static inline IndexType& access(Index & idx) { return idx[Axis]; }
    static inline const IndexType& access(const Index & idx) { return idx.at(Axis); }
  };

  /**
    * Axis out-of-bounds case
    * @see CompileTimeAccessor
    */
  template<unsigned Axis>
  struct CompileTimeAccessor<Axis, false> {
    static inline IndexType& access(Index & idx) {
      std::stringstream ss;
      ss << "Axis " << Axis << " is not less than rank " << Rank;
      throw std::runtime_error(ss.str());
    }
    static inline const IndexType& access(const Index & idx) {
      std::stringstream ss;
      ss << "Axis " << Axis << " is not less than rank " << Rank;
      throw std::runtime_error(ss.str());
    }
  };

public:
  /**
   * Default constructor initializes all indices to 0
   */
  Index() = default;

  template <typename T> explicit Index(const std::array<T, Rank> indices) {
    static_assert(std::is_convertible<T, IndexType>::value,
                  "Type must be convertible to IndexType");
    for (unsigned d = 0; d < Rank; ++d) {
      this->indices[d] = indices[d];
    }
  }

  /**
   * @param that the index to test again
   * @return true iff that index is the same as this one
   */
  inline bool operator==(const Index &that) const {
    for (unsigned d = 0; d < Rank; ++d) {
      if (this->indices[d] != that.indices[d]) {
        return false;
      }
    }
    return true;
  }

  /**
   * Get reference to index in the specified dimension
   * @param dim the dimension 0 <= dim < Rank
   * @return the index
   */
  inline IndexType &operator[](unsigned dim) {
    assert(dim < Rank);
    return indices[dim];
  }

  /**
   * Get the index in the specified dimension
   * @param dim the dimension 0 <= dim < Rank
   * @return the index
   */
  inline const IndexType& at(unsigned dim) const {
    assert(dim < Rank);
    return indices[dim];
  }

#define INDEX_SPACE_ACCESS_NAMED_AXIS(AXIS_NAME, AXIS) \
  inline IndexType& AXIS_NAME() { \
    return CompileTimeAccessor<AXIS>::access(*this); \
  } \
  inline const IndexType& AXIS_NAME() const { \
    return CompileTimeAccessor<AXIS>::access(*this); \
  }

  INDEX_SPACE_ACCESS_NAMED_AXIS(i, 0)
  INDEX_SPACE_ACCESS_NAMED_AXIS(j, 1)
  INDEX_SPACE_ACCESS_NAMED_AXIS(k, 2)
  INDEX_SPACE_ACCESS_NAMED_AXIS(l, 3)
  INDEX_SPACE_ACCESS_NAMED_AXIS(m, 4)
  INDEX_SPACE_ACCESS_NAMED_AXIS(n, 5)

  /**
   * Print an index
   * @tparam Rank rank of the multi-dimensional space the index is in
   * @tparam IndexType type of the index values
   * @param out output stream
   * @param idx the index to print
   * @return the modified output stream
   */
  friend std::ostream& operator<<(std::ostream& out, const Index<Rank, IndexType> &idx) {
    out << "{";
    for(unsigned d = 0; d < Rank - 1; ++d) {
      out << idx.at(d) << ", ";
    }
    out << idx.at(Rank - 1) << "}";
    return out;
  }
};

template <typename T> struct Interval {
  T low, high;
  Interval(T low, T high) : low{low}, high{high} {
    if (low > high) {
      throw std::runtime_error("low > high");
    }
  }

  explicit Interval(T high) : low{0}, high{high} {
    if(low > high) {
      throw std::runtime_error("low > high");
    }
  }

  Interval() : low{0}, high{0} {}
};

/**
 * A column-major index space
 * @tparam Rank rank of the index space
 * @tparam IndexType type of the indices
 */
template <unsigned Rank, typename IndexType = unsigned> class IndexSpace {
private:
  Interval<IndexType> bounds[Rank];
public:
  /**
   * Create an index object with the given extents (0th index first)
   * @param indices the indices
   */
  explicit IndexSpace(std::array<IndexType, Rank> extents) {
    for(unsigned d = 0; d < Rank; ++d) {
      this->bounds[d] = Interval<IndexType>(extents[d]);
    }
  }

  /**
   * Create an index object with the given extents (0th index first)
   * @param indices the indices
   */
  explicit IndexSpace(const IndexType *extents) {
    for(unsigned d = 0; d < Rank; ++d) {
      this->bounds[d] = Interval<IndexType>(extents[d]);
    }
  }

  /**
   * Create an index object with the given bounds (0th index first)
   * @param indices the indices
   */
  explicit IndexSpace(std::array<Interval<IndexType>, Rank> bounds) {
    for(unsigned d = 0; d < Rank; ++d) {
      this->bounds[d] = bounds[d];
    }
  }

  /**
   * @param dim 0 <= dim < Rank the dim to get the bounds for
   */
  void getBounds(unsigned dim) {
    assert(dim < Rank);
    return bounds[dim];
  }

  /**
   * Generic template for specialization
   * @tparam Range0ToRank automatically supplied to build 0,...,Rank-1
   * @see Iterator
   */
  template <typename Range0ToRank =
                typename templateutils::UnsignedIndexSequence<Rank>::type>
  class Iterator;

  /**
   * Iterator through index space
   * @tparam Range0ToRank automatically supplied: 0,...,Rank-1
   */
  template <unsigned... Range0ToRank>
  class Iterator<templateutils::ParameterPackManipulator<unsigned>::Pack<
      Range0ToRank...>> {
  private:
    static_assert(
        std::is_same<
            templateutils::ParameterPackManipulator<unsigned>::Pack<
                Range0ToRank...>,
            typename templateutils::UnsignedIndexSequence<Rank>::type>::value,
        "User should not supply Range0ToRank template type");
    Index<Rank, IndexType> index;
    const Interval<IndexType> bounds[Rank];

  public:
    using iterator_category = std::random_access_iterator_tag;
    using difference_type = size_t;
    using value_type = Index<Rank, IndexType>;
    using pointer = value_type*;
    using reference = value_type&;

    /**
     * Build an iterator in the given space from the given startIndex
     * @param indexSpace the space
     * @param startIndex the start index
     */
    Iterator(const IndexSpace &indexSpace, Index<Rank, IndexType> startIndex)
        : bounds{indexSpace.bounds[Range0ToRank]...} {
      index = startIndex;
    }

    Iterator& operator=(const Iterator &that) {
      if(this != &that) {
        for(unsigned d = 0; d < Rank; ++d) {
          if(this->bounds[d] != that.bounds[d]) {
            throw std::runtime_error("bounds must match when using operator=");
          }
        }
        for(unsigned d = 0; d < Rank; ++d) {
          this->index.at(d) = that.index.at(d);
        }
      }
      return *this;
    }

    /**
     * Only checks that the indices are equal, not the bounds!
     * @param that the other iterator
     * @return true iff the indices are equal
     */
    inline bool operator==(const Iterator &that) const {
      return this->index == that.index;
    }

    inline bool operator!=(const Iterator &that) const {
      return !this->operator==(that);
    }

    inline reference operator*() {
      return index;
    }

    inline pointer operator->() {
      return &index;
    }

    // Prefix increment
    inline Iterator& operator++() {
#pragma unroll
      for(unsigned d = 0; d < Rank; ++d) {
        this->index[d]++;
        if(this->index[d] >= this->bounds[d].high && d < Rank - 1) {
          this->index[d] = this->bounds[d].low;
        }
        else {
          break;
        }
      }
      return *this;
    }

    // Postfix Increment
    inline Iterator operator++(int) { // NOLINT(cert-dcl21-cpp)
      const Iterator tmp = *this;
      ++(*this);
      return tmp;
    }

    // Prefix decrement
    inline Iterator& operator--() {
#pragma unroll
      for(unsigned d = 0; d < Rank; ++d) {
        if(this->index[d] <= this->bounds[d].low && d < Rank - 1) {
          this->index[d] = this->bounds[d].high - 1;
        }
        else {
          --this->index[d];
          break;
        }
      }
      return *this;
    }

    // Postfix Decrement
    inline Iterator operator--(int) { // NOLINT(cert-dcl21-cpp)
      Iterator tmp = *this;
      --(*this);
      return tmp;
    }

    inline Iterator operator+(size_t n) const {
      if(n < 0) {
        return *this - n;
      }
      Iterator incremented{*this};
      unsigned d = 0;
      while(n > 0 && d < Rank) {
        IndexType low = incremented.bounds[d].low;
        IndexType high = incremented.bounds[d].high;
        incremented.index[d] += n;
        IndexType offset = incremented.index[d] - low;
        n = offset / (high - low);
        if(d < Rank - 1) {
          incremented.index[d] = offset % (high - low) + low;
        } else {
          incremented.index[d] = offset + low;
        }
        d++;
      }
      return incremented;
    }

    inline Iterator operator-(size_t n) const {
      Iterator decremented{*this};
      unsigned d = 0;
      IndexType low = decremented.bounds[d].low;
      IndexType high = decremented.bounds[d].high;
      IndexType indexInDim = decremented.index[d];
      while(d < Rank && low + n > indexInDim) {
        n -= (indexInDim - low);
        if(n % (high - low) == 0) {
          decremented.index[d] = low;
          n /= (high - low);
        } else {
          decremented.index[d] = ((high - low) - n % (high - low)) + low;
          n /= (high - low);
          n++;
        }

        d++;
        if(d < Rank) {
          low = decremented.bounds[d].low;
          high = decremented.bounds[d].high;
          indexInDim = decremented.index[d];
        }
      }
      decremented.index[d] -= n;
      return decremented;
    }

    inline Iterator operator+=(size_t n) {
      return (*this = *this + n);
    }

    inline Iterator operator-=(size_t n) {
      return (*this = *this - n);
    }
  };


  typedef Iterator<> const_iterator;

  const_iterator begin() const {
    std::array<IndexType, Rank> lows{};
    for(unsigned d = 0; d < Rank; ++d) {
      lows[d] = bounds[d].low;
    }
    return Iterator<>(*this, Index<Rank, IndexType>(lows));
  }

  const_iterator end() const {
    std::array<IndexType, Rank> highs{};
    for(unsigned d = 0; d < Rank - 1; ++d) {
      highs[d] = bounds[d].low;
    }
    highs[Rank - 1] = bounds[Rank - 1].high;
    return Iterator<>(*this, Index<Rank, IndexType>(highs));
  }
};


} // end namespace brick

#endif // BRICK_INDEXSPACE_H
