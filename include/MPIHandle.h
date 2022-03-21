//
// Created by Benjamin Sepanski on 3/8/22.
//

#ifndef BRICK_MPIHANDLE_H
#define BRICK_MPIHANDLE_H

#include "Array.h"
#include "array-mpi.h"
#include "brick.h"
#include "brick-mpi.h"
#include "mpi-cuda-util.h"

/**
 * @brief check for MPI failure
 *
 * @param return_value the value returned from an MPI call
 * @param func name of the MPI function being invoked
 * @param filename filename of call site
 * @param line line number of call site
 */
inline void _check_MPI(int return_value, const char *func, const char *filename,
                       const int line) {
  if (return_value != MPI_SUCCESS) {
    char error_msg[MPI_MAX_ERROR_STRING + 1];
    int error_length;
    std::ostringstream error_stream;
    error_stream << "MPI Error during call " << func << " " << filename << ":"
                 << line << std::endl;
    if (MPI_Error_string(return_value, error_msg, &error_length) !=
        MPI_SUCCESS) {
      error_stream << "Invalid argument passed to MPI_Error_string"
                   << std::endl;
    } else {
      error_stream << error_msg << std::endl;
    }
    throw std::runtime_error(error_stream.str());
  }
}

#ifndef NDEBUG
#define check_MPI(x) _check_MPI(x, #x, __FILE__, __LINE__)
#else
#define check_MPI(x) x
#endif

namespace brick {

/**
 * Basic MPI operations on a cartesian grid
 * @tparam Rank the rank of the cartesian coordinate system MPI ranks are laid out in
 * @tparam CommunicatingDims the dimensions in which communication is occurring
 */
template <unsigned Rank, typename CommunicatingDims = CommDims<>> class MPIHandle;

/**
 * A handle for the MPI types used by MPILayout to exchange arrays
 * @see MPILayout
 */
class MPIArrayTypesHandle {
private:
  template <unsigned Rank, typename CommunicatingDims>
  friend class brick::MPIHandle;
  std::unordered_map<uint64_t, MPI_Datatype> stypemap;
  std::unordered_map<uint64_t, MPI_Datatype> rtypemap;
  std::vector<long> extentWithoutGhost{}, ghostExtent{}, paddingExtent{};
};

/**
 * @see MPIHandle
 * @tparam CommInDim false in the dimensions in which no communication occurs, true otherwise
 */
template <unsigned Rank, bool... CommInDim> class MPIHandle<Rank, CommDims<CommInDim...>> {
  /// constexprs/typedefs
public:
  // public constexprs/typedefs
  using CommunicatingDims = CommDims<CommInDim...>;
  static constexpr unsigned RANK = Rank;

  /// members
private:
  // private members
  int rank, size; // MPI Rank/size
public:
  // public members
  MPI_Comm comm;
  std::unordered_map<uint64_t, int> rank_map;    ///< Mapping from neighbor to each neighbor's rank

  /// methods
private:
  // private methods
  template<typename ArrType>
  void validateExtentsAndGhost(const ArrType &a, const std::array<unsigned, RANK> &ghost) {
    for (unsigned d = 0; d < RANK; ++d) {
      if(ghost[d] > 0 && !CommunicatingDims::communicatesInDim(d)) {
        throw std::runtime_error("Ghost dimension must be zero in dimensions which do not communicate");
      }
      if (a.extent[d] < 3 * ghost[d]) {
        throw std::runtime_error("Extents in ghostExtent-dimension must be at least 3 * ghostExtent size");
      }
    }
  }

public:
  // public methods
  /**
   * Create an MPI Handle on the provided cartesian communicator
   * @param cartesianComm a cartesian communicator in a space of ambient dimension Rank
   */
  MPIHandle(MPI_Comm cartesianComm) : comm(cartesianComm) {
    // get rank, size
    check_MPI(MPI_Comm_rank(cartesianComm, &rank));
    check_MPI(MPI_Comm_size(cartesianComm, &size));
    std::array<int, Rank> coordsOfProc{};
    check_MPI(MPI_Cart_coords(cartesianComm, rank, Rank, coordsOfProc.data()));
    populate<Rank, CommunicatingDims>(cartesianComm, this->rank_map, 0, 1, coordsOfProc.data());
  }

  /**
   * @return a reference to the cartesian communicator
   */
  MPI_Comm & getMPIComm() {
    return this->comm;
  }

  inline int myRank() const {
    return this->rank;
  }

  inline int commSize() const {
    return this->size;
  }

  /**
   * Exchange arrays
   * @tparam DataType data-type of the array
   * @tparam ArrPadding the padding of the array
   * @tparam ArrSizeType the size-type of the array
   * @tparam ArrIndexType the index-type of the array
   * @param arr the array to exchange
   * @param ghost the number of ghostExtent elements in each dimension
   */
  template <typename DataType, typename ArrPadding, typename ArrSizeType,
            typename ArrIndexType>
  void exchangeArray(
      brick::Array<DataType, RANK, ArrPadding, ArrSizeType, ArrIndexType> &arr,
      const std::array<unsigned, RANK> &ghost) {
    validateExtentsAndGhost(arr, ghost);
    std::vector<long> padding, extentWithoutGhost, ghostAsVector;
    padding.reserve(RANK);
    for (unsigned d = 0; d < RANK; ++d) {
      padding.push_back(arr.PADDING(d));
      extentWithoutGhost.push_back(arr.extent[d] - 2 * ghost[d]);
      ghostAsVector.push_back(ghost[d]);
    }
    exchangeArr<RANK>(arr.getData().get(), this->comm,
                      this->rank_map, extentWithoutGhost, padding,
                      ghostAsVector);
  }

  /**
   * Build a handle for exchanging ghostExtent-elements of arrays
   * using MPI Types.
   *
   * @tparam DataType data type of the array
   * @tparam ArrPadding array padding
   * @tparam ArrSizeType array size type
   * @tparam ArrIndexType array index type
   * @param arr the array
   * @param ghost the number of ghostExtent elements in each dimension
   * @return the handle
   * @see exchangeArray
   */
  template <typename DataType, typename ArrPadding, typename ArrSizeType,
            typename ArrIndexType>
  MPIArrayTypesHandle buildArrayTypesHandle(
      Array<DataType, RANK, ArrPadding, ArrSizeType, ArrIndexType> arr,
      std::array<unsigned, RANK> &ghost) {
    validateExtentsAndGhost(arr, ghost);

    MPIArrayTypesHandle handle;
    for (unsigned d = 0; d < RANK; ++d) {
      handle.extentWithoutGhost.push_back(arr.extent[d] - 2 * ghost[d]);
      handle.ghostExtent.push_back(ghost[d]);
      handle.paddingExtent.push_back(arr.PADDING(d));
    }
    exchangeArrPrepareTypes<RANK, DataType>(handle.stypemap,
                                            handle.rtypemap,
                                            handle.extentWithoutGhost,
                                            handle.paddingExtent,
                                            handle.ghostExtent);
    return handle;
  }

  /**
   * Exchange arrays using the provided handle
   * @tparam DataType data-type of the array
   * @tparam ArrPadding the padding of the array
   * @tparam ArrSizeType the size-type of the array
   * @tparam ArrIndexType the index-type of the array
   * @param arr the array to exchange
   * @param handle the handle describing the array types
   * @see buildArrayTypesHandle
   */
  template <typename DataType, typename ArrPadding, typename ArrSizeType,
            typename ArrIndexType>
  void exchangeArray(
      Array<DataType, RANK, ArrPadding, ArrSizeType, ArrIndexType> &arr,
      MPIArrayTypesHandle &handle) {
    std::vector<long> padding;
    padding.reserve(RANK);
    for (unsigned d = 0; d < RANK; ++d) {
      padding.push_back(arr.PADDING(d));
      if(arr.extent[d] != handle.extentWithoutGhost[d] + 2 * handle.ghostExtent[d]) {
        throw std::runtime_error("Array extent does not match handle extent");
      }
      if(arr.PADDING(d) != handle.paddingExtent[d]) {
        throw std::runtime_error("Array padding does not match handle padding");
      }
    }
    exchangeArrTypes<RANK>(arr.getData().get(),
                           this->comm,
                           this->rank_map,
                           handle.stypemap,
                           handle.rtypemap);
  }
};
} // end namespace brick

#endif // BRICK_MPIHANDLE_H
