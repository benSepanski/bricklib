/**
 * @file
 * @brief Reference MPI communication with arrays
 *
 * This includes packing/unpacking and communication with MPI_Types
 */

#ifndef BRICK_ARRAY_MPI_H
#define BRICK_ARRAY_MPI_H

#include "brick-mpi.h"
#include <mpi.h>

/**
 * OpenMP-enabled copy kernel
 * @param dst destination
 * @param src source
 * @param size in number of bElem
 */
template<typename elemType>
inline void elemcpy(elemType *dst, const elemType *src, unsigned long size) {
#pragma omp simd
  for (unsigned long i = 0; i < size; ++i) {
    dst[i] = src[i];
  }
}

template<unsigned dim, typename elemType>
inline typename std::enable_if<dim != 1, elemType*>::type 
pack(elemType *arr, BitSet neighbor, elemType *buffer_out, const std::vector<unsigned long> &arrstride,
     const std::vector<long> &dimlist, const std::vector<long> &padding, const std::vector<long> &ghost) {
  // Inner region
  long sec = 0;
  long st = 0;
  int d = dim - 1;
  if (neighbor.get(dim)) {
    sec = 1;
    st = padding[d] + dimlist[d];
  } else if (neighbor.get(-(int) dim)) {
    sec = -1;
    st = padding[d] + ghost[d];
  }
  if (sec) {
    for (unsigned i = 0; i < ghost[d]; ++i)
      buffer_out = pack<dim - 1>(arr + arrstride[d] * (st + i), neighbor, buffer_out,
                                 arrstride, dimlist, padding, ghost);
  } else {
    for (unsigned i = 0; i < dimlist[d]; ++i)
      buffer_out = pack<dim - 1>(arr + arrstride[d] * (padding[d] + ghost[d] + i), neighbor, buffer_out,
                                 arrstride, dimlist, padding, ghost);
  }

  return buffer_out;
}

template<unsigned dim, typename elemType>
inline typename std::enable_if<dim == 1, elemType*>::type 
pack(elemType *arr, BitSet neighbor, elemType *buffer_out, const std::vector<unsigned long> &arrstride,
     const std::vector<long> &dimlist, const std::vector<long> &padding,
     const std::vector<long> &ghost) {
  // Inner region
  long sec = 0;
  long st = 0;
  int d = 0;
  if (neighbor.get(1)) {
    sec = 1;
    st = padding[d] + dimlist[d];
  } else if (neighbor.get(-1)) {
    sec = -1;
    st = padding[d] + ghost[d];
  }
  const size_t ARR_SIZE = arrstride[dimlist.size() - 1] * (dimlist.back() + 2 * (padding.back() + ghost.back())); ///< Used for assertions
  if (sec != 0) {
    assert(st >= 0);
    assert(st + ghost[d] <= ARR_SIZE);
    elemcpy(buffer_out, arr + st, ghost[d]);
    return buffer_out + ghost[d];
  } else {
    assert(padding[d] + ghost[d] >= 0);
    assert(padding[d] + ghost[d] + dimlist[d] <= ARR_SIZE);
    elemcpy(buffer_out, arr + padding[d] + ghost[d], dimlist[d]);
    return buffer_out + dimlist[d];
  }
}

template<unsigned dim, typename elemType>
inline typename std::enable_if<dim != 1, elemType*>::type 
unpack(elemType *arr, BitSet neighbor, elemType *buffer_recv, const std::vector<unsigned long> &arrstride,
       const std::vector<long> &dimlist, const std::vector<long> &padding,
       const std::vector<long> &ghost) {
  // Inner region
  long sec = 0;
  long st = 0;
  int d = (int) dim - 1;
  if (neighbor.get(dim)) {
    sec = 1;
    st = padding[d] + dimlist[d] + ghost[d];
  } else if (neighbor.get(-(int) dim)) {
    sec = -1;
    st = padding[d];
  }
  if (sec) {
    for (unsigned i = 0; i < ghost[d]; ++i)
      buffer_recv = unpack<dim - 1>(arr + arrstride[d] * (st + i), neighbor, buffer_recv,
                                    arrstride, dimlist, padding, ghost);
  } else {
    for (unsigned i = 0; i < dimlist[d]; ++i)
      buffer_recv = unpack<dim - 1>(arr + arrstride[d] * (padding[d] + ghost[d] + i), neighbor, buffer_recv,
                                    arrstride, dimlist, padding, ghost);
  }
  return buffer_recv;
}

template<unsigned dim, typename elemType>
inline typename std::enable_if<dim == 1, elemType*>::type 
unpack(elemType *arr, BitSet neighbor, elemType *buffer_recv, const std::vector<unsigned long> &arrstride,
       const std::vector<long> &dimlist, const std::vector<long> &padding,
       const std::vector<long> &ghost) {
  // Inner region
  long sec = 0;
  long st = 0;
  int d = 0;
  if (neighbor.get(1)) {
    sec = 1;
    st = padding[d] + dimlist[d] + ghost[d];
  } else if (neighbor.get(-1)) {
    sec = -1;
    st = padding[d];
  }
  if (sec) {
    elemcpy(arr + st, buffer_recv, ghost[d]);
    return buffer_recv + ghost[d];
  } else {
    elemcpy(arr + padding[d] + ghost[d], buffer_recv, dimlist[d]);
    return buffer_recv + dimlist[d];
  }
}


inline unsigned
evalsize(BitSet region, const std::vector<long> &dimlist, const std::vector<long> &ghost, bool inner = true) {
  // Inner region
  unsigned size = 1;
  for (int i = 1; i <= (int) dimlist.size(); ++i)
    if (region.get(i) || region.get(-i))
      size = size * ghost[i - 1];
    else
      size = size * (dimlist[i - 1] - (inner ? 2 * ghost[i - 1] : 0));
  return size;
}

extern std::vector<bElem *> arr_buffers_out;
extern std::vector<bElem *> arr_buffers_recv;

// ID is used to prevent message mismatch from messages with the same node, low performance only for validation testing.
template<unsigned dim, typename elemType>
void exchangeArr(elemType *arr, const MPI_Comm &comm, std::unordered_map<uint64_t, int> &rank_map,
                 const std::vector<long> &dimlist, const std::vector<long> &padding, const std::vector<long> &ghost) {
  std::vector<BitSet> neighbors;
  allneighbors(0, 1, dim, neighbors);
  neighbors.erase(neighbors.begin() + (neighbors.size() / 2));
  std::vector<unsigned long> tot;
  std::vector<MPI_Request> requests;
  std::vector<MPI_Status> stats;

  std::vector<unsigned long> arrstride(dimlist.size());
  unsigned long stri = 1;

  for (int i = 0; i < arrstride.size(); ++i) {
    arrstride[i] = stri;
    stri = stri * ((padding[i] + ghost[i]) * 2 + dimlist[i]);
  }

  std::vector<BitSet> non_empty_neighbors;
  for (int i = 0; i < (int) neighbors.size(); ++i) {
    unsigned long sector_size = evalsize(neighbors[i], dimlist, ghost, false);
    if(sector_size > 0) {
      non_empty_neighbors.emplace_back(neighbors[i]);
      tot.emplace_back(sector_size);
    }
  }
  // allocate MPI requests/stats arrays
  requests.resize(non_empty_neighbors.size());
  stats.resize(non_empty_neighbors.size());

  if (arr_buffers_out.size() == 0) {
    for(unsigned long sector_size : tot) {
      size_t size_in_bytes = sizeof(elemType) * sector_size;
      arr_buffers_recv.emplace_back((bElem*)aligned_alloc(4096, size_in_bytes));
      arr_buffers_out.emplace_back((bElem*)aligned_alloc(4096, size_in_bytes));
    }
  }
  assert(non_empty_neighbors.size() <= arr_buffers_out.size());
  assert(non_empty_neighbors.size() <= arr_buffers_recv.size());

  double st = omp_get_wtime(), ed;
  // Pack
#pragma omp parallel for
  for (int i = 0; i < (int) non_empty_neighbors.size(); ++i) {
    pack<dim>(arr, non_empty_neighbors[i], reinterpret_cast<elemType*>(arr_buffers_out[i]), arrstride, dimlist, padding, ghost);
  }

  ed = omp_get_wtime();
  packtime += ed - st;

#ifdef BARRIER_TIMESTEP
  MPI_Barrier(comm);
#endif

  st = omp_get_wtime();

  for (int i = 0; i < (int) non_empty_neighbors.size(); ++i) {
    MPI_Irecv(arr_buffers_recv[i], (int) (tot[i] * sizeof(elemType)), MPI_CHAR, rank_map[non_empty_neighbors[i].set],
              (int) non_empty_neighbors.size() - i - 1, comm, &(requests[i * 2]));
    MPI_Isend(arr_buffers_out[i], (int) (tot[i] * sizeof(elemType)), MPI_CHAR, rank_map[non_empty_neighbors[i].set], i, comm,
              &(requests[i * 2 + 1]));
  }

  ed = omp_get_wtime();
  calltime += ed - st;
  st = ed;

  // Wait
  MPI_Waitall(static_cast<int>(requests.size()), requests.data(), stats.data());

  ed = omp_get_wtime();
  waittime += ed - st;
  st = ed;

  // Unpack
#pragma omp parallel for
  for (int i = 0; i < (int) non_empty_neighbors.size(); ++i) {
    unpack<dim>(arr, non_empty_neighbors[i], reinterpret_cast<elemType*>(arr_buffers_recv[i]), arrstride, dimlist, padding, ghost);
  }

  ed = omp_get_wtime();
  packtime += ed - st;
}

/**
 * @brief Get the MPI_Datatype for a scalar of type elemType
 * 
 * @tparam elemType the type of the scalar
 * @return MPI_Datatype the MPI equivalent
 */
template<typename elemType>
MPI_Datatype get_scalar_MPI_Datatype() {
  static_assert(!std::is_same<elemType, elemType>::value, "Unrecognized elemType");
  return MPI_DATATYPE_NULL;
}
/**
 * @see get_scalar_MPI_Datatype
 */
template<> inline
MPI_Datatype get_scalar_MPI_Datatype<float>() { return MPI_FLOAT; }
/**
 * @see get_scalar_MPI_Datatype
 */
template<> inline
MPI_Datatype get_scalar_MPI_Datatype<double>() { return MPI_DOUBLE; }
/**
 * @see get_scalar_MPI_Datatype
 */
template<> inline MPI_Datatype get_scalar_MPI_Datatype<bComplexElem>() {
  static_assert(std::is_same<bElem, float>::value || std::is_same<bElem, double>::value);
  if(std::is_same<bElem, float>::value) return MPI_C_FLOAT_COMPLEX;
  return MPI_C_DOUBLE_COMPLEX;
}

/**
 * @return MPI_Datatype a sub-array mpi datatype of the appropriate size/type,
 *         or MPI_DATATYPE_NULL if the sub-array is of size 0
 */
template<typename elemType = bElem>
inline MPI_Datatype pack_type(BitSet neighbor, const std::vector<long> &dimlist, const std::vector<long> &padding,
                       const std::vector<long> &ghost) {
  int ndims = dimlist.size();
  std::vector<int> size(ndims), subsize(ndims), start(ndims);
  for (long dd = 0; dd < dimlist.size(); ++dd) {
    long d = (long)dimlist.size() - dd - 1;
    size[dd] = dimlist[d] + 2 * (padding[d] + ghost[d]);
    long dim = d + 1;
    long sec = 0;
    if (neighbor.get(dim)) {
      sec = 1;
      start[dd] = padding[d] + dimlist[d];
    } else if (neighbor.get(-(int) dim)) {
      sec = -1;
      start[dd] = padding[d] + ghost[d];
    }
    if (sec) {
      subsize[dd] = ghost[d];
    } else {
      subsize[dd] = dimlist[d];
      start[dd] = padding[d] + ghost[d];
    }
    // Handle size == 0 case
    if(subsize[dd] <= 0) {
      return MPI_DATATYPE_NULL;
    }
  }
  MPI_Datatype ret, scalar_mpi_datatype = get_scalar_MPI_Datatype<elemType>();
  // Subarray is most contiguous dimension first (largest index)
  MPI_Type_create_subarray(ndims, size.data(), subsize.data(), start.data(), MPI_ORDER_C, scalar_mpi_datatype, &ret);
  return ret;
}

/**
 * @return MPI_Datatype a sub-array mpi datatype of the appropriate size/type,
 *         or MPI_DATATYPE_NULL if the sub-array is of size 0
 */
template<typename elemType = bElem>
inline MPI_Datatype unpack_type(BitSet neighbor, const std::vector<long> &dimlist,
                                const std::vector<long> &padding,
                                const std::vector<long> &ghost) {
  int ndims = dimlist.size();
  std::vector<int> size(ndims), subsize(ndims), start(ndims);
  for (long dd = 0; dd < dimlist.size(); ++dd) {
    long d = (long)dimlist.size() - dd - 1;
    size[dd] = dimlist[d] + 2 * (padding[d] + ghost[d]);
    long dim = d + 1;
    long sec = 0;
    if (neighbor.get(dim)) {
      sec = 1;
      start[dd] = padding[d] + dimlist[d] + ghost[d];
    } else if (neighbor.get(-(int) dim)) {
      sec = -1;
      start[dd] = padding[d];
    }
    if (sec) {
      subsize[dd] = ghost[d];
    } else {
      subsize[dd] = dimlist[d];
      start[dd] = padding[d] + ghost[d];
    }
    // Handle size == 0 case
    if(subsize[dd] <= 0) {
      return MPI_DATATYPE_NULL;
    }
  }
  MPI_Datatype ret, scalar_mpi_datatype = get_scalar_MPI_Datatype<elemType>();
  // Subarray is most contiguous dimension first (largest index)
  MPI_Type_create_subarray(ndims, size.data(), subsize.data(), start.data(), MPI_ORDER_C, scalar_mpi_datatype, &ret);
  return ret;
}

template<unsigned dim, typename elemType = bElem>
void exchangeArrPrepareTypes(std::unordered_map<uint64_t, MPI_Datatype> &stypemap,
                             std::unordered_map<uint64_t, MPI_Datatype> &rtypemap,
                             const std::vector<long> &dimlist, const std::vector<long> &padding,
                             const std::vector<long> &ghost) {
  std::vector<BitSet> neighbors;
  allneighbors(0, 1, dim, neighbors);
  neighbors.erase(neighbors.begin() + (neighbors.size() / 2));
  std::vector<MPI_Request> requests(neighbors.size() * 2);

  for (auto n: neighbors) {
    MPI_Datatype MPI_rtype = unpack_type<elemType>(n, dimlist, padding, ghost);
    if(MPI_rtype != MPI_DATATYPE_NULL) {
      MPI_Type_commit(&MPI_rtype);
    }
    rtypemap[n.set] = MPI_rtype;
    MPI_Datatype MPI_stype = pack_type<elemType>(n, dimlist, padding, ghost);
    assert(!(MPI_rtype == MPI_DATATYPE_NULL ^ MPI_stype == MPI_DATATYPE_NULL));
    if(MPI_stype != MPI_DATATYPE_NULL) {
      MPI_Type_commit(&MPI_stype);
    }
    stypemap[n.set] = MPI_stype;
  }
}

// Using data types
template<unsigned dim, typename elemType>
void exchangeArrTypes(elemType *arr, const MPI_Comm &comm, std::unordered_map<uint64_t, int> &rank_map,
                      std::unordered_map<uint64_t, MPI_Datatype> &stypemap,
                      std::unordered_map<uint64_t, MPI_Datatype> &rtypemap) {
  std::vector<BitSet> neighbors;
  allneighbors(0, 1, dim, neighbors);
  neighbors.erase(neighbors.begin() + (neighbors.size() / 2));
  // get non-empty neighbors
  std::vector<BitSet> non_empty_neighbors;
  assert(stypemap.size() == rtypemap.size());
  assert(stypemap.size() == neighbors.size());
  for(unsigned i = 0; i < stypemap.size(); ++i) {
    uint64_t section = neighbors[i];
    assert(!(stypemap[section] == MPI_DATATYPE_NULL ^ rtypemap[section] == MPI_DATATYPE_NULL));
    if(stypemap[section] != MPI_DATATYPE_NULL) {
      non_empty_neighbors.push_back(neighbors[i]);
    }
  }
  std::vector<MPI_Request> requests(non_empty_neighbors.size() * 2);

  int rank;
  MPI_Comm_rank(comm, &rank);

  double st = omp_get_wtime(), ed;

  for (int i = 0; i < (int) non_empty_neighbors.size(); ++i) {
    MPI_Irecv(arr, 1, rtypemap[non_empty_neighbors[i].set], rank_map[non_empty_neighbors[i].set],
              (int) non_empty_neighbors.size() - i - 1, comm, &(requests[i * 2]));
    MPI_Isend(arr, 1, stypemap[non_empty_neighbors[i].set], rank_map[non_empty_neighbors[i].set], i, comm, &(requests[i * 2 + 1]));
  }

  ed = omp_get_wtime();
  calltime += ed - st;
  st = ed;

  // Wait
  std::vector<MPI_Status> stats(requests.size());
  MPI_Waitall(static_cast<int>(requests.size()), requests.data(), stats.data());

  ed = omp_get_wtime();
  waittime += ed - st;
}

typedef struct {
  bElem *arr;
  std::unordered_map<uint64_t, int> *rank_map;
  std::unordered_map<uint64_t, int> *id_map;
  int id;
} ArrExPack;

template<unsigned dim>
void exchangeArrAll(std::vector<ArrExPack> arr, const MPI_Comm &comm,
                    const std::vector<long> &dimlist, const std::vector<long> &padding,
                    const std::vector<long> &ghost) {
  std::vector<BitSet> neighbors;
  allneighbors(0, 1, dim, neighbors);
  neighbors.erase(neighbors.begin() + (neighbors.size() / 2));
  std::vector<bElem *> buffers_out(arr.size() * neighbors.size(), nullptr);
  std::vector<bElem *> buffers_recv(arr.size() * neighbors.size(), nullptr);
  std::vector<unsigned long> tot(neighbors.size());
  std::vector<MPI_Request> requests(arr.size() * neighbors.size() * 2);

  std::vector<unsigned long> arrstride(dimlist.size());
  unsigned long stri = 1;

  for (int i = 0; i < arrstride.size(); ++i) {
    arrstride[i] = stri;
    stri = stri * ((padding[i] + ghost[i]) * 2 + dimlist[i]);
  }

  for (int i = 0; i < (int) neighbors.size(); ++i) {
    tot[i] = (unsigned long) evalsize(neighbors[i], dimlist, ghost, false);
    for (int s = 0; s < arr.size(); ++s) {
      buffers_recv[i + s * neighbors.size()] = new bElem[tot[i]];
      buffers_out[i + s * neighbors.size()] = new bElem[tot[i]];
    }
  }

  double st = omp_get_wtime(), ed;

  // Pack
#pragma omp parallel for
  for (int i = 0; i < (int) neighbors.size(); ++i)
    for (int s = 0; s < arr.size(); ++s)
      pack<dim>(arr[s].arr, neighbors[i], buffers_out[i + s * neighbors.size()], arrstride, dimlist, padding, ghost);

  ed = omp_get_wtime();
  packtime += ed - st;

#ifdef BARRIER_TIMESTEP
  MPI_Barrier(comm);
#endif

  st = omp_get_wtime();

  for (int i = 0; i < (int) neighbors.size(); ++i)
    for (int s = 0; s < arr.size(); ++s) {
      MPI_Irecv(buffers_recv[i + s * neighbors.size()], (int) (tot[i] * sizeof(bElem)), MPI_CHAR,
                arr[s].rank_map->at(neighbors[i].set),
                arr[s].id_map->at(neighbors[i].set) * 100 + (int) neighbors.size() - i - 1,
                comm, &(requests[i * 2 + s * neighbors.size() * 2]));
      MPI_Isend(buffers_out[i + s * neighbors.size()], (int) (tot[i] * sizeof(bElem)), MPI_CHAR,
                arr[s].rank_map->at(neighbors[i].set), arr[s].id * 100 + i, comm,
                &(requests[i * 2 + s * neighbors.size() * 2 + 1]));
    }

  ed = omp_get_wtime();
  calltime += ed - st;
  st = ed;

  // Wait
  std::vector<MPI_Status> stats(requests.size());
  MPI_Waitall(static_cast<int>(requests.size()), requests.data(), stats.data());

  ed = omp_get_wtime();
  waittime += ed - st;
  st = ed;

  // Unpack
#pragma omp parallel for
  for (int i = 0; i < (int) neighbors.size(); ++i)
    for (int s = 0; s < arr.size(); ++s)
      unpack<dim>(arr[s].arr, neighbors[i], buffers_recv[i + s * neighbors.size()], arrstride, dimlist, padding, ghost);

  ed = omp_get_wtime();
  packtime += ed - st;

  // Cleanup
  for (auto b: buffers_out)
    delete[] b;
  for (auto b: buffers_recv)
    delete[] b;
}

#endif //BRICK_ARRAY_MPI_H
