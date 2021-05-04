/*Copyright(c) 2020, The Regents of the University of California, Davis.            */
/*                                                                                  */
/*                                                                                  */
/*Redistribution and use in source and binary forms, with or without modification,  */
/*are permitted provided that the following conditions are met :                    */
/*                                                                                  */
/*1. Redistributions of source code must retain the above copyright notice, this    */
/*list of conditions and the following disclaimer.                                  */
/*2. Redistributions in binary form must reproduce the above copyright notice,      */
/*this list of conditions and the following disclaimer in the documentation         */
/*and / or other materials provided with the distribution.                          */
/*                                                                                  */
/*THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND   */
/*ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED     */
/*WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.*/
/*IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,  */
/*INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES(INCLUDING, BUT */
/*NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR*/
/*PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, */
/*WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT(INCLUDING NEGLIGENCE OR OTHERWISE) */
/*ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE        */
/*POSSIBILITY OF SUCH DAMAGE.                                                       */
/************************************************************************************/
/************************************************************************************/

#pragma once

#include <cstdint>

namespace GpuBTree {
namespace kernels {
template<typename KeyT, typename ValueT, typename SizeT, typename AllocatorT>
__global__ void insert_keys(uint32_t* d_root,
                            KeyT* d_keys,
                            ValueT* d_values,
                            SizeT num_keys,
                            AllocatorT allocator) {
  uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  uint32_t laneId = threadIdx.x & 0x1F;

  KeyT myKey;
  ValueT myValue;
  bool to_insert = false;

  if ((tid - laneId) >= num_keys)
    return;

  if (tid < num_keys) {
    myKey = d_keys[tid] + 2;
    myValue = d_values[tid] + 2;
    to_insert = true;
  }

  warps::insertion_unit(to_insert, myKey, myValue, d_root, &allocator);
}

template<typename AllocatorT>
__global__ void init_btree(uint32_t* d_root, AllocatorT allocator) {
  uint32_t laneId = threadIdx.x & 0x1F;

  uint32_t root_id;
  if (laneId == 0)
    root_id = allocator.allocate();

  root_id = __shfl_sync(WARP_MASK, root_id, 0);

  *d_root = root_id;
  uint32_t* tree_root = allocator.getAddressPtr(root_id);

  if (laneId < 2)
    tree_root[laneId] = 1 - laneId;
}

template<typename KeyT, typename ValueT, typename SizeT, typename AllocatorT>
__global__ void search_b_tree(uint32_t* d_root,
                              KeyT* d_queries,
                              ValueT* d_results,
                              SizeT num_queries,
                              AllocatorT allocator) {
  uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  uint32_t laneId = threadIdx.x & 0x1F;
  if ((tid - laneId) >= num_queries)
    return;

  uint32_t myQuery = 0;
  uint32_t myResult = SEARCH_NOT_FOUND;
  bool to_search = false;

  if (tid < num_queries) {
    myQuery = d_queries[tid] + 2;
    to_search = true;
  }

  warps::search_unit(to_search, laneId, myQuery, myResult, d_root, &allocator);

  if (tid < num_queries)
    myResult = myResult ? myResult - 2 : myResult;
  d_results[tid] = myResult;
}

template<typename KeyT, typename SizeT, typename AllocatorT>
__global__ void compact_tree(uint32_t* d_root,
                             KeyT* d_tree,
                             SizeT* d_num_nodes,
                             AllocatorT allocator) {
  uint32_t laneId = threadIdx.x & 0x1F;

  uint32_t leftMostBranch = *d_root;

  uint32_t nodesCount = 0;
  bool isIntermediate = true;

  while (isIntermediate) {
    uint32_t laneData = *(allocator.getAddressPtr(leftMostBranch) + laneId);

    isIntermediate = ((laneData & 0x80000000) != 0);
    isIntermediate = __shfl_sync(WARP_MASK, isIntermediate, 0, WARP_WIDTH);
    laneData &= 0x7FFFFFFF;

    uint32_t writeData = laneData;
    writeData = laneData;
    writeData = isIntermediate ? (writeData | 0x80000000) : writeData;
    d_tree[leftMostBranch * NODE_WIDTH + laneId] = writeData;
    nodesCount++;

    leftMostBranch = __shfl_sync(WARP_MASK, laneData, 1, WARP_WIDTH);
    uint32_t nextInLevel = __shfl_sync(WARP_MASK, laneData, 31, WARP_WIDTH);

    while (nextInLevel) {
      laneData = *(allocator.getAddressPtr(nextInLevel) + laneId);
      laneData &= 0x7FFFFFFF;

      writeData = laneData;
      writeData = isIntermediate ? (writeData | 0x80000000) : writeData;
      d_tree[nextInLevel * NODE_WIDTH + laneId] = writeData;
      nodesCount++;

      nextInLevel = __shfl_sync(WARP_MASK, laneData, 31, WARP_WIDTH);
    }
  }

  *d_num_nodes = nodesCount;
}

template<typename KeyT, typename SizeT, typename AllocatorT>
__global__ void delete_b_tree(uint32_t* d_root,
                              KeyT* d_queries,
                              SizeT num_queries,
                              AllocatorT allocator) {
  uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  uint32_t laneId = threadIdx.x & 0x1F;
  if ((tid - laneId) >= num_queries)
    return;

  KeyT myQuery = 0xFFFFFFFF;

  if (tid < uint32_t(num_queries)) {
    myQuery = d_queries[tid] + 2;
  }

  warps::delete_unit_bulk(laneId, myQuery, d_root, &allocator);
}

template<typename KeyT, typename ValueT, typename SizeT, typename AllocatorT>
__global__ void range_b_tree(uint32_t* d_root,
                             KeyT* d_queries_lower,
                             KeyT* d_queries_upper,
                             ValueT* d_range_results,
                             SizeT num_queries,
                             SizeT range_length,
                             AllocatorT allocator) {
  uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  uint32_t laneId = lane_id();
  if ((tid - laneId) >= num_queries)
    return;

  uint32_t lower_bound = 0;
  uint32_t upper_bound = 0;
  bool to_search = false;

  if (tid < num_queries) {
    lower_bound = d_queries_lower[tid] + 2;
    upper_bound = d_queries_upper[tid] + 2;
    to_search = true;
  }

  warps::range_unit(laneId,
                    to_search,
                    lower_bound,
                    upper_bound,
                    d_range_results,
                    d_root,
                    range_length,
                    &allocator);
}


// template<typename KeyT, typename ValueT, typename SizeT, typename AllocatorT>
// __global__ void concurrent_ops_b_tree(uint32_t* d_root,
//                                       KeyT* d_keys,
//                                       ValueT* d_values,
//                                       OperationT* d_ops,
//                                       SizeT num_keys,
  //                                     AllocatorT allocator) {
  // uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  // uint32_t laneId = threadIdx.x & 0x1F;

  // KeyT myKey = 0xFFFFFFFF;
  // ValueT myValue = 0xFFFFFFFF;
  // OperationT myOp = OperationT::NOP;
  // bool to_insert = false;
  // bool to_delete = false;
  // bool to_query = false;

  // if ((tid - laneId) >= num_keys)
  //   return;

  // if (tid < num_keys) {
  //   myKey = d_keys[tid] + 2;
//     myOp = d_ops[tid];

//     to_insert = myOp == OperationT::INSERT;
//     to_delete = myOp == OperationT::DELETE;
//     to_query = myOp == OperationT::QUERY;

//     if (to_insert)
//       myValue = d_values[tid] + 2;
//   }

//   warps::insertion_unit(to_insert, myKey, myValue, d_root, &allocator);
//   warps::concurrent_search_unit(to_query, laneId, myKey, myValue, d_root, &allocator);

//   if (to_query) {
//     myValue = myValue ? myValue - 2 : myValue;
//     d_values[tid] = myValue;
//   }
// }

template<typename KeyT, typename ValueT, typename SizeT, typename AllocatorT>
__global__ void fused_ops_b_tree(uint32_t* d_root,
                             KeyT* d_keys,
			                       ValueT* d_values,
			                       OperationT* d_ops,
   			                     SizeT num_keys,
			                       KeyT* range_queries_lower,
                             KeyT* range_queries_upper,
                             ValueT* d_range_results,
                             SizeT num_range_queries,
                             SizeT range_length,
			                       KeyT* delete_queries,
			                       SizeT num_delete_queries,
                             AllocatorT allocator) {
  uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  uint32_t laneId = lane_id();
  if ((tid - laneId) >= num_range_queries)
    return;
  if ((tid - laneId) >= num_delete_queries)
    return;
  if ((tid - laneId) >= num_keys)
    return;


  KeyT myKey = 0xFFFFFFFF;
  ValueT myValue = 0xFFFFFFFF;
  OperationT myOp = OperationT::NOP;
  bool to_insert = false;
	//  bool to_delete = false;
  bool to_query = false;

  if (tid < num_keys) {
    myKey = d_keys[tid] + 2;
    myOp = d_ops[tid];

    to_insert = myOp == OperationT::INSERT;
//    to_delete = myOp == OperationT::DELETE;
    to_query = myOp == OperationT::QUERY;

    if (to_insert)
      myValue = d_values[tid] + 2;
  }

  warps::insertion_unit(to_insert, myKey, myValue, d_root, &allocator);
  warps::concurrent_search_unit(to_query, laneId, myKey, myValue, d_root, &allocator);



  uint32_t lower_bound = 0;
  uint32_t upper_bound = 0;
  bool to_search = false;

  if (tid < num_range_queries) {
    lower_bound = range_queries_lower[tid] + 2;
    upper_bound = range_queries_upper[tid] + 2;
    to_search = true;
  }

  warps::range_unit(laneId,
                    to_search,
                    lower_bound,
                    upper_bound,
                    d_range_results,
                    d_root,
                    range_length,
                    &allocator);

  __syncthreads();
  __threadfence();

  KeyT deleteQuery = 0xFFFFFFFF;
  if (tid < uint32_t(num_delete_queries)) {
    deleteQuery = delete_queries[tid] + 2;
  }

  warps::delete_unit_bulk(laneId, deleteQuery, d_root, &allocator);

}


};  // namespace kernels
};  // namespace GpuBTree