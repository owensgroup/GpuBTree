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

};  // namespace kernels
};  // namespace GpuBTree