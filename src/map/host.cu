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
template<typename KeyT, typename ValueT, typename SizeT, typename AllocatorT>
cudaError_t GpuBTreeMap<KeyT, ValueT, SizeT, AllocatorT>::initBTree(
    uint32_t*& d_root,
    cudaStream_t stream_id) {
  kernels::init_btree<<<1, 32, 0, stream_id>>>(d_root, _mem_allocator);
  return cudaDeviceSynchronize();
}

template<typename KeyT, typename ValueT, typename SizeT, typename AllocatorT>
cudaError_t GpuBTreeMap<KeyT, ValueT, SizeT, AllocatorT>::insertKeys(
    uint32_t*& d_root,
    KeyT*& d_keys,
    ValueT*& d_values,
    SizeT& count,
    cudaStream_t stream_id) {
  const uint32_t num_blocks = (count + BLOCKSIZE_BUILD_ - 1) / BLOCKSIZE_BUILD_;
  const uint32_t shared_bytes = 0;
  kernels::insert_keys<<<num_blocks, BLOCKSIZE_BUILD_, shared_bytes, stream_id>>>(
      d_root, d_keys, d_values, count, _mem_allocator);

  return cudaSuccess;
}

template<typename KeyT, typename ValueT, typename SizeT, typename AllocatorT>
cudaError_t GpuBTreeMap<KeyT, ValueT, SizeT, AllocatorT>::searchKeys(
    uint32_t*& d_root,
    KeyT*& d_queries,
    ValueT*& d_results,
    SizeT& count,
    cudaStream_t stream_id) {
  const uint32_t num_blocks = (count + BLOCKSIZE_SEARCH_ - 1) / BLOCKSIZE_SEARCH_;
  const uint32_t shared_bytes = 0;
  kernels::search_b_tree<<<num_blocks, BLOCKSIZE_SEARCH_, shared_bytes, stream_id>>>(
      d_root, d_queries, d_results, count, _mem_allocator);

  return cudaSuccess;
}

template<typename KeyT, typename ValueT, typename SizeT, typename AllocatorT>
cudaError_t GpuBTreeMap<KeyT, ValueT, SizeT, AllocatorT>::compactTree(
    uint32_t*& d_root,
    KeyT*& d_tree,
    SizeT*& d_num_nodes,
    cudaStream_t stream_id) {
  const uint32_t num_blocks = 1;
  const uint32_t block_size = 32;
  const uint32_t shared_bytes = 0;
  kernels::compact_tree<<<num_blocks, block_size, shared_bytes, stream_id>>>(
      d_root, d_tree, d_num_nodes, _mem_allocator);

  return cudaSuccess;
}

template<typename KeyT, typename ValueT, typename SizeT, typename AllocatorT>
cudaError_t GpuBTreeMap<KeyT, ValueT, SizeT, AllocatorT>::deleteKeys(
    uint32_t*& d_root,
    KeyT*& d_queries,
    SizeT& count,
    cudaStream_t stream_id) {
  const uint32_t num_blocks = (count + BLOCKSIZE_SEARCH_ - 1) / BLOCKSIZE_SEARCH_;
  const uint32_t shared_bytes = 0;
  kernels::delete_b_tree<<<num_blocks, BLOCKSIZE_SEARCH_, shared_bytes, stream_id>>>(
      d_root, d_queries, count, _mem_allocator);

  return cudaSuccess;
}

template<typename KeyT, typename ValueT, typename SizeT, typename AllocatorT>
cudaError_t GpuBTreeMap<KeyT, ValueT, SizeT, AllocatorT>::rangeQuery(
    uint32_t*& d_root,
    KeyT*& d_queries_lower,
    KeyT*& d_queries_upper,
    ValueT*& d_range_results,
    SizeT& count,
    SizeT& range_lenght,
    cudaStream_t stream_id) {
  const uint32_t block_size = 512;
  const uint32_t num_blocks = (count + block_size - 1) / block_size;
  const uint32_t shared_bytes = 0;
  kernels::range_b_tree<<<num_blocks, block_size, shared_bytes, stream_id>>>(
      d_root,
      d_queries_lower,
      d_queries_upper,
      d_range_results,
      count,
      range_lenght,
      _mem_allocator);

  return cudaSuccess;
}

template<typename KeyT, typename ValueT, typename SizeT, typename AllocatorT>
cudaError_t GpuBTreeMap<KeyT, ValueT, SizeT, AllocatorT>::concurrentOpsWithRangeQueries(
  uint32_t*& d_root,
  KeyT*& d_keys,
  ValueT*& d_values,
  OperationT*& d_ops,
  SizeT& num_keys,
  KeyT*& range_queries_lower,
  KeyT*& range_queries_upper,
  ValueT*& d_range_results,
  SizeT& num_range_queries,
  SizeT& range_length,
  KeyT*& delete_queries,
  SizeT& num_delete_queries,
  cudaStream_t stream_id) {
    const uint32_t block_size = 512;
    const uint32_t num_blocks = (num_keys + block_size - 1) / block_size;
    const uint32_t shared_bytes = 0;
  kernels::fused_ops_b_tree<<<num_blocks, block_size, shared_bytes, stream_id>>>(
    d_root,
    d_keys,
    d_values,
    d_ops,
    num_keys,
    range_queries_lower,
    range_queries_upper,
    d_range_results,
    num_range_queries,
    range_length,
    delete_queries,
    num_delete_queries,
    _mem_allocator);

    return cudaSuccess;
  }

};  // namespace GpuBTree
