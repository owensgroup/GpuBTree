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
template<typename SizeT, typename KeyT, typename ValueT, typename AllocatorT>
cudaError_t GpuBTreeMap<SizeT, KeyT, ValueT, AllocatorT>::initBTree(
    uint32_t*& d_root,
    cudaStream_t stream_id) {
  kernels::init_btree<<<1, 32, 0, stream_id>>>(d_root, _mem_allocator);
  return cudaDeviceSynchronize();
}

template<typename SizeT, typename KeyT, typename ValueT, typename AllocatorT>
cudaError_t GpuBTreeMap<SizeT, KeyT, ValueT, AllocatorT>::insertKeys(
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

template<typename SizeT, typename KeyT, typename ValueT, typename AllocatorT>
cudaError_t GpuBTreeMap<SizeT, KeyT, ValueT, AllocatorT>::searchKeys(
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
}  // namespace GpuBTree
