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

#include <stdio.h>

#include <cstdint>

namespace GpuBTree {

template<typename KeyT,
         typename ValueT,
         typename SizeT = KeyT,
         typename AllocatorT = BoolAllocator>
class GpuBTreeMap {
 private:
  static constexpr uint32_t EMPTY_KEY = 0xFFFFFFFF;
  static constexpr uint32_t DELETED_KEY = 0xFFFFFFFF;
  static constexpr uint32_t BLOCKSIZE_BUILD_ = 128;
  static constexpr uint32_t BLOCKSIZE_SEARCH_ = 1024;

  SizeT _num_keys;
  int _device_id;
  uint32_t* _d_root;
  AllocatorT _mem_allocator;

  cudaError_t initBTree(uint32_t*& root, cudaStream_t stream_id = 0);
  cudaError_t insertKeys(uint32_t*& root,
                         KeyT*& d_keys,
                         ValueT*& d_values,
                         SizeT& count,
                         cudaStream_t stream_id = 0);
  cudaError_t searchKeys(uint32_t*& root,
                         KeyT*& d_queries,
                         ValueT*& d_results,
                         SizeT& count,
                         cudaStream_t stream_id = 0);
  cudaError_t compactTree(uint32_t*& root,
                          KeyT*& d_tree,
                          SizeT*& d_num_nodes,
                          cudaStream_t stream_id = 0);
  bool _handle_memory;

 public:
  GpuBTreeMap(AllocatorT* mem_allocator = nullptr, int device_id = 0) {
    if (mem_allocator) {
      _mem_allocator = *mem_allocator;
      _handle_memory = false;
    } else {
      BoolAllocator allocator;
      allocator.init();
      _mem_allocator = allocator;
      _mem_allocator.init();
      CHECK_ERROR(memoryUtil::deviceAlloc(_d_root, 1));
      _handle_memory = true;
    }
    _device_id = device_id;
    CHECK_ERROR(cudaSetDevice(_device_id));
    initBTree(_d_root);
  }
  cudaError_t init(AllocatorT mem_allocator, uint32_t* root_, int deviceId = 0) {
    _device_id = deviceId;
    _mem_allocator = mem_allocator;
    _d_root = root_;  // assumes that the root already contains a one
    return cudaSuccess;
  }
  ~GpuBTreeMap() {}
  void free() {
    if (_handle_memory) {
      CHECK_ERROR(cudaDeviceSynchronize());
      _mem_allocator.free();
    }
  }

  __host__ __device__ AllocatorT* getAllocator() { return &_mem_allocator; }
  __host__ __device__ uint32_t* getRoot() { return _d_root; }
  cudaError_t insertKeys(KeyT* keys,
                         ValueT* values,
                         SizeT count,
                         SourceT source = SourceT::DEVICE) {
    KeyT* d_keys;
    ValueT* d_values;
    if (source == SourceT::HOST) {
      CHECK_ERROR(memoryUtil::deviceAlloc(d_keys, count));
      CHECK_ERROR(memoryUtil::deviceAlloc(d_values, count));
      CHECK_ERROR(memoryUtil::cpyToDevice(keys, d_keys, count));
      CHECK_ERROR(memoryUtil::cpyToDevice(values, d_values, count));
    } else {
      d_keys = keys;
      d_values = values;
    }

    CHECK_ERROR(insertKeys(_d_root, d_keys, d_values, count));

    if (source == SourceT::HOST) {
      CHECK_ERROR(memoryUtil::deviceFree(d_keys));
      CHECK_ERROR(memoryUtil::deviceFree(d_values));
    }

    return cudaSuccess;
  }

  cudaError_t searchKeys(KeyT* queries,
                         ValueT* results,
                         SizeT count,
                         SourceT source = SourceT::DEVICE) {
    KeyT* d_queries;
    ValueT* d_results;
    if (source == SourceT::HOST) {
      CHECK_ERROR(memoryUtil::deviceAlloc(d_queries, count));
      CHECK_ERROR(memoryUtil::deviceAlloc(d_results, count));

      CHECK_ERROR(memoryUtil::cpyToDevice(queries, d_queries, count));
    } else {
      d_queries = queries;
      d_results = results;
    }

    CHECK_ERROR(searchKeys(_d_root, d_queries, d_results, count));

    if (source == SourceT::HOST) {
      CHECK_ERROR(memoryUtil::cpyToHost(d_results, results, count));
      CHECK_ERROR(memoryUtil::deviceFree(d_queries));
      CHECK_ERROR(memoryUtil::deviceFree(d_results));
    }

    return cudaSuccess;
  }

  cudaError_t compactTree(KeyT*& btree,
                          SizeT max_nodes,
                          SizeT& num_nodes,
                          SourceT source = SourceT::DEVICE) {
    KeyT* d_tree;
    KeyT* d_num_nodes;
    if (source == SourceT::HOST) {
      CHECK_ERROR(memoryUtil::deviceAlloc(d_tree, max_nodes * NODE_WIDTH));
      CHECK_ERROR(memoryUtil::deviceAlloc(d_num_nodes, 1));
    } else {
      d_tree = btree;
      d_num_nodes = &num_nodes;
    }

    CHECK_ERROR(compactTree(_d_root, d_tree, d_num_nodes));

    if (source == SourceT::HOST) {
      CHECK_ERROR(memoryUtil::cpyToHost(d_num_nodes, &num_nodes, 1));
      CHECK_ERROR(memoryUtil::deviceFree(d_num_nodes));

      CHECK_ERROR(memoryUtil::cpyToHost(d_tree, btree, num_nodes * NODE_WIDTH));
      CHECK_ERROR(memoryUtil::deviceFree(d_tree));
    }

    return cudaSuccess;
  }
  cudaError_t deleteKeys(KeyT* keys, SizeT count, SourceT source = SourceT::DEVICE) {}
};
};  // namespace GpuBTree
