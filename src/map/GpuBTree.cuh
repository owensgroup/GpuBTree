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
         typename AllocatorT = PoolAllocator>
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
  cudaError_t deleteKeys(uint32_t*& root,
                         KeyT*& d_queries,
                         SizeT& count,
                         cudaStream_t stream_id = 0);
  cudaError_t rangeQuery(uint32_t*& root,
                         KeyT*& d_queries_lower,
                         KeyT*& d_queries_upper,
                         ValueT*& d_range_results,
                         SizeT& count,
                         SizeT& range_lenght,
                         cudaStream_t stream_id = 0);
  cudaError_t concurrentOpsWithRangeQueries(uint32_t*& d_root,
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
                                            cudaStream_t stream_id = 0);
  bool _handle_memory;

 public:
  GpuBTreeMap(AllocatorT* mem_allocator = nullptr, int device_id = 0) {
    if (mem_allocator) {
      _mem_allocator = *mem_allocator;
      _handle_memory = false;
    } else {
      PoolAllocator allocator;
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
  cudaError_t deleteKeys(KeyT* queries, SizeT count, SourceT source = SourceT::DEVICE) {
    KeyT* d_queries;
    if (source == SourceT::HOST) {
      CHECK_ERROR(memoryUtil::deviceAlloc(d_queries, count));
      CHECK_ERROR(memoryUtil::cpyToDevice(queries, d_queries, count));
    } else {
      d_queries = queries;
    }

    CHECK_ERROR(deleteKeys(_d_root, d_queries, count));

    if (source == SourceT::HOST) {
      CHECK_ERROR(memoryUtil::deviceFree(d_queries));
    }

    return cudaSuccess;
  }
  cudaError_t rangeQuery(KeyT* queries_lower,
                         KeyT* queries_upper,
                         ValueT* results,
                         SizeT average_length,
                         SizeT count,
                         SourceT source = SourceT::DEVICE) {
    KeyT* d_queries_lower;
    KeyT* d_queries_upper;
    KeyT* d_results;
    auto total_range_lenght = count * average_length * 2;
    if (source == SourceT::HOST) {
      CHECK_ERROR(memoryUtil::deviceAlloc(d_queries_lower, count));
      CHECK_ERROR(memoryUtil::deviceAlloc(d_queries_upper, count));
      CHECK_ERROR(memoryUtil::deviceAlloc(d_results, total_range_lenght));
      CHECK_ERROR(memoryUtil::cpyToDevice(queries_lower, d_queries_lower, count));
      CHECK_ERROR(memoryUtil::cpyToDevice(queries_upper, d_queries_upper, count));
    } else {
      d_queries_lower = queries_lower;
      d_queries_upper = queries_upper;
      d_results = results;
    }

    CHECK_ERROR(rangeQuery(
        _d_root, d_queries_lower, d_queries_upper, d_results, count, average_length));

    if (source == SourceT::HOST) {
      CHECK_ERROR(memoryUtil::cpyToHost(d_results, results, total_range_lenght));
      CHECK_ERROR(memoryUtil::deviceFree(d_results));
      CHECK_ERROR(memoryUtil::deviceFree(d_queries_lower));
      CHECK_ERROR(memoryUtil::deviceFree(d_queries_upper));
    }

    return cudaSuccess;
  }

  cudaError_t concurrentOpsWithRangeQueries(
    KeyT* keys,
    ValueT* values,
    OperationT* ops,
    SizeT num_keys,
    KeyT* range_queries_lower,
    KeyT* range_queries_upper,
    ValueT* range_results,
    SizeT num_range_queries,
    SizeT average_length,
    KeyT* delete_queries,
    SizeT num_delete_queries,
    SourceT source = SourceT::DEVICE) {
      KeyT* d_keys;
      ValueT* d_values;
      OperationT* d_ops;

      KeyT* d_range_queries_lower;
      KeyT* d_range_queries_upper;
      ValueT* d_range_results;
      auto total_range_length = num_range_queries * average_length * 2;

      KeyT* d_delete_queries;
      if (source == SourceT::HOST) {
        // Search and insert
        CHECK_ERROR(memoryUtil::deviceAlloc(d_keys, num_keys));
        CHECK_ERROR(memoryUtil::deviceAlloc(d_values, num_keys));
        CHECK_ERROR(memoryUtil::deviceAlloc(d_ops, num_keys));
        CHECK_ERROR(memoryUtil::cpyToDevice(keys, d_keys, num_keys));
        CHECK_ERROR(memoryUtil::cpyToDevice(values, d_values, num_keys));
        CHECK_ERROR(memoryUtil::cpyToDevice(ops, d_ops, num_keys));

        // Range queries
        CHECK_ERROR(memoryUtil::deviceAlloc(d_range_queries_lower, num_range_queries));
        CHECK_ERROR(memoryUtil::deviceAlloc(d_range_queries_upper, num_range_queries));
        CHECK_ERROR(memoryUtil::deviceAlloc(d_range_results, total_range_length));
        CHECK_ERROR(memoryUtil::cpyToDevice(range_queries_lower, d_range_queries_lower, num_range_queries));
        CHECK_ERROR(memoryUtil::cpyToDevice(range_queries_upper, d_range_queries_upper, num_range_queries));

        // Delete
        CHECK_ERROR(memoryUtil::deviceAlloc(d_delete_queries, num_delete_queries));
        CHECK_ERROR(memoryUtil::cpyToDevice(delete_queries, d_delete_queries, num_delete_queries));
      } else {
        d_keys = keys;
        d_values = values;

        d_range_queries_lower = range_queries_lower;
        d_range_queries_upper = range_queries_upper;
        d_range_results = range_results;

        d_delete_queries = delete_queries;
      }
  
      CHECK_ERROR(concurrentOpsWithRangeQueries(
        _d_root, 
        d_keys, 
        d_values, 
        d_ops, 
        num_keys,
        d_range_queries_lower,
        d_range_queries_upper,
        d_range_results,
        num_range_queries,
        average_length,
        d_delete_queries,
        num_delete_queries));
  
      if (source == SourceT::HOST) {
        CHECK_ERROR(memoryUtil::cpyToHost(d_values, values, num_keys));
        CHECK_ERROR(memoryUtil::deviceFree(d_keys));
        CHECK_ERROR(memoryUtil::deviceFree(d_values));
        CHECK_ERROR(memoryUtil::deviceFree(d_ops));

        CHECK_ERROR(memoryUtil::cpyToHost(d_range_results, range_results, total_range_length));
        CHECK_ERROR(memoryUtil::deviceFree(d_range_results));
        CHECK_ERROR(memoryUtil::deviceFree(d_range_queries_lower));
        CHECK_ERROR(memoryUtil::deviceFree(d_range_queries_upper));

        CHECK_ERROR(memoryUtil::deviceFree(d_delete_queries));

      }
  
      return cudaSuccess;
      
    }


};
};  // namespace GpuBTree
