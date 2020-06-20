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

#include <gtest/gtest.h>

#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <random>
#include <vector>
#include <queue>

#include "GpuBTree.h"

template<typename key_t>
void validate_tree_strucutre(key_t* h_tree, std::vector<key_t> keys, size_t num_keys) {
  // resize to match the requested size
  keys.resize(num_keys);

  // Traverse the tree:
  const int PAIRS_PER_NODE = NODE_WIDTH >> 1;

  uint32_t numNodes = 0;
  uint32_t prevLevel = PAIRS_PER_NODE - 1;
  uint32_t pairsCount = 0;
  uint32_t pairsPerLevel = PAIRS_PER_NODE;
  uint32_t curLevel = 0;
  uint32_t levelNodes = 0;
  uint32_t addedToLevel = 0;
  uint32_t treeHeight = 0;
  auto pivotToKey = [](key_t key) { return key & 0x7FFFFFFF; };
  auto isValidEntry = [pivotToKey](key_t key) { return pivotToKey(key) != 0; };
  auto isLeafNode = [](key_t key) { return (key & 0x80000000) == 0; };

  std::vector<int> levelNodesId;
  std::vector<std::vector<int>> levelsNodesId;
  using value_t = key_t;
  using tuple_t = std::pair<key_t, value_t>;
  std::vector<tuple_t> tree_pairs;
  std::queue<uint32_t> queue;
  queue.push(0);
  while (!queue.empty()) {
    key_t* current = h_tree + queue.front() * NODE_WIDTH;
    numNodes++;
    levelNodesId.push_back(queue.front());
    queue.pop();
    key_t curKey = *current;
    levelNodes++;
    value_t curValue;
    if (pivotToKey(curKey) == 1)  // first key is always 1
      treeHeight++;
    for (int iPair = 0; iPair < PAIRS_PER_NODE - 1; iPair++) {
      bool leafNode = isLeafNode(*current);
      if (isValidEntry(*current)) {
        if (leafNode) {
          curKey = *current;
          curValue = *(current + 1);
          if (curKey != 1)
            tree_pairs.push_back(std::make_pair(curKey, curValue));

          if (tree_pairs.size() == 0) {
            // first pair must be {1,0}
            ASSERT_EQ(curKey, 1);
            ASSERT_EQ(curValue, 0);
          }
        } else {
          curKey = pivotToKey(*current);
          key_t o = pivotToKey(current[1]);
          queue.push(o);
          addedToLevel++;
        }
      }
      pairsCount++;
      current += 2;
    }
    if (pairsCount == prevLevel) {
      pairsCount = 0;
      pairsPerLevel *= PAIRS_PER_NODE;
      curLevel++;
      levelsNodesId.push_back(levelNodesId);
      levelNodesId.clear();
      levelNodes = 0;
      prevLevel = addedToLevel * (PAIRS_PER_NODE - 1);
      addedToLevel = 0;
    }
  }

  ASSERT_EQ(tree_pairs.size(), keys.size());

  // Validate the tree structure
  for (uint32_t iLevel = 0; iLevel < levelsNodesId.size(); iLevel++) {
    for (uint32_t iNode = 0; iNode < levelsNodesId[iLevel].size(); iNode++) {
      uint32_t nodeIdx = levelsNodesId[iLevel][iNode];

      key_t linkPtr = pivotToKey((h_tree + nodeIdx * NODE_WIDTH)[31]);
      key_t linkMin = pivotToKey((h_tree + nodeIdx * NODE_WIDTH)[30]);

      if (iNode == (levelsNodesId[iLevel].size() - 1))  // last node in level
      {
        // link should be zero at the last node in the level
        ASSERT_EQ(linkPtr, 0);
        ASSERT_EQ(linkMin, 0);
      } else {
        key_t correctPtr = levelsNodesId[iLevel][iNode + 1];
        ASSERT_EQ(linkPtr, correctPtr);  // expected node idx from the tree next level
        key_t* neighborNode = (h_tree + correctPtr * NODE_WIDTH);
        key_t* curNode = (h_tree + nodeIdx * NODE_WIDTH);
        for (int i = 0; i < PAIRS_PER_NODE; i++) {
          uint32_t nextKey = pivotToKey(*neighborNode);
          uint32_t curKey = pivotToKey(*curNode);
          if (isValidEntry(nextKey)) {
            ASSERT_GE(nextKey, linkMin);
          }
          if (i < (PAIRS_PER_NODE - 1) && isValidEntry(curKey)) {
            ASSERT_LT(curKey, linkMin);
          }
          neighborNode += 2;
          curNode += 2;
        }
      }
    }
  }

  std::sort(keys.begin(), keys.end());  // sort keys
  for (int iKey = 0; iKey < keys.size(); iKey++) {
    key_t treeKey = tree_pairs[iKey].first - 2;
    value_t treeVal = tree_pairs[iKey].second - 2;
    ASSERT_EQ(keys[iKey], treeKey);
    ASSERT_EQ(treeKey, treeVal);
  }
}
TEST(BTreeMap, SimpleBuild) {
  using key_t = uint32_t;
  using value_t = uint32_t;

  GpuBTree::GpuBTreeMap<key_t, value_t> btree;

  // Input number of keys
  size_t numKeys = 1 << 10;

  // Prepare the keys
  std::vector<key_t> keys;
  std::vector<value_t> values;
  keys.reserve(numKeys);
  values.reserve(numKeys);
  for (int iKey = 0; iKey < numKeys; iKey++) {
    keys.push_back(iKey);
  }

  // shuffle the keys
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(keys.begin(), keys.end(), g);

  // assign the values
  for (int iKey = 0; iKey < numKeys; iKey++) {
    values.push_back(keys[iKey]);
  }

  // Move data to GPU
  key_t* d_keys;
  value_t* d_values;
  CHECK_ERROR(memoryUtil::deviceAlloc(d_keys, numKeys));
  CHECK_ERROR(memoryUtil::deviceAlloc(d_values, numKeys));
  CHECK_ERROR(memoryUtil::cpyToDevice(keys.data(), d_keys, numKeys));
  CHECK_ERROR(memoryUtil::cpyToDevice(values.data(), d_values, numKeys));

  // Build the tree
  GpuTimer timer;
  timer.timerStart();
  btree.insertKeys(d_keys, d_values, numKeys, SourceT::DEVICE);
  timer.timerStop();

  uint32_t max_nodes = 1 << 19;
  key_t* h_tree = new uint32_t[max_nodes * NODE_WIDTH];
  uint32_t num_nodes = 0;
  btree.compactTree(h_tree, max_nodes, num_nodes, SourceT::HOST);

  // Validation
  validate_tree_strucutre(h_tree, keys, keys.size());
  // cleanup
  cudaFree(d_keys);
  cudaFree(d_values);
  delete[] h_tree;
  btree.free();
}

TEST(BTreeMap, BuildSameKeys) {
  using key_t = uint32_t;
  using value_t = uint32_t;

  GpuBTree::GpuBTreeMap<key_t, value_t> btree;

  // Input number of keys
  size_t numKeys = 1 << 10;

  // Prepare the keys
  std::vector<key_t> keys;
  std::vector<key_t> unique_keys;
  std::vector<value_t> values;
  keys.reserve(numKeys);
  values.reserve(numKeys);
  for (int iKey = 0; iKey < numKeys / 2; iKey++) {
    keys.push_back(iKey);
    keys.push_back(iKey);
    unique_keys.push_back(iKey);
  }

  // shuffle the keys
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(keys.begin(), keys.end(), g);

  // assign the values
  for (int iKey = 0; iKey < numKeys; iKey++) {
    values.push_back(keys[iKey]);
  }

  // Move data to GPU
  key_t* d_keys;
  value_t* d_values;
  CHECK_ERROR(memoryUtil::deviceAlloc(d_keys, numKeys));
  CHECK_ERROR(memoryUtil::deviceAlloc(d_values, numKeys));
  CHECK_ERROR(memoryUtil::cpyToDevice(keys.data(), d_keys, numKeys));
  CHECK_ERROR(memoryUtil::cpyToDevice(values.data(), d_values, numKeys));

  // Build the tree
  GpuTimer timer;
  timer.timerStart();
  btree.insertKeys(d_keys, d_values, numKeys, SourceT::DEVICE);
  timer.timerStop();

  uint32_t max_nodes = 1 << 19;
  key_t* h_tree = new uint32_t[max_nodes * NODE_WIDTH];
  uint32_t num_nodes = 0;
  btree.compactTree(h_tree, max_nodes, num_nodes, SourceT::HOST);

  // Validation
  validate_tree_strucutre(h_tree, unique_keys, unique_keys.size());
  // cleanup
  cudaFree(d_keys);
  cudaFree(d_values);
  delete[] h_tree;
  btree.free();
}

TEST(BTreeMap, SearchRandomKeys) {
  GpuBTree::GpuBTreeMap<uint32_t, uint32_t> btree;

  // Input number of keys
  uint32_t numKeys = 512;

  // RNG
  std::random_device rd;
  std::mt19937 g(rd());

  // Prepare the keys
  std::vector<uint32_t> keys;
  std::vector<uint32_t> values;
  keys.reserve(numKeys);
  values.reserve(numKeys);
  for (int iKey = 0; iKey < numKeys; iKey++) {
    keys.push_back(iKey);
  }

  // shuffle the keys
  std::shuffle(keys.begin(), keys.end(), g);

  // assign the values
  for (int iKey = 0; iKey < numKeys; iKey++) {
    values.push_back(keys[iKey]);
  }

  // Move data to GPU
  uint32_t *d_keys, *d_values;
  CHECK_ERROR(memoryUtil::deviceAlloc(d_keys, numKeys));
  CHECK_ERROR(memoryUtil::deviceAlloc(d_values, numKeys));
  CHECK_ERROR(memoryUtil::cpyToDevice(keys.data(), d_keys, numKeys));
  CHECK_ERROR(memoryUtil::cpyToDevice(values.data(), d_values, numKeys));

  // Build the tree
  GpuTimer build_timer;
  build_timer.timerStart();
  btree.insertKeys(d_keys, d_values, numKeys, SourceT::DEVICE);
  build_timer.timerStop();

  // Input number of queries
  uint32_t numQueries = numKeys;

  // Prepare the query keys
  std::vector<uint32_t> query_keys;
  std::vector<uint32_t> query_results;
  query_keys.reserve(numQueries * 2);
  query_results.resize(numQueries);
  for (int iKey = 0; iKey < numQueries * 2; iKey++) {
    query_keys.push_back(iKey);
  }

  // shuffle the queries
  std::shuffle(query_keys.begin(), query_keys.end(), g);

  // Move data to GPU
  uint32_t *d_queries, *d_results;
  CHECK_ERROR(memoryUtil::deviceAlloc(d_queries, numQueries));
  CHECK_ERROR(memoryUtil::deviceAlloc(d_results, numQueries));
  CHECK_ERROR(memoryUtil::cpyToDevice(query_keys.data(), d_queries, numQueries));

  GpuTimer query_timer;
  query_timer.timerStart();
  btree.searchKeys(d_queries, d_results, numQueries, SourceT::DEVICE);
  query_timer.timerStop();

  // Copy results back
  CHECK_ERROR(memoryUtil::cpyToHost(d_results, query_results.data(), numQueries));

  // Expected results
  std::vector<uint32_t> expected_results(numQueries, 0);
  for (int iKey = 0; iKey < numQueries; iKey++) {
    if (query_keys[iKey] < numKeys) {
      expected_results[iKey] = query_keys[iKey];
    }
  }

  // Validate
  EXPECT_EQ(expected_results, query_results);
  // cleanup
  cudaFree(d_keys);
  cudaFree(d_values);
  cudaFree(d_queries);
  cudaFree(d_results);
  btree.free();
}

TEST(BTreeMap, DeleteRandomKeys) {
  using key_t = uint32_t;
  using value_t = uint32_t;

  GpuBTree::GpuBTreeMap<key_t, value_t> btree;

  // Input number of keys
  size_t numKeys = 1 << 10;

  // Prepare the keys
  std::vector<key_t> keys;
  std::vector<value_t> values;
  keys.reserve(numKeys);
  values.reserve(numKeys);
  for (int iKey = 0; iKey < numKeys; iKey++) {
    keys.push_back(iKey);
  }

  // shuffle the keys
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(keys.begin(), keys.end(), g);

  // assign the values
  for (int iKey = 0; iKey < numKeys; iKey++) {
    values.push_back(keys[iKey]);
  }

  // Move data to GPU
  key_t* d_keys;
  value_t* d_values;
  CHECK_ERROR(memoryUtil::deviceAlloc(d_keys, numKeys));
  CHECK_ERROR(memoryUtil::deviceAlloc(d_values, numKeys));
  CHECK_ERROR(memoryUtil::cpyToDevice(keys.data(), d_keys, numKeys));
  CHECK_ERROR(memoryUtil::cpyToDevice(values.data(), d_values, numKeys));

  // Build the tree
  GpuTimer timer;
  timer.timerStart();
  btree.insertKeys(d_keys, d_values, numKeys, SourceT::DEVICE);
  timer.timerStop();

  // Generate a batch of keys to delete
  std::vector<key_t> keys_deleted;
  uint32_t numDeletedKeys = 512;
  keys_deleted.reserve(numDeletedKeys);
  std::shuffle(keys.begin(), keys.end(), g);  // shuffle the keys again

  // delete the last numDeletedKeys
  int starting_idx = keys.size() - numDeletedKeys;
  for (int iKey = starting_idx; iKey < keys.size(); iKey++) {
    keys_deleted.push_back(keys[iKey]);
  }

  // Move data to GPU
  key_t* d_keys_deleted;
  CHECK_ERROR(memoryUtil::deviceAlloc(d_keys_deleted, numDeletedKeys));
  CHECK_ERROR(
      memoryUtil::cpyToDevice(keys_deleted.data(), d_keys_deleted, numDeletedKeys));

  // Apply the deleteion batch to the btree
  btree.deleteKeys(d_keys_deleted, numDeletedKeys, SourceT::DEVICE);

  // Now we can apply deleteion by resizeing the vector
  keys.resize(starting_idx);

  uint32_t max_nodes = 1 << 19;
  key_t* h_tree = new uint32_t[max_nodes * NODE_WIDTH];
  uint32_t num_nodes = 0;
  btree.compactTree(h_tree, max_nodes, num_nodes, SourceT::HOST);

  // Validation
  validate_tree_strucutre(h_tree, keys, keys.size());
  // cleanup
  cudaFree(d_keys_deleted);
  cudaFree(d_keys);
  cudaFree(d_values);
  delete[] h_tree;
  btree.free();
}

TEST(BTreeMap, DeleteAllKeys) {
  using key_t = uint32_t;
  using value_t = uint32_t;

  GpuBTree::GpuBTreeMap<key_t, value_t> btree;

  // Input number of keys
  size_t numKeys = 1 << 10;

  // Prepare the keys
  std::vector<key_t> keys;
  std::vector<value_t> values;
  keys.reserve(numKeys);
  values.reserve(numKeys);
  for (int iKey = 0; iKey < numKeys; iKey++) {
    keys.push_back(iKey);
  }

  // shuffle the keys
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(keys.begin(), keys.end(), g);

  // assign the values
  for (int iKey = 0; iKey < numKeys; iKey++) {
    values.push_back(keys[iKey]);
  }

  // Move data to GPU
  key_t* d_keys;
  value_t* d_values;
  CHECK_ERROR(memoryUtil::deviceAlloc(d_keys, numKeys));
  CHECK_ERROR(memoryUtil::deviceAlloc(d_values, numKeys));
  CHECK_ERROR(memoryUtil::cpyToDevice(keys.data(), d_keys, numKeys));
  CHECK_ERROR(memoryUtil::cpyToDevice(values.data(), d_values, numKeys));

  // Build the tree
  GpuTimer timer;
  timer.timerStart();
  btree.insertKeys(d_keys, d_values, numKeys, SourceT::DEVICE);
  timer.timerStop();

  // Generate a batch of keys to delete which is all keys
  std::shuffle(keys.begin(), keys.end(), g);

  // Apply the deleteion batch to the btree
  btree.deleteKeys(d_keys, numKeys, SourceT::DEVICE);

  uint32_t max_nodes = 1 << 19;
  key_t* h_tree = new uint32_t[max_nodes * NODE_WIDTH];
  uint32_t num_nodes = 0;
  btree.compactTree(h_tree, max_nodes, num_nodes, SourceT::HOST);

  // Validation
  keys.clear();  // Deleting all keys
  validate_tree_strucutre(h_tree, keys, keys.size());
  // cleanup
  cudaFree(d_keys);
  cudaFree(d_values);
  delete[] h_tree;
  btree.free();
}

TEST(BTreeMap, DeleteNoKeys) {
  using key_t = uint32_t;
  using value_t = uint32_t;

  GpuBTree::GpuBTreeMap<key_t, value_t> btree;

  // Input number of keys
  size_t numKeys = 1 << 10;

  // Prepare the keys
  std::vector<key_t> keys;
  std::vector<value_t> values;
  keys.reserve(numKeys);
  values.reserve(numKeys);
  for (int iKey = 0; iKey < numKeys; iKey++) {
    keys.push_back(iKey);
  }

  // shuffle the keys
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(keys.begin(), keys.end(), g);

  // assign the values
  for (int iKey = 0; iKey < numKeys; iKey++) {
    values.push_back(keys[iKey]);
  }

  // Move data to GPU
  key_t* d_keys;
  value_t* d_values;
  CHECK_ERROR(memoryUtil::deviceAlloc(d_keys, numKeys));
  CHECK_ERROR(memoryUtil::deviceAlloc(d_values, numKeys));
  CHECK_ERROR(memoryUtil::cpyToDevice(keys.data(), d_keys, numKeys));
  CHECK_ERROR(memoryUtil::cpyToDevice(values.data(), d_values, numKeys));

  // Build the tree
  GpuTimer timer;
  timer.timerStart();
  btree.insertKeys(d_keys, d_values, numKeys, SourceT::DEVICE);
  timer.timerStop();

  // Generate a batch of keys to delete
  std::vector<key_t> keys_deleted;
  uint32_t numDeletedKeys = 512;
  keys_deleted.reserve(numDeletedKeys);

  for (int iKey = 0; iKey < numDeletedKeys; iKey++) {
    keys_deleted.push_back(numKeys + iKey);
  }
  std::shuffle(keys_deleted.begin(), keys_deleted.end(), g);

  // Move data to GPU
  key_t* d_keys_deleted;
  CHECK_ERROR(memoryUtil::deviceAlloc(d_keys_deleted, numDeletedKeys));
  CHECK_ERROR(
      memoryUtil::cpyToDevice(keys_deleted.data(), d_keys_deleted, numDeletedKeys));

  // Apply the deleteion batch to the btree
  btree.deleteKeys(d_keys_deleted, numDeletedKeys, SourceT::DEVICE);

  uint32_t max_nodes = 1 << 19;
  key_t* h_tree = new uint32_t[max_nodes * NODE_WIDTH];
  uint32_t num_nodes = 0;
  btree.compactTree(h_tree, max_nodes, num_nodes, SourceT::HOST);

  // Validation
  validate_tree_strucutre(h_tree, keys, keys.size());
  // cleanup
  cudaFree(d_keys_deleted);
  cudaFree(d_keys);
  cudaFree(d_values);
  delete[] h_tree;
  btree.free();
}

TEST(BTreeMap, ConcurrentOpsInsertOnly) {
  using key_t = uint32_t;
  using value_t = uint32_t;

  GpuBTree::GpuBTreeMap<key_t, value_t> btree;

  // Input number of keys
  size_t initialNumKeys = 1 << 10;
  size_t maxKeys = 1 << 20;

  // Prepare the keys
  std::vector<key_t> keys;
  std::vector<value_t> values;
  keys.reserve(maxKeys);
  values.reserve(maxKeys);
  for (int iKey = 0; iKey < maxKeys; iKey++) {
    keys.push_back(iKey);
  }

  // shuffle the keys
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(keys.begin(), keys.end(), g);

  // assign the values
  for (int iKey = 0; iKey < maxKeys; iKey++) {
    values.push_back(keys[iKey]);
  }

  // Build the tree
  btree.insertKeys(keys.data(), values.data(), initialNumKeys, SourceT::HOST);

  uint32_t maxNodes = 1 << 19;
  key_t* h_tree = new uint32_t[maxNodes * NODE_WIDTH];
  uint32_t numNodes = 0;
  btree.compactTree(h_tree, maxNodes, numNodes, SourceT::HOST);

  // Validation
  validate_tree_strucutre(h_tree, keys, initialNumKeys);

  // Now perform concurrent key insertion
  // Initialize the batch
  size_t insertionBatchSize = 1 << 10;
  assert((insertionBatchSize + initialNumKeys) <= maxKeys);
  std::vector<OperationT> ops(insertionBatchSize, OperationT::INSERT);

  // Perform the operations
  btree.concurrentOperations(keys.data() + initialNumKeys,
                             values.data() + initialNumKeys,
                             ops.data(),
                             insertionBatchSize,
                             SourceT::HOST);

  // Validation again
  numNodes = 0;
  btree.compactTree(h_tree, maxNodes, numNodes, SourceT::HOST);
  validate_tree_strucutre(h_tree, keys, initialNumKeys + insertionBatchSize);

  // cleanup
  delete[] h_tree;
  btree.free();
}

TEST(BTreeMap, ConcurrentOpsInsertNewQueryPast) {
  using key_t = uint32_t;
  using value_t = uint32_t;

  GpuBTree::GpuBTreeMap<key_t, value_t> btree;

  // Input number of keys
  size_t initialNumKeys = 1 << 10;
  size_t maxKeys = 1 << 20;

  // Prepare the keys
  std::vector<key_t> keys;
  std::vector<value_t> values;
  keys.reserve(maxKeys);
  values.reserve(maxKeys);
  for (int iKey = 0; iKey < maxKeys; iKey++) {
    keys.push_back(iKey);
  }

  // shuffle the keys
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(keys.begin(), keys.end(), g);

  // assign the values
  for (int iKey = 0; iKey < maxKeys; iKey++) {
    values.push_back(keys[iKey]);
  }

  // Build the tree
  btree.insertKeys(keys.data(), values.data(), initialNumKeys, SourceT::HOST);

  uint32_t maxNodes = 1 << 19;
  key_t* h_tree = new uint32_t[maxNodes * NODE_WIDTH];
  uint32_t numNodes = 0;
  btree.compactTree(h_tree, maxNodes, numNodes, SourceT::HOST);

  // Validation
  validate_tree_strucutre(h_tree, keys, initialNumKeys);

  // Now perform concurrent key insertion
  // Initialize the batch
  size_t insertionBatchSize = 1 << 10;
  size_t queryBatchSize = initialNumKeys;
  size_t totalBatchSize = insertionBatchSize + queryBatchSize;

  assert(totalBatchSize <= maxKeys);
  std::vector<OperationT> ops;
  ops.reserve(totalBatchSize);
  for (size_t op = 0; op < totalBatchSize; op++) {
    if (op < queryBatchSize) {
      ops[op] = OperationT::QUERY;
    } else {
      ops[op] = OperationT::INSERT;
    }
  }

  // Perform the operations
  btree.concurrentOperations(
      keys.data(), values.data(), ops.data(), totalBatchSize, SourceT::HOST);

  // Validate again
  numNodes = 0;
  btree.compactTree(h_tree, maxNodes, numNodes, SourceT::HOST);
  validate_tree_strucutre(h_tree, keys, initialNumKeys + insertionBatchSize);

  // Validate the query
  for (int iKey = 0; iKey < queryBatchSize; iKey++) {
    EXPECT_EQ(values[iKey], keys[iKey]);
  }

  // cleanup
  delete[] h_tree;
  btree.free();
}

TEST(BTreeMap, ConcurrentOpsInsertNewQueryPastShuffled) {
  using key_t = uint32_t;
  using value_t = uint32_t;

  GpuBTree::GpuBTreeMap<key_t, value_t> btree;

  // Input number of keys
  size_t initialNumKeys = 1 << 10;
  size_t maxKeys = 1 << 20;

  // Prepare the keys
  std::vector<key_t> keys;
  std::vector<value_t> values;
  keys.reserve(maxKeys);
  values.reserve(maxKeys);
  for (int iKey = 0; iKey < maxKeys; iKey++) {
    keys.push_back(iKey);
  }

  // shuffle the keys
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(keys.begin(), keys.end(), g);

  // assign the values
  for (int iKey = 0; iKey < maxKeys; iKey++) {
    values.push_back(keys[iKey]);
  }

  // Build the tree
  btree.insertKeys(keys.data(), values.data(), initialNumKeys, SourceT::HOST);

  uint32_t maxNodes = 1 << 19;
  key_t* h_tree = new uint32_t[maxNodes * NODE_WIDTH];
  uint32_t numNodes = 0;
  btree.compactTree(h_tree, maxNodes, numNodes, SourceT::HOST);

  // Validation
  validate_tree_strucutre(h_tree, keys, initialNumKeys);

  // Now perform concurrent key insertion
  // Initialize the batch
  size_t insertionBatchSize = 1 << 10;
  size_t queryBatchSize = initialNumKeys;
  size_t totalBatchSize = insertionBatchSize + queryBatchSize;

  assert(totalBatchSize <= maxKeys);
  std::vector<OperationT> ops;
  ops.reserve(totalBatchSize);
  for (size_t op = 0; op < totalBatchSize; op++) {
    if (op < queryBatchSize) {
      ops[op] = OperationT::QUERY;
    } else {
      ops[op] = OperationT::INSERT;
    }
  }

  // shuffle all arrays
  // http://www.cplusplus.com/reference/algorithm/shuffle/
  for (auto i = (keys.begin() - keys.end()) - 1; i > 0; --i) {
    std::uniform_int_distribution<decltype(i)> d(0, i);
    auto randomizer = d(g);
    std::swap(keys[i], keys[randomizer]);
    std::swap(values[i], values[randomizer]);
    std::swap(ops[i], ops[randomizer]);
  }

  // Perform the operations
  btree.concurrentOperations(
      keys.data(), values.data(), ops.data(), totalBatchSize, SourceT::HOST);

  // Validate again
  numNodes = 0;
  btree.compactTree(h_tree, maxNodes, numNodes, SourceT::HOST);
  validate_tree_strucutre(h_tree, keys, totalBatchSize);

  // Validate the query
  for (int iKey = 0; iKey < totalBatchSize; iKey++) {
    if (ops[iKey] == OperationT::QUERY) {
      EXPECT_EQ(values[iKey], keys[iKey]);
    }
  }

  // cleanup
  delete[] h_tree;
  btree.free();
}
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
