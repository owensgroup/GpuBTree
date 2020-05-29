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

#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <random>
#include <vector>

#include "GpuBTree.h"

int main(int argc, char* argv[]) {
  GpuBTree::GpuBTreeMap<uint32_t, uint32_t, uint32_t> btree;

  // Input number of keys
  uint32_t numKeys = 1 << 20;
  if (argc > 1)
    numKeys = std::atoi(argv[1]);

  // RNG
  std::random_device rd;
  std::mt19937 g(rd());

  ///////////////////////////////////
  ///		 Build the tree    	  ///
  ///////////////////////////////////

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

  ///////////////////////////////////
  ///		 Query the tree       ///
  ///////////////////////////////////

  // Input number of queries
  uint32_t numQueries = numKeys;
  if (argc > 2)
    numQueries = std::atoi(argv[2]);
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

  // Validate
  uint32_t exist_count = 0;
  for (int iKey = 0; iKey < numQueries; iKey++) {
    if (query_keys[iKey] < numKeys) {
      exist_count++;
      if (query_results[iKey] != query_keys[iKey]) {
        printf("Error validating queries (Key = %i, Value = %i) found (Value = %i)\n",
               query_keys[iKey],
               query_keys[iKey],
               query_results[iKey]);
        exit(0);
      }
    } else {
      if (query_results[iKey] != 0) {
        printf(
            "Error validating queries (Key = %i, Value = NOT_FOUND) found (Value = %i)\n",
            query_keys[iKey],
            query_results[iKey]);
        exit(0);
      }
    }
  }

  // output
  printf("SUCCESS. ([%0.2f%%] queries exist in search.)\n",
         float(exist_count) / float(numQueries) * 100.0);

  printf("Build: %i pairs in %f ms (%0.2f MKeys/sec)\n",
         numKeys,
         build_timer.getMsElapsed(),
         float(numKeys) * 1e-6 / build_timer.getSElapsed());

  printf("Query: %i pairs in %f ms (%0.2f MKeys/sec)\n",
         numQueries,
         query_timer.getMsElapsed(),
         float(numQueries) * 1e-6 / query_timer.getSElapsed());

  // cleanup
  cudaFree(d_keys);
  cudaFree(d_values);
  cudaFree(d_queries);
  cudaFree(d_results);
  btree.free();
  return 0;
}