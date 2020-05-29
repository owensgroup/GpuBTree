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

class BoolAllocator {
 public:
  BoolAllocator() {}
  ~BoolAllocator() {}
  void init() {
    CHECK_ERROR(memoryUtil::deviceAlloc(d_bool, MAX_SIZE));
    CHECK_ERROR(memoryUtil::deviceSet(d_bool, MAX_SIZE, 0x00));
    CHECK_ERROR(memoryUtil::deviceAlloc(d_count, 1));
    CHECK_ERROR(memoryUtil::deviceSet(d_count, uint32_t(1), 0x00));
  }
  void free() {
    CHECK_ERROR(memoryUtil::deviceFree(d_bool));
    CHECK_ERROR(memoryUtil::deviceFree(d_count));
  }

  BoolAllocator& operator=(const BoolAllocator& rhs) {
    d_bool = rhs.d_bool;
    d_count = rhs.d_count;
    return *this;
  }

  template<typename AddressT = uint32_t>
  __device__ __forceinline__ AddressT allocate() {
    return atomicAdd(d_count, 1);
  }
  template<typename AddressT = uint32_t>
  __device__ __forceinline__ uint32_t* getAddressPtr(AddressT& address) {
    return d_bool + address * 32;
  }
  template<typename AddressT = uint32_t>
  __device__ __forceinline__ void freeAddress(AddressT& address) {}

  __host__ __device__ uint32_t getCapacity() { return MAX_SIZE; }

  __host__ __device__ uint32_t getOffset() { return *d_count; }

 private:
  uint32_t* d_bool;

  uint32_t MAX_SIZE = 1 << 25;
  uint32_t* d_count;
};