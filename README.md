# GpuBTree
A GPU B-Tree optimized for updates.

## Publication
Muhammad A. Awad, Saman Ashkiani, Rob Johnson, Martín Farach-Colton, and John D. Owens. **Engineering a High-Performance GPU B-Tree**, In *Proceedings of the 24th ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming,* PPoPP 2019, pages 145–157, February 2019. [[Link]](https://ieeexplore.ieee.org/abstract/document/8425196)

## Cloning and Building
1. Clone: `git clone https://github.com/owensgroup/GpuBTree.git`
2. Update the [CMakeLists.txt](https://github.com/owensgroup/GpuBTree/blob/master/CMakeLists.txt#L39) with the GPU hardware architecture
3. `mkdir build && cd build`
4. `cmake ..`
5. `make`

## Sample Driver Code
The repository contains two sample driver code for [build](https://github.com/owensgroup/GpuBTree/blob/master/test/test_map.cu) and [query](https://github.com/owensgroup/GpuBTree/blob/master/test/test_map_search.cu) operations.
To test the code after building you can run: `./bin/test_map numberOfKeys` `./bin/test_search numberOfKeys numberOfQueries`

## Limitaions
- 32-bit keys and values ranging between (0 to 2^31 - 2)

## Questions or Bug Report
This code was tested on an NVIDIA Tesla K40c and Volta Titan V GPUs. Please open an [issue](https://github.com/owensgroup/GpuBTree/issues) if you find any bugs or if you have any questions. This [issue](https://github.com/owensgroup/GpuBTree/issues/1) contains the planned future additions to this repository.





