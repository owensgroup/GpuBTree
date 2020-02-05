# GpuBTree
A GPU B-Tree optimized for updates.

# Publication
Muhammad A. Awad, Saman Ashkiani, Rob Johnson, Martín Farach-Colton, and John D. Owens. **Engineering a High-Performance GPU B-Tree**, In *Proceedings of the 24th ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming,* PPoPP 2019, pages 145–157, February 2019. [[Link]](https://ieeexplore.ieee.org/abstract/document/8425196)

## Cloning and Building
1. Clone: `git clone https://github.com/owensgroup/GpuBTree.git`
2. Update the CMakeLists.txt with the GPU hardware architecture
3. `mkdir build && cd build`
4. `cmake ..`
5. `make`

## Sample Driver Code
The repository contains two sample driver code for build and query operations.
To test the code after building you can run: `./bin/test_map numberOfKeys` `./bin/test_search numberOfKeys numberOfQueries`

#Limitaions
- 32-bit keys and values ranging between (1 to 2^32 - 3)

# Questions or Bug Report
This code was tested on an NVIDIA Tesla K40c and Volta Titan V GPUs. Please open an [issue](https://github.com/owensgroup/GpuBTree/issues) if you find any bugs or if you have any questions. This [issue](https://github.com/owensgroup/GpuBTree/issues/1) contains the planned future additions to this repository.





