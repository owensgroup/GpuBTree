# GpuBTree
A GPU B-Tree optimized for updates.

***An improved implementation of the GPU B-Tree is now available at [MVGpuBTree](https://github.com/owensgroup/MVGpuBTree).***


## Publication
Muhammad A. Awad, Saman Ashkiani, Rob Johnson, Martín Farach-Colton, and John D. Owens. **Engineering a High-Performance GPU B-Tree**, In *Proceedings of the 24th ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming,* PPoPP 2019, pages 145–157, February 2019. [[Link]](https://escholarship.org/content/qt1ph2x5td/qt1ph2x5td.pdf?t=pkuy5m)

## Cloning and Building
1. Clone: `git clone https://github.com/owensgroup/GpuBTree.git`
2. `mkdir build && cd build`
3. `cmake ..`
4. `make`

## Sample Driver Code
The repository contains two sample driver code for [build](https://github.com/owensgroup/GpuBTree/blob/master/test/test_map.cu) and [query](https://github.com/owensgroup/GpuBTree/blob/master/test/test_map_search.cu) operations.
To test the code after building you can run: `./bin/test_map numberOfKeys` `./bin/test_search numberOfKeys numberOfQueries`

## Limitaions
- 32-bit keys and values ranging between (0 to 2^31 - 2)
- In general, 90% of the tree nodes will be leaf nodes, which will be (on average) 2/3 full. So if we [allocate 4 GiBs](https://github.com/owensgroup/GpuBTree/blob/master/src/allocator/pool_allocator.cuh#L75), then the tree will store up to 2.4 GiBs pairs (i.e., around 307.2 million 4-bytes keys before performing  any deletion).
## Questions or Bug Report
This code was tested on an NVIDIA Tesla K40c and Volta Titan V GPUs. Please open an [issue](https://github.com/owensgroup/GpuBTree/issues) if you find any bugs or if you have any questions. This [issue](https://github.com/owensgroup/GpuBTree/issues/1) contains the planned future additions to this repository.





