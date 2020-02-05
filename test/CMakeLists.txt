cuda_add_executable(test_map test_map.cu 
		OPTIONS ${GENCODE} ${CUDA_VERBOSE_PTXAS})

cuda_add_executable(test_search test_map_search.cu 
		OPTIONS ${GENCODE} ${CUDA_VERBOSE_PTXAS})		