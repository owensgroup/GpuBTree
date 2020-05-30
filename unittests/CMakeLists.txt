cuda_add_executable(btree_map_test btree_map_test.cu
					OPTIONS ${GENCODE} ${CUDA_VERBOSE_PTXAS}) 
target_link_libraries(btree_map_test gtest)