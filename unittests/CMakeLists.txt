project (unittests)

add_executable(btree_map_test btree_map_test.cu)
set_target_properties(btree_map_test PROPERTIES
								FOLDER "unittests")
target_link_libraries(btree_map_test PRIVATE gpu_btree PRIVATE gtest)

