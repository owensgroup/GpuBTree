project (test)

add_executable(test_map test_map.cu)
set_target_properties(test_map PROPERTIES
								FOLDER "tests")
target_link_libraries(test_map PRIVATE gpu_btree)


add_executable(test_search test_map_search.cu)
set_target_properties(test_search PROPERTIES
								FOLDER "tests")
target_link_libraries(test_search test_map_search gpu_btree)