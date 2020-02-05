set(CUHFILES
	GpuBTree.h)

cuda_add_library(gpu_btree STATIC
	${CUHFILES}
	${CUFILES}
	OPTIONS ${GENCODE} ${VERBOSE_PTXAS})
