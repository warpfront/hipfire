#define IU4_SYMMETRIC_FOLD 1
#define IU4_WPRESHUFFLE 1
#include "../kernels/src/block_i4_128_quant.hip"
#include "../kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx12.hip"
