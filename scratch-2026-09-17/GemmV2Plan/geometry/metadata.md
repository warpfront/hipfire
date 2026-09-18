| kernel | geom | block | LDS | 2WG/CU | grid@N512 | wave | VGPR | SGPR | spill | scratch |
|---|---|---|---|---|---|---|---|---|---|---|
| gemm_gate_up_mq4g256v2_wmma_fp8_v2_b128x128_gfx1201 | 128x128x64/8w | 256 | 19968 | yes | [272, 4] | 32 | 201 | 31 | 0/0 | 0 |
| gemm_mq4g256v2_residual_wmma_fp8_v2_b128x128_gfx1201 | 128x128x64/8w | 256 | 19968 | yes | [40, 4] | 32 | 201 | 27 | 0/0 | 0 |
| gemm_qkv_mq4g256v2_wmma_fp8_v2_b128x128_gfx1201 | 128x128x64/8w | 256 | 19968 | yes | [112, 4] | 32 | 201 | 36 | 0/0 | 0 |
| gemm_qkvza_mq4g256v2_wmma_fp8_v2_b128x128_gfx1201 | 128x128x64/8w | 256 | 19968 | yes | [129, 4] | 32 | 201 | 36 | 0/0 | 0 |
| gemm_gate_up_mq4g256v2_wmma_fp8_v2_b128x64w4_gfx1201 | 128x64x64/4w | 128 | 14848 | yes | [544, 4] | 32 | 199 | 32 | 0/0 | 0 |
| gemm_mq4g256v2_residual_wmma_fp8_v2_b128x64w4_gfx1201 | 128x64x64/4w | 128 | 14848 | yes | [80, 4] | 32 | 199 | 28 | 0/0 | 0 |
| gemm_qkv_mq4g256v2_wmma_fp8_v2_b128x64w4_gfx1201 | 128x64x64/4w | 128 | 14848 | yes | [224, 4] | 32 | 199 | 37 | 0/0 | 0 |
| gemm_qkvza_mq4g256v2_wmma_fp8_v2_b128x64w4_gfx1201 | 128x64x64/4w | 128 | 14848 | yes | [258, 4] | 32 | 199 | 37 | 0/0 | 0 |
| gemm_gate_up_mq4g256v2_wmma_fp8_v2_gfx1201 | 256x64x64/8w | 256 | 24576 | yes | [544, 2] | 32 | 192 | 33 | 0/0 | 0 |
| gemm_mq4g256v2_residual_wmma_fp8_v2_gfx1201 | 256x64x64/8w | 256 | 24576 | yes | [80, 2] | 32 | 192 | 29 | 0/0 | 0 |
| gemm_qkv_mq4g256v2_wmma_fp8_v2_gfx1201 | 256x64x64/8w | 256 | 24576 | yes | [224, 2] | 32 | 196 | 38 | 0/0 | 0 |
| gemm_qkvza_mq4g256v2_wmma_fp8_v2_gfx1201 | 256x64x64/8w | 256 | 24576 | yes | [258, 2] | 32 | 196 | 38 | 0/0 | 0 |
| gemm_gate_up_mq4g256v2_wmma_fp8_v2_b64x256_gfx1201 | 64x256x64/8w | 256 | 25344 | yes | [136, 8] | 32 | 204 | 32 | 0/0 | 0 |
| gemm_mq4g256v2_residual_wmma_fp8_v2_b64x256_gfx1201 | 64x256x64/8w | 256 | 25344 | yes | [20, 8] | 32 | 204 | 28 | 0/0 | 0 |
| gemm_qkv_mq4g256v2_wmma_fp8_v2_b64x256_gfx1201 | 64x256x64/8w | 256 | 25344 | yes | [56, 8] | 32 | 204 | 37 | 0/0 | 0 |
| gemm_qkvza_mq4g256v2_wmma_fp8_v2_b64x256_gfx1201 | 64x256x64/8w | 256 | 25344 | yes | [65, 8] | 32 | 204 | 37 | 0/0 | 0 |
