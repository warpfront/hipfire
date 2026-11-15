# gfx942 wide-lane MoE glue: bitwise A/B harness

`gluetest.cpp` loads three code objects and compares, buffer for buffer and bit for bit, the
`qwen4_g942_moe_wide` entries against the kernels they replace, then times both:

| new entry (module `qwen4_g942_moe_wide`) | reference |
|---|---|
| `qwen4_g942_moe_wide_router` | `moe_router_softmax_top10_f32` (incumbent, BF16 recipe) |
| `qwen4_g942_moe_wide_group_scatter_desc` | `qwen4_g942_moe_group_scatter_desc` (module `qwen4_g942_glue_moe`, fold 1) |
| `qwen4_g942_moe_wide_unscatter_act` | `qwen4_g942_moe_unscatter_act` (fold 2) |
| `qwen4_g942_moe_wide_combine_shared_add` | `qwen4_g942_moe_combine_shared_add` (fold 3) |

Inputs: router logits of eight kinds (gaussian, coarse ties, all equal, +-0 / tiny values, huge
magnitudes, NaN / Inf injection, large-negative rows, BF16-tied rows) at 8..64 tokens with
`normalize_topk_prob` in {0, 1}; routes of ten distinct experts per token (uniform, hot-set, all
tokens on the same experts, top of the range) with 1 % unrouted markers (-1, 512, 99999, ...);
random grouped activations, shared rows, weights and residuals. Every output buffer is
poison-filled first, so an unwritten word is a mismatch too.

```
F="--genco --offload-arch=gfx942 -O3 --no-offload-compress -fuse-cuid=none -mcode-object-version=6"
hipcc $F -o wide.co   kernels/src/qwen4_g942_moe_wide.gfx942.hip
hipcc $F -o router.co kernels/src/moe_router_softmax_top10_f32.hip
hipcc -O2 -std=c++17 --offload-arch=gfx942 -o gluetest docs/investigations/2026-11-12-g942-moe-wide-glue/gluetest.cpp
./gluetest kernels/qwen4_g942_glue_moe_hipcc_gfx942.co router.co wide.co 800   # exit 0 = PASS
```

Receipt (MI300X, 800 router trials, 201 grouped-glue trials, 4410 compared buffers):
0 mismatches. Back-to-back launch times at 32 rows (320 slots, 5,120 grouped rows):

| kernel | before (us) | after (us) |
|---|---:|---:|
| router | 14.2 | 7.0 |
| group + scatter + tile descriptors | 16.3 | 4.7 |
| unscatter + shared activation | 8.9 | 3.0 |
| combine + shared add | 10.5 | 7.7 |
