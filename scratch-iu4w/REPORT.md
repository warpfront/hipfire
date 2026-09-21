# gfx1201 iu4 fragment-order W experiment

## Bottom line

**Do not land this lever.** A load-time, size-preserving fragment-order W image plus direct register W loads improves the controlled gate-up + down standalone aggregate from **250.015 to 253.201 TOPS-equivalent (+1.274%)**, but loses in the warmed product daemon: pp8192 is **3606.4 -> 3595.0 tok/s (-0.316%)** and 5,909-token TTFT is **1669.73 -> 1674.97 ms (+5.24 ms paired; +3.97 ms versus the 1,671 ms reference)**. Decode is flat at **36.6131 -> 36.6118 tok/s** and the serve battery is clean, so this is a performance rejection rather than a correctness failure.

Together with the folded-fp8 sibling's neutral result and the measured fold-free+preshuffle win (186.1 versus 167.9 TF), the result is consistent: **fragment-order W pays materially only when the K128 fold is absent.** On folded iu4, removing W staging merely trades it for more A-fragment reads and leaves the expensive fold intact.

No daemon trace was taken after the paired daemon gate rejected the lever. The record's 197 us/token GEMM sum is about 247 TOPS-equivalent (`48.7 GOP/token / 197 us`); this candidate did not improve end-to-end prefill and therefore does not replace that record.

## Pure iu4 WMMA ceiling

A register-resident `wmma_i32_16x16x32_iu4` microbenchmark uses eight independent accumulator chains per wave, 256-thread workgroups, 10,000 loop trips, one untimed warm-up, and five event-timed launches (median). Each wave instruction is counted as `2*16*16*32 = 16,384` integer operations. Dynamic LDS alone constrains the first row to the GEMM's actual three-block/CU occupancy; the second row removes that constraint and uses the occupancy API maximum.

| Mode | Active blocks/CU | Dynamic LDS | Median ms | Throughput |
|---|---:|---:|---:|---:|
| GEMM occupancy | 3 | 21,845 B | 1.448 | **695.085 TOPS** |
| Maximum occupancy | 8 | 0 B | 3.890 | **690.040 TOPS** |

The production GEMM's ~247 TOPS-equivalent is 35.5% of the three-block ceiling. A 5,000 tok/s target requires about 424 TOPS-equivalent, 61.0% of peak. The target is not ruled out by hardware throughput, and occupancy is not the constraint: maximum occupancy is slightly slower than the three-block run.

Source: `wmma_peak.hip`.

## Implementation

The experiment replaces eligible gfx1201 symmetric MQ4V2 resident images at load time; there is no second device copy. The CPU permutation buffer is temporary and VRAM delta is **0 bytes**. `HIPFIRE_IU4_WPRESHUFFLE=0` is the baseline override.

Payload byte `(row, k/2)`, for even nibble column `k`, maps to:

```text
((((row/16)*(K/64)+k/64)*32 + 16*((k/16)&1) + row%16)*16
 + 8*((k/32)&1) + (k%16)/2)
```

This gives lane `16*khalf + row_in_tile` one contiguous b128 containing its two K32 operands for a K64 slab. The four metadata bytes are stored in byte planes after the payload. The transform is bijective and size-preserving.

The candidate kernel changes the wave tile from two 16-row halves x one 64-token half to one 16-row tile x all 128 tokens. W is direct; A remains double-buffered in LDS. Dedicated symbol names and pointer metadata prevent a permuted allocation from reaching a row-major kernel.

## Consumer audit

Eligible resident tensors:

- MLP `gate_proj`, `up_proj`, `down_proj`;
- linear-attention `in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`;
- full-attention `q_proj`, `k_proj`, `v_proj`.

Converted consumers:

- prefill, batched prefill, and DFlash target verify: `gemm_mq4g256v2_residual_mmq_iu4{,_full_add,_full_set}_symfold_wp` through the existing plain/residual/fused projection dispatches;
- ordinary n=1 projection decode: `gemv_mq4g256v2_wp`;
- residual n=1 decode: `gemv_mq4g256v2_residual_wp`;
- fused MLP decode: `fused_gate_up_mq4g256v2_wp`.

The Qwen decode forward uses `qkvza_via_execute_steps` (and the analogous full-attention dispatch), so the eligible Q/K/V/Z/A/B tensors reach the converted generic GEMV readers. DFlash verify uses the same batched projection/GEMM routes as prefill.

Intentionally row-major and excluded at load: lm_head/output, embeddings, routers and gathered MoE weights, full-attention `o_proj`, and TP column slices. Their `load_weight_tensor_raw` calls pass `false`, so lm_head and unconverted specialized readers cannot observe a permuted allocation.

Audit method: loader callsite enumeration, attempted LSP reference lookup on the pointer metadata accessor, and text enumeration of MQ4V2 GEMV/GEMM, lm_head, and DFlash verify families. Runtime receipts are tg128 parity and the clean five-prompt battery below.

## Resource and ISA table

| Property | Baseline symfold | Fragment-order W |
|---|---:|---:|
| Threads / waves | 256 / 8 | 256 / 8 |
| Logical SGPR (`full_set`) | 40 | 34 |
| Logical VGPR | 182 | 179 |
| Dynamic LDS | 20,480 B | 12,288 B |
| Scratch / spills | 0 / 0 | 0 / 0 |
| Compiler waves/SIMD | 8 | 8 |
| Measured active blocks/CU | 3 | 3 |
| iu4 WMMAs / wave / K64 | 16 | 16 |
| W global payload / lane / K64 | 2 x b64 staged | **1 x b128 direct** |
| W LDS operand reads / wave / K64 | 4 x b64 logical | **0** |
| A LDS operand reads / wave / K64 | 8 x b64 logical | 16 x b64 logical |
| W-path address VALU in the K64 drain | 0 (precomputed staging offsets) | **0** (base + immediate K64 offset) |
| W-path address VALU / WMMA | 0 | 0 |

The candidate assembly contains one `global_load_b128` for each direct K64 W word, no W `ds_load`, and 16 iu4 WMMAs per wave/K64. The saved compiler reports and ISA are `compile_{baseline,candidate}-hip-amdgcn-amd-amdhsa-gfx1201.s`.

## Exactness

CPU inversion proves byte equality after round-tripping the permutation for every product shape, including the 96-row qkvza tail:

```text
gate  M=17408 K=5120 PASS
up    M=17408 K=5120 PASS
down  M=5120  K=17408 PASS
qkvza M=16480 K=5120 PASS
qkv   M=14336 K=5120 PASS
```

The GPU oracle at `B=128, M=160, K=256` (including a 32-row edge tile) produces the same output hash on baseline and candidate:

```text
baseline fnv=9481066e3f2a7653
candidate fnv=9481066e3f2a7653
max_abs=0.0 mean_abs=0.0
```

The shipped artifact's WT2 c2 KLD is byte-identical between override-off and candidate:

```text
baseline  KLD=0.064864 NLL=2.275025 PPL=9.7282 md5=eb59d8e51d8633d748aaee3fde6c37ce
candidate KLD=0.064864 NLL=2.275025 PPL=9.7282 md5=eb59d8e51d8633d748aaee3fde6c37ce
```

The full c24 candidate was not rerun after byte-identical c2 and the performance rejection; the shipped iu4 reference remains 0.081199.

## Standalone timing

Card-A, exact logical product shapes, zero-filled native packed inputs, five warm-up launches then 20 individually event-timed launches per process. Three A/B pairs follow an additional fresh-binary warm-up. Values below are means of the three reported medians.

| Shape | Baseline us | Candidate us | Baseline TOPS-eq | Candidate TOPS-eq | Delta |
|---|---:|---:|---:|---:|---:|
| fused gate-up (`B=8192 M=34816 K=5120`) | 12663.423 | 12296.281 | 230.64 | 237.52 | +2.986% |
| down (`B=8192 M=5120 K=17408`) | 4858.979 | 5005.642 | 300.60 | 291.79 | -2.930% |
| aggregate | 17522.402 | 17301.923 | **250.015** | **253.201** | **+1.274%** |

The gate win is largely canceled by the down regression. Both remain at three active blocks/CU.

## Daemon gates

Card-E, same shipped artifact, graph on, fp8 KV, warmed kernel cache, three runs and one in-process warm-up per arm.

| Gate | Baseline | Candidate | Delta |
|---|---:|---:|---:|
| pp8192 | 3606.4 tok/s | 3595.0 tok/s | **-0.316%** |
| tg128@128 | 36.6131 tok/s | 36.6118 tok/s | -0.004% |
| 5909-token TTFT | 1669.730 ms | 1674.968 ms | **+5.239 ms** |
| 5909-token prefill | 3538.896 tok/s | 3527.828 tok/s | **-0.313%** |

Candidate samples:

```text
pp8192: 3598.7, 3595.0, 3589.0 tok/s
tg128: 36.6204, 36.6118, 36.5864 tok/s
TTFT: 1677.477, 1672.435, 1674.968 ms
full-prompt prefill: 3522.552, 3533.172, 3527.828 tok/s
```

## Serve battery

The five decoded previews were read, not merely counted: they contain a coherent Python merge implementation, the correct 150-mile arithmetic, an accurate axial-tilt explanation, coherent lighthouse prose, and sensible coding instructions. No exclamation-mark attractor or empty/runaway response appeared.

Verbatim battery result block:

```text
### RUN qwen3.8:27b-mq4-xt|off|battery  kv=auto sampling={'temperature': 1.0, 'top_p': 0.95, 'top_k': 20, 'min_p': 0.0, 'presence_penalty': 0.0} seed=None ###
  [code]t1  finish=stop   ctx=84     cached=0      gen=174  (think 44/ans 47w) prefill=67.7ms/1241.1tok/s decode=36.5tok/s tau=None | '```python\ndef merge_sorted(a, b):\n    """Merge two sorted lists into a single sorted list.'
  [reason]t2  finish=stop   ctx=95     cached=0      gen=169  (think 18/ans 51w) prefill=62.2ms/1527.2tok/s decode=36.5tok/s tau=None | 'Step 1: First part of the trip  \n\(60 \text{ mph} \times 2.5 \text{ hours} = 150 \text{ mi'
  [factual]t3  finish=stop   ctx=65     cached=0      gen=193  (think 92/ans 45w) prefill=57.9ms/1123.2tok/s decode=36.6tok/s tau=None | "The seasons are caused by Earth's axial tilt of about 23.5 degrees. As Earth orbits the Su"
  [prose]t4  finish=stop   ctx=73     cached=0      gen=278  (think 133/ans 81w) prefill=58.4ms/1250.2tok/s decode=36.5tok/s tau=None | 'Old Marnie climbed the storm-scarred rocks at dawn to find a cracked glass bottle glitteri'
  [instruct]t5  finish=stop   ctx=71     cached=0      gen=206  (think 94/ans 47w) prefill=58.4ms/1215.7tok/s decode=36.5tok/s tau=None | '1. Keep functions small and focused on a single responsibility.\n2. Use clear, descriptive '
[qwen3.8:27b-mq4-xt|off|battery DONE] turns=5 runaway=0 empty=0 attractor=0 retrieval_miss=0 avg_prefill=1271.5tok/s avg_decode=36.5tok/s
  prompt_md5=43ca0d15712d3dfb777b51ae76d8fd5f request_md5=b45624e909a1eb3f40f63272c1f41355 step=
  prompt_md5=640e0fd4f55996cb175a422f0a12cef5 request_md5=98908f629ef79dce205ca9aea163085b step=
  prompt_md5=8f66b4c97988825bd8e7840aaf44357e request_md5=9502b5d5a6579d96c0a00e350be5375f step=
  prompt_md5=8fe0ad36f61bcf4992cc9df81cdf3817 request_md5=cfbd00094fb95f078fad91980cc566bf step=
  prompt_md5=8bed8e2d056dc1d47dccae9d32dbecf4 request_md5=ae84c5a3746b4b9a8247c6601ff2687c step=
  daemon_binary_md5=966a7214a2226e2189e062333cebecaf path=/home/kaden/ClaudeCode/warpfront/wt-iu4w/target/release/daemon
```

## Reproduction artifacts

- `prove_layout.cpp`: CPU bijection/inverse proof;
- `exact_iu4.hip`: baseline/candidate GPU exactness oracle;
- `bench_iu4.hip`: exact-shape standalone timing harness;
- `wmma_peak.hip`: pure iu4 WMMA ceiling;
- `compile_baseline.hip`, `compile_candidate.hip`, and saved gfx1201 ISA;
- `c2-{baseline,candidate}.kldseq`: byte-identical c2 receipts;
- `battery.json`: full decoded serve transcript.
