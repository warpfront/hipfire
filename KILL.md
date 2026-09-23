# KILL: widened-chunk phantom rows (gfx1201-row-exact-chunk)

## Verdict
KILLED by measurement. A 5,909-token TTFT request on card-B runs as one
widened chunk (ceiling 8192, `commit_stride=512`) that is **already
row-exact at the host level**. There are no 2,283 phantom rows. The only
over-coverage is **107 sub-tile rows per iu4 GEMM launch** (47x128 = 6016
coverage, masked in-kernel), which host chunk planning cannot remove, and
which the single-chunk plan already minimizes (proof below). No code
change; branch left at base `96ec01887`, worktree clean.

## Phantom-row table (per kernel class, per 5909-row request)
Source: in-daemon `rocprofv3 --kernel-trace -f csv` of
`hipfire bench qwen3.8:27b-mq4-xt --ttft --prompt-file
benchmarks/prompts/ttft_5900.txt --runs 1 --warmups 1`
(trace `base3/hiptrx/814917_kernel_trace.csv`, 8022 dispatches = 2 identical
prefills; rocprofv3 reports Grid_Size as global work-items, so groups =
Grid_X / WG_X. `prefill_chunk: requested=8192 admitted=8192
commit_stride=512`.)

| class | batch-axis grid | rows doing work | real | phantom | mechanism |
|---|---|---|---|---|---|
| iu4 symfold GEMM `full_set` (368/req) | Y=47 tiles | 6016 | 5909 | **107, masked** | `batch_tiles=ceil(n/128)`; zero-filled slab + guarded writeback, `n_val=5909` |
| iu4 symfold GEMM `full_add` (128/req) | Y=47 tiles | 6016 | 5909 | **107, masked** | same (m=5120: wo + w_down, 40 m-tiles) |
| set-GEMM m-tiles (exact) | 136/96/80/48/8/1 WG | m exact | - | 0 | m in {17408,12288,10240,6144,1024,128}, all 128-multiples |
| rmsnorm/silu/gated-norm/sigmoid-rotate producers | 5909 WG / (68,5909) / (24,5909) / (24,5909) | 5909 | 5909 | 0 | row-parallel, K-groups exact (68=K/256, 24=K/256) |
| rope/deinterleave/epilogue/rmsnorm-final/embed | Y=5909 / 5909-factored | 5909 | 5909 | 0 | n bound as kernel arg |
| FA attention | 11x packet(512) + 1x tile(277) per FA layer | 5909 segs | 5909 | 0 | per-seg views; n%512!=0 skips packet-merge (correct) |
| FA tail KV scan | full-context reads | - | - | 0 | required causal math, not phantom; writes exact |
| GDN (scan/kkt/q8_fast/prep/conv) | 12 segs (11x512+277); q8_fast 1x whole-chunk n=5909 | 5909 | 5909 | 0 | seg views; tail 277 in [64,512) keeps chunk-scan |
| KV writes fp8 batched | 11x(4,512)+1x(4,277) x2(k,v) per FA layer | 5909 | 5909 | 0 | seg views |

Before/after: identical (no change). GEMM time dominates (1.17 s of
1.65 s kernel sum per prefill); the 107 masked rows cost ~1.8% of batch
MACs per launch (~21 ms TTFT upper bound) and are unreachable from host
planning (kernels out of scope).

## The three suspects (all innocent)
1. **Chunk plan / rung snapping** (`prefill.rs`): the rung only sets the
   *ceiling* (8192). Chunk = `remaining.min(ceiling)` = 5909;
   `owned_rows = limit.min(n)` = 5909-sized PBS. No padding; trace confirms
   (no 64-tile / 8192-row axis anywhere).
2. **Odd-length admissions `3f8fba35e` / `34df3f2b9`**: removed `%128`/`%64`
   gates so odd tails stay on fast paths with native partial-tile masking.
   They *removed* fallback work; introduced no padding.
3. **`gemm.rs` M padding** (`pad_f32_batch_to_128`, `pad_prefill_batch_128`,
   `pad_f32_batch_to_64`, `pad_prefill_batch`): used by Lloyd MMQ-LUT and
   FP8-LUT launchers only. The iu4-symfold route passes `n=5909` straight
   through (`batch_tiles = ceil`, `n_val = batch_size`). Zero host padding.

## Notes for the owner
- `ceil(5909/128)x128` = **6016**, not 5952. **5952 = pad-to-64**, i.e. the
  *fp8v2 arm's* host padding (`pad_f32_batch_to_64`), which still pads
  5909 -> 5952 (+43 rows + copies) per `_lloyd` launch despite `79bbf4e20`
  noting the fp8 kernels mask partial N tiles in-kernel. If anyone chases
  phantom rows further, that arm (not the iu4 ship arm) is where they are.
- Optimality: masked rows for any exact partition of 5909 into 128-tiles
  are `128 x #tiles - 5909 >= 107`, with equality only for the single
  chunk. The widened plan already attains the bound; splitting (e.g. for
  smaller FA tails) can only add masked rows and move GDN requant
  boundaries (eval-md5 break). The 512/277 commit sequence itself is
  trajectory-defining and untouchable.
- Unmapped (rows-exact, attribution uncertain, irrelevant to verdict):
  `full_set` m=128 x96/req and m=12288 x16/req; FA q-projection's
  `fused_qkv_mq4g256v2` (grid-stride, n-bound) alongside symfold sets.

## Evidence (base `96ec01887`, card-B `GPU-e475645fe0200397`, ab1)
- TTFT pair: 1681.1 ms (3515 tok/s), 1691.1 ms (3494 tok/s)
  (`--ttft --prompt-file benchmarks/prompts/ttft_5900.txt --runs 8
  --warmups 2 --json`; smoke 1642.5 ms).
- Matrix pair: pp512 3354.5/3346.9, pp8192 3596.7/3587.8,
  decode ctx128/tg128 36.52/36.53 tok/s.
- c2 (exact recipe): **0.070094**, md5
  `3d6a027724522065873a02362063788e` — matches the tree pin exactly.
- Serve battery 5/5, runaway=0 empty=0 attractor=0 retrieval_miss=0.
- Raw: `/tmp/phantom-trace/` (trace csv, bench jsons/logs, c2.bin).
