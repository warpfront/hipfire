# G0: gfx1100 fwht3-K/q8-V FA2 at deep context and wide B

**Verdict: GO for this standalone direct-kernel gate, not a production/default promotion.** The unchanged KMODE=3 body was bit-identical to the same body's 512-row segmentation at all 18 requested shapes, including B8192 ending at position 262143. For B2048/8192 the one-call body beat segmented512 in every one of three fresh-process ABBA blocks (observed 9.3–20.8% saving); B512 is literally the same body and grid in both arms, with ±0.8% event noise. Its worst observed paired factor against q8/q8 FA2 was **1.083×**, below the stated **1.10×** ceiling. No production dispatch, launcher, or HIP kernel changed. This does not establish a full-model/VMM/transcode/end-to-end result or gfx1151 eligibility.

## Provenance and experiment

Base `29469bb9107bb9eddb089b547af2a06b19f51028` (`gfx11-fa2-wide-v2`), isolated hipx worktree `/home/kaden/hipfire-fwht3g0` (`/home/kaden/ClaudeCode/warpfront/wt-fwht3g0`). Native compile:

```
/opt/rocm/bin/hipcc -O3 --offload-arch=gfx1100 -std=c++17 scratch-fwht3g0/fwht3_g0.hip scratch-fwht3g0/q8_module.hip -o scratch-fwht3g0/fwht3_g0
```

`fwht3_g0.hip` includes the **unchanged** production `attention_q8_0_fa2_gqa.gfx11.hip` at KMODE=3, the production `kv_cache_write_asym_k_fwht3_batched` K writer, and the production q8 V/K writer. `q8_module.hip` compiles the same FA2 source at KMODE=0 in a separate translation unit; only its symbol is renamed to coexist in this one executable. The unchanged FA2 source SHA-256 is `2c05b3587a173365295a6be4fa68e1755b8d06c8b3b35c84f676cead94067ba8`; benchmark binary SHA-256 `6f750d7f63121e977398444e93bb41305523516e6ccae1816d28abb659832eac` (binary removed after measurement). Device was asserted *before allocations* as HIP ordinal 0 `gcnArchName=gfx1100` in every process; resumed blocks explicitly exported `HIP_VISIBLE_DEVICES=0`, as do the retained runners. Never used HIP ordinal 3. Pre/post `rocm-smi --showpids` captured; post GPU[0] physical VRAM returned to ~28 MB baseline. Do not interpret its KFD `GPU(s)` process column as a physical card index under ROCR remapping.

Inputs are deterministic **synthetic, real-shape** H24/KV4/D256 f32 activations, not checkpoint activations. Nonconstant signed Q/K/V values are generated on-device by `fill_f32`; two deterministic ±1 sign tables are shared by the production fwht3 writer and the FWHT Q preconverter. For each shape the writer builds **every cache position 0..start+B−1** in 4096-row chunks, in production 400-B/position fwht3 K, 1088-B/position q8 V, and a second 1088-B/position q8 K made from the *same original K* for cost comparison. Each shape has fresh complete cache; no fake packed bytes or unread prefix. A final-position record check, 4096-B K/V/output tail sentinels, HIP launch checks, finite-output checks, and raw f32 bit comparison guard the test. Full Q16 scratch is allocated before running, 6/24/96 MiB for B512/2048/8192. Cache writing, fixture generation, H2D sign upload, Q preconversion, and allocation are **outside** the kernel timing; timings are FA2 **body kernel only**. The preconverter is the real KMODE-specific production function, not a host cast. The direct harness bypasses only host admission caps.

`F` = one unchanged fwht3 FA2 launch `[ceil(B/8),4,1]`, block128, dynamic LDS32768. `S` = 1/4/16 launches of the identical KMODE=3 body on complete *identical* K/V with Q16/output row and `positions+offset` advanced correctly for successive B512 slices; this is the reference fwht3 route for the full-output comparisons and equal-work performance screen, admitted **in this direct harness** past today's host cap. `Q` = one KMODE=0 q8/q8 FA2 launch on the separately production-written q8 K and same q8 V. For every `(start,B)`, three separate fresh-process `F-S-S-F` blocks; two warmups and five HIP-event samples per process, median of five per shape. One process handles all three B values at a given start/arm but exits before the next arm. `Q` is also measured in each fresh `F` process after its F samples; it is not itself an ABBA arm. The paired block value averages its two process medians; table shows the **median of three block values**, observed range of three within-block F-vs-S savings, and worst F/Q factor across the three blocks. Raw per-process and per-sample observations: `run.log`. An interrupted initial attempt at blocks 2–3 was excluded and fully restarted; only the complete three blocks in `run.log` enter these numbers. The measured `S` route is a counterfactual direct launch at deep context, **not** the current production slow tile+reduce dispatch path.

## Parity

Each cell shows `(context end, f32 bit mismatches / compared elements)` for the requested `start` and `B`; all outputs finite, first differing element none, maximum absolute difference exactly zero. Both arms use the same Q16 and completely populated cache. At B512 it is the identical single call; at B2048/8192 this establishes bit identity across launch widths and row offsets.

| Start position | B512 | B2048 | B8192 |
|---:|---:|---:|---:|
| 0 | 512, 0 / 3,145,728 | 2,048, 0 / 12,582,912 | 8,192, 0 / 50,331,648 |
| 32,768 | 33,280, 0 / 3,145,728 | 34,816, 0 / 12,582,912 | 40,960, 0 / 50,331,648 |
| 65,536 | 66,048, 0 / 3,145,728 | 67,584, 0 / 12,582,912 | 73,728, 0 / 50,331,648 |
| 131,072 | 131,584, 0 / 3,145,728 | 133,120, 0 / 12,582,912 | 139,264, 0 / 50,331,648 |
| 245,760 | 246,272, 0 / 3,145,728 | 247,808, 0 / 12,582,912 | 253,952, 0 / 50,331,648 |
| 253,952 (exact 262,144 end) | 254,464, 0 / 3,145,728 | 256,000, 0 / 12,582,912 | 262,144, 0 / 50,331,648 |

Additional deep batch tails B65/511/513/8191 at start253952 also had **zero mismatches**, finite values, and intact sentinels (`tail_probe.hip`; compared respectively 399,360 / 3,139,584 / 3,151,872 / 50,325,504 f32 values). `tile_probe.hip` additionally compared one 6,144-value last-query row at **each** of the six starts to the actual incumbent `attention_flash_fwht3_tile_batched` + `attention_flash_asym_reduce_batched` on these same production-writer bytes (tile size128). The maximum absolute difference across six samples was **2.78e−6**, below a declared `3e−6` absolute tolerance; no nonfinite values. This comparison is **not bit-exact**: the FA2 route casts signed-FWHT Q and dequantized K/V/P to f16, uses f16 WMMA QK/PV plus online softmax, while the scalar tile accumulates QK/V in f32 and applies a separate tile reduction (`attention_q8_0_fa2_gqa.gfx11.hip:59-60,930-966,625-665,857-887`; `attention_flash_fwht3_tile_batched.hip:70-122,125-145`; `attention_flash_asym_reduce_batched.hip:60-89`). These are the diverging operations; scalar numerical checks cover one row/context, while complete outputs use the bit-exact segmented reference. `graph_probe.hip` captured the actual FWHT Q-preconvert + body on an explicit stream and replayed twice bit-exact versus eager at deep context for B512, destroyed that graph, grew Q16 from 6 to 96 MiB, recaptured B8192 ending at 262144, and replayed twice bit-exact over 50,331,648 output values. It does not exercise the production wrapper's graph invalidation policy.

## Body kernel timings (ms)

| Start | B | F | S (seg512) | Q (q8/q8) | F saving vs S, 3-block range | Worst F/Q |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 512 | 0.385 | 0.387 | 0.383 | −0.0–0.4% | 1.014× |
| 0 | 2048 | 1.723 | 2.033 | 1.976 | 15.2–15.8% | 0.884× |
| 0 | 8192 | 26.761 | 31.126 | 25.374 | 11.2–14.0% | 1.064× |
| 32,768 | 512 | 16.753 | 16.882 | 15.602 | −0.8–1.4% | 1.080× |
| 32,768 | 2048 | 53.269 | 60.688 | 54.054 | 11.7–12.6% | 0.988× |
| 32,768 | 8192 | 233.872 | 264.242 | 237.078 | 11.1–11.5% | 0.987× |
| 65,536 | 512 | 32.854 | 32.729 | 30.332 | −0.4–−0.1% | 1.083× |
| 65,536 | 2048 | 107.597 | 118.974 | 108.924 | 9.3–9.6% | 0.988× |
| 65,536 | 8192 | 448.007 | 505.458 | 452.164 | 11.4–11.5% | 0.991× |
| 131,072 | 512 | 60.517 | 61.333 | 60.320 | 0.3–2.4% | 1.015× |
| 131,072 | 2048 | 215.718 | 241.704 | 219.574 | 10.0–11.1% | 0.989× |
| 131,072 | 8192 | 887.246 | 1,120.244 | 894.061 | 20.8–20.8% | 1.000× |
| 245,760 | 512 | 115.290 | 114.873 | 115.967 | −0.6–−0.1% | 0.998× |
| 245,760 | 2048 | 415.020 | 479.062 | 427.185 | 12.7–13.6% | 0.973× |
| 245,760 | 8192 | 1,709.549 | 2,117.115 | 1,709.377 | 17.9–19.5% | 1.007× |
| 253,952 | 512 | 119.152 | 119.342 | 120.404 | −0.2–0.6% | 0.995× |
| 253,952 | 2048 | 429.242 | 497.112 | 441.778 | 13.4–13.7% | 0.978× |
| 253,952 | 8192 | 1,770.598 | 2,166.671 | 1,763.227 | 18.0–18.7% | 1.010× |

The B512 F/S arms execute **the same kernel, one launch, same grid and bytes**; sub-1% signed variations are timing noise, not a distinct slow path. Deep/wide B2048 and B8192 beat the segmented reference in all 30 paired comparisons (five deep starts × two B × three blocks). This is a body-only comparison: production preconvert, cache writer, host overhead, graph construction and full-model wall time were deliberately not attributed to either arm. Q uses a different K codec; its numbers compare kernel cost, not equal outputs/model quality.

## Kernel and host limits (source `file:line`)

- The **current host gates**, not device bounds, are the obstacle: gfx11 fwht3 dispatch admits only `64..=32768` context and B512 except exact1024 on gfx1151 (`crates/hipfire-dispatch/src/families/attention.rs:1857-1868`); fwht3 launcher hard-rejects B>512 on gfx1100 and `max_ctx_len>32768` (`crates/rdna-compute/src/attention.rs:5784-5806`). Q8 launcher also hard-rejects context>32768 (`:5569-5575`). The harness calls kernel entries directly; no production cap was widened.
- Device causal bounds are signed i32 `positions[qr]`, reduced to `gmax`, then `seq_len=gmax+1` (`kernels/src/attention_q8_0_fa2_gqa.gfx11.hip:175-208`). Highest tested position is **262143** and `seq_len=262144`, far below `INT_MAX`. `tile*KT`, `ktile+KT`, and `gk` remain ≤262144 at KT32; guarded loads at K fill, V fill and next-tile K prefetch (`:234-237,298-312,408-415,447-480`). No int32 rollover at the 32768 cliff or 262144 end.
- FWHT3 K uses **64-bit pointer products** `gk*400ULL + kv_h*100` (`:299-312,466-479`), and q8 V `k0*row_stride` using `unsigned long long` (`:403-415`); the production K writer likewise uses `(size_t)pos*400` (`kernels/src/kv_cache_write_asym_k_fwht3_batched.hip:30-35,81-89`) and q8 V writer uses `kv_offset_for_k` (`kernels/src/kv_cache_write_q8_0_batched.hip:34-56`), whose product is explicitly `unsigned long long` (`kernels/src/kv_slot_desc.h:28-35`). At 262144 positions allocated fwht K=104,857,600 B and q8 V=285,212,672 B; neither reaches 32-bit signed byte indexing. Q/O use unsigned element offsets (`attention_q8_0_fa2_gqa.gfx11.hip:164-169,877-887`), largest valid local element index 50,331,647 at B8192, far below 2³². Preconvert's signed `batch_size*24` maximum196608 (`:938-947`). Caller must still check full allocation sizes before dispatch.
- KT32 LDS is `KT*1024 = 32,768 B` per body WG, independent of context/B (`:7-16,128-149`); body grid B8192 `[1024,4,1]` =4096 WGs (`:982-1025`). Q16 scratch grows with B only: B8192=100,663,296 B (`crates/rdna-compute/src/attention.rs:5849-5861`), not with context; body has no per-context scratch or split partials. Compiled gfx1100 metadata from `llvm-readobj --notes` on the extracted fatbin reports fwht3 body **212 VGPR, 26 SGPR, zero VGPR/SGPR spills and zero private fixed bytes**; FWHT preconvert 32 VGPR/zero spills; q8 FA2 body 224 VGPR/28 SGPR/zero spills. These are compiled facts for this native harness, not hypothetical q8 occupancy transferred to fwht3.

**Scope of GO.** This kills the *device-body deep/wide concern on gfx1100* under direct launches. It does not by itself authorize a K-only default: the host gate, production graph/scratch lifecycle, full-model quality and wall-time, VMM physical reclaim, and gfx1151 screen remain separate gates in the plan.
