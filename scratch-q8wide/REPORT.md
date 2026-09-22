# Slice 0: wide Q8/f16 FA2 admission — evidence report

Worktree: `/home/kaden/ClaudeCode/warpfront/wt-q8wide`, branch `gfx1201-q8-fa2-wide`,
commit `2f186cfd3` (implementation) — five host files, no HIP source change.
Binaries (one build for all arms): daemon `32d1f92cf6cc5c85db63b3b13522f61f`,
hipfire `bf04e4b776ac73990c32151e890e743f`,
eval_hipfire `9555fa87f7a1fa6b8cc20db62aacdf38`.
Card B (`GPU-e475645fe0200397`), HOME `ab1`, `HIPFIRE_GRAPH=1` throughout.
Model: `/home/kaden/qcal/qat/v25/qwen3.8-27b.mq4v2.xt.sym-a035.qat-r7s200.hfq`
(symlink `~/.hipfire/models/qwen3.8-27b.mq4-xt`, unmodified).

## Diff summary by surface

1. `crates/hipfire-config/src/lib.rs`: new `process_bool_field!`
   `kernel.gfx12_q8_fa2_wide` / `HIPFIRE_GFX12_Q8_FA2_WIDE`, default false,
   experimental. Verified: `hipfire config get` → `false` (built-in) pre-set.
2. `crates/rdna-compute/src/feature_flags.rs`: new `pub gfx12_q8_fa2_wide`
   field; `from_lookup` resolves `HIPFIRE_GFX12_Q8_FA2_WIDE` default false on
   every arch; `for_test` false. Scoped tests: 14 feature_flags + 75 config pass.
3. `crates/rdna-compute/src/attention.rs`: wide ingress in
   `attention_q8_0_flash_prefill_wmma` (knob + master FA2 switch + exact
   gfx1201/H24/KV4/D256 + 64..=32768 rows with above-512 %512==0, no %16,
   64..=32768 ctx, no recorder/capture gates); legacy predicate byte-for-byte
   when off. Direct launcher range 1..=512 (off) vs 1..=32768 (on) for
   oracle/tail coverage.
4. `crates/hipfire-dispatch/src/families/attention.rs`: Q8-arm wide decision
   independent of the query16 default envelope (reaches ctx 64..255), after
   explicit flash-off / explicit scalar-or-batched / loaded-CK / spec-verify
   precedence; requires v_mode 8, window 0, no tree. 277 dispatch tests pass.
## Gate 1 — equivalence (oracle `q8_wide_oracle equiv`, card B)

One wide launch vs concatenated ≤512 launches on identical already-written Q8
buffers: bit-identical raw output (`to_bits` equality on all B×24×256 f32),
zero non-finite, unchanged input/cache bytes (FNV of q/K/V before == after).
Full log: `gate1_equiv.log`. 18/18 PASS, 0 FAIL:

| B | start | segs | mismatch | nonfinite | cache_unchanged |
|---|---|---|---|---|---|
| 64 | 0 | 2 (8+56) | 0 | 0 | true |
| 65 | 0 | 2 | 0 | 0 | true |
| 127 | 0 | 2 | 0 | 0 | true |
| 511 | 0 | 2 | 0 | 0 | true |
| 512 | 0 | 2 | 0 | 0 | true |
| 1024 | 0 | 2×512 | 0 | 0 | true |
| 8192 | 0 | 16×512 | 0 | 0 | true |
| 32768 | 0 | 64×512 | 0 | 0 | true |
| 1 | 0 | 1 | 0 | 0 | true |
| 7 | 0 | 1 | 0 | 0 | true |
| 8 | 0 | 1 | 0 | 0 | true |
| 9 | 0 | 2 (8+1) | 0 | 0 | true |
| 15 | 0 | 2 | 0 | 0 | true |
| 16 | 0 | 2 (8+8) | 0 | 0 | true |
| 17 | 0 | 2 (8+9) | 0 | 0 | true |
| 1024 | 1000 | 2 | 0 | 0 | true |
| 65 | 7 | 2 | 0 | 0 | true |
| 8192 | 256 | 16×512 | 0 | 0 | true |

Sub-512 splits at row 8 exercise the `q_base+=8` workgroup boundary; offset
positions (S=1000/7/256 over longer written prefixes) validate causal bounds
away from regular positions.

## Gate 2 — capture (oracle `q8_wide_oracle capture`, card B)

Eager vs hipGraph replay after launcher return + host-stack churn
(1 MiB stack trash + 64-deep 2 KiB-frame recursion between capture end and
replay). 3 replays per shape, bit-identical outputs, canaries intact, inputs
unchanged. Specified order with growth-with-live-graph. Full log:
`gate2_capture.log`.

| check | result |
|---|---|
| B512 capture + 3 replays (mismatch/canary/inputs) | PASS ×3 |
| graph alive after 512 capture | PASS |
| B8192 eager with live 512-graph → invalidated (exec gone, blobs 0) | PASS |
| B8192 rebuild + 3 replays, full-8192-row equality (new-shape proof) | PASS ×3 |
| ingress under `capture_mode` routes to `attention_q8_0_fa2_gqa_gfx1201`, output == direct | PASS |
| retained recording: begin → record 2 launches (preconvert+body) → finish → prefix replay, bit-identical | PASS |

hipGraph and retained-recording evidence are separate rows above (blobs=2 per
capture: Q preconvert + FA2 body).
## Gate 3 — route proof (rocprof daemon traces, HIPFIRE_GRAPH=1)

rocprofv3 `--kernel-trace --stats` on the engine-spawned daemon (manual
`hipfire serve`, one request per launch, TERM after response; bench teardown
SIGKILLs so wrapper+bench cannot flush — see `Drop for EngineInner`).
Grid sizes are work-items; divide GridX by block (128) for workgroups.
`LDS_Block_Size` reads 0 for every kernel (static-only column); the 65,536 B
dynamic LDS is grounded in the launch literal
(`attention.rs`: `[grid_x,4,1],[128,1,1],65536`). VGPR 240 observed on the
FA2 body (planning-time scoped compile: 238; same kernel, newer HIP 7.15).

### W arm, 5909-token pass (knob true, `trace-W`)

192 FA2 + 192 preconvert = 16 layers × (11×B512 segs + 1×B277 tail).
The 16 B277-tail FA2 calls (WG[35,4,1]) are the in-daemon wide-predicate
proof: the legacy `%16` guard rejects 277.

| symbol | WG grid | block | calls | total ms |
|---|---|---|---|---|
| attention_q8_0_fa2_gqa_gfx1201 | [64,4,1] (B512) | 128 | 176 | — |
| attention_q8_0_fa2_gqa_gfx1201 | [35,4,1] (B277) | 128 | 16 | — |
| attention_q8_0_fa2_gqa_gfx1201 total | | | 192 | 340.66 |
| attention_fa2_q_preconvert_gfx1201 | | 128 | 192 | 6.27 |
| attention_flash_q8_0_tile/reduce (1-token decode) | | | 16+16 | 1.30 |

Attention total: 346.9 ms / 5909 tok = 58.7 us/tok.

### Q control, 5909-token pass (knob false, `trace-Q`)

176 segmented FA2 (B512) + 16 incumbent WMMA tails (B277, `%16`-rejected) +
decode. The tail routes to `attention_q8_0_flash_prefill_wmma` (WG[18,24,1]).

| symbol | WG grid | block | calls | total ms |
|---|---|---|---|---|
| attention_q8_0_fa2_gqa_gfx1201 | [64,4,1] (B512) | 128 | 176 | 308.75 |
| attention_q8_0_flash_prefill_wmma | [18,24,1] (B277) | 32 | 16 | 68.62 |
| attention_fa2_q_preconvert_gfx1201 | | 128 | 176 | 5.99 |
| attention_flash_q8_0_tile/reduce (decode) | | | 16+16 | 1.33 |

Attention total: 383.4 ms / 5909 tok = 64.9 us/tok.

### W arm, 8283-token pass (knob true, `trace-W8192`)

First chunk 8192 → ONE whole-chunk FA2 call per layer at WG[1024,4,1]
(WI[131072,4,1]); 91-row tail → 16 wide-tail calls at WG[12,4,1].
Consolidation: 256 segmented calls → 16 whole-chunk (+16 tail) calls.

| symbol | WG grid | block | calls | total ms |
|---|---|---|---|---|
| attention_q8_0_fa2_gqa_gfx1201 | [1024,4,1] (B8192) | 128 | 16 | — |
| attention_q8_0_fa2_gqa_gfx1201 | [12,4,1] (B91) | 128 | 16 | — |
| attention_q8_0_fa2_gqa_gfx1201 total | | | 32 | 612.25 |
| attention_fa2_q_preconvert_gfx1201 | | 128 | 32 | 8.15 |
| attention_flash_q8_0_tile/reduce (decode) | | | 16+16 | 1.73 |

Attention total: 620.4 ms / 8283 tok = 74.9 us/tok.

### Q control, 8283-token pass (knob false, `trace-Q8192`)

256 segmented FA2 (16 layers × 16 B512 segs) + 16 incumbent WMMA tails (B91)
+ decode. No wide calls.

| symbol | WG grid | block | calls | total ms |
|---|---|---|---|---|
| attention_q8_0_fa2_gqa_gfx1201 | [64,4,1] (B512) | 128 | 256 | 608.43 |
| attention_q8_0_flash_prefill_wmma | [6,24,1] (B91) | 32 | 16 | 58.45 |
| attention_fa2_q_preconvert_gfx1201 | | 128 | 256 | 8.58 |
| attention_flash_q8_0_tile/reduce (decode) | | | 16+16 | 1.72 |

Attention total: 675.5 ms / 8283 tok = 81.6 us/tok. Whole-request kernels:
3224.3 ms total, 2547.1 ms non-attention.

### F arm, 8283-token pass (fp8, `trace-F8192`)

16 qresident whole-chunk calls at the plan's exact native grid WG[8,4,16],
block 768, VGPR 216 + 16 qresident tails at WG[2,4,1]. No Q8 symbols.

| symbol | WG grid | block | calls | total ms |
|---|---|---|---|---|
| attention_fp8_e4m3_fa2_gqa_qresident_gfx1201 | [8,4,16] (B8192) | 768 | 16 | — |
| attention_fp8_e4m3_fa2_gqa_qresident_gfx1201 | [2,4,1] (B91) | 768 | 16 | — |
| attention_fp8_e4m3_fa2_gqa_qresident_gfx1201 total | | | 32 | 183.05 |
| attention_flash_fp8_e4m3_tile + attention_flash_q8_0_reduce (decode) | | | 16+16 | 1.54 |

Attention total: 184.6 ms / 8283 tok = 22.3 us/tok. Whole-request kernels:
2309.5 ms total, 2124.9 ms non-attention.

### Attention budget (plan kill criterion 6, pp8192 shape)

Per 8283-token prefill+1-decode pass, traced kernel sums (graph-on serve):

| arm | attention | non-attention | total kernel | FA2-kind calls/pass |
|---|---|---|---|---|
| Q | 677.2 ms (81.6 us/tok) | 2547.1 ms | 3224.3 ms | 256 seg + 16 incumbent tails |
| W | 622.1 ms (74.9 us/tok) | 2549.2 ms | 3171.3 ms | 16 whole + 16 wide tails |
| F | 184.6 ms (22.3 us/tok) | 2124.9 ms | 2309.5 ms | 16 qresident + 16 tails |

W consolidates 256 → 16 calls and saves 55.1 ms of attention (8.1%) —
per-launch overhead is provably not the cost. F's attention is another
437.5 ms lower AND its non-attention pipeline is 424.3 ms lower (fp8
projections/writes vs Q8 tier). End-to-end gap F−W ≈ 861 ms kernel time,
matching the tok/s gap. With T_other_C ≈ 2549 ms > T_F/0.99 ≈ 2262 ms
(interim T_F = 8192/3658 ≈ 2239 ms from the sibling's same-commit F arm;
replaced by this build's F block in gate 4 below), the criterion-6 pp8192
budget is negative: no attention kernel on the Q8 tier can reach F parity —
the 424 ms non-attention tier gap alone forecloses it.
### F arm, 5909-token pass (fp8, `trace-F5909`)

176 qresident B512 calls at WG[8,4,1] + 16 B277-tail calls at WG[5,4,1],
block 768, VGPR 216. Attention total 101.70 + 1.24 decode ≈ 102.9 ms
= 17.4 us/tok.

### Attention budget, TTFT shape (5909 tokens)

| arm | attention | us/tok |
|---|---|---|
| Q | 383.4 ms | 64.9 |
| W | 346.9 ms | 58.7 |
| F | 102.9 ms | 17.4 |

TTFT criterion (1.01·T_F − T_other_C with this build's TTFT block, gate 4)
is evaluated against the wall-clock pairs below; the traced split already
shows F's attention ≈ 1/3 of W's on the identical shape.

## Gate 4 — serving gates (one build, three arms, graph on)

ABBA pairs per arm, TTFT pairs, c24 ×2 refs ×3 arms, battery 5/5 verbatim.
### Matrix ABBA: two [Q,W,F,F,W,Q] blocks (`--matrix --pp 512,8192 --ctx 128
--tg 128 --spec off --runs 3 --warmups 1 --json`, per-run medians)

| run | arm | pp512 | pp8192 | tg128 |
|---|---|---|---|---|
| Q1 | Q | 2836.8 | 2633.3 | 36.52 |
| W1 | W | 2823.8 | 2624.4 | 36.50 |
| F1 | F | 3348.6 | 3632.3 | 36.50 |
| F2 | F | 3340.7 | 3631.4 | 36.48 |
| W2 | W | 2827.4 | 2626.3 | 36.50 |
| Q2 | Q | 2825.2 | 2616.4 | 36.51 |
| Q3 | Q | 2825.5 | 2615.3 | 36.53 |
| W3 | W | 2822.2 | 2624.3 | 36.52 |
| F3 | F | 3347.1 | 3632.8 | 36.52 |
| F4 | F | 3351.3 | 3632.4 | 36.49 |
| W4 | W | 2828.8 | 2622.0 | 36.53 |
| Q4 | Q | 2824.2 | 2614.4 | 36.52 |

W-vs-Q pp8192: block A (Q1,W1,W2,Q2) +0.02%; block B (Q3,W3,W4,Q4) +0.32%;
pooled +0.17%. W-vs-F pp8192: −27.7%. F pp8192 rock-stable (spread 1.4 =
0.04%); Q8 arms drift down across the session (Q1 2633 → Q4 2614).
Decode identical across arms (36.50–36.53; B=1 path untouched).
pp512: Q ≈ W ≈ 2826 (no regression, within 1%); F 3347.
Route status of the bench pp rows: VRAM high-water probe below.

### VRAM high-water probe (bench-shaped route proof)

INTERRUPTED by time-box (NOT MEASURED). Question it was answering: whether
bench pp rows take the whole-chunk path (96 MB Q16 high-water) or segmented
(6 MB). Oracle source archived as `q8_wide_oracle.rs` in this directory.
## Tooling findings (outlive this slice)

1. `Drop for EngineInner` (`crates/hipfire-client/src/lib.rs:1196-1203`)
   SIGKILLs the daemon child at bench teardown (`child.kill()` + `wait`).
   Consequence: `scripts/rocprof-daemon-wrap.sh` + `hipfire bench` can NEVER
   yield rocprof CSVs — rocprofv3 is SIGKILLed with its target before it
   flushes (verified: empty output dir, exit 0, full bench numbers present).
   Gate-3 traces in this report were taken via manual `hipfire serve` with
   the wrapper as `HIPFIRE_DAEMON_BIN` (engine-spawned daemon is the rocprof
   target), one request per launch, TERM after the response.
2. `hipfire serve` (manual) + rocprof + `HIPFIRE_GRAPH=1` decode: the first
   traced serve answered one 5909-token request normally, then both gateway
   and daemon processes vanished before a second request (connection
   refused; rocprof still flushed complete CSVs). Bench-driven daemons on
   the same build survive full matrix runs, and plain (unwrapped) manual
   serve survives multiple requests — so the disappearance is specific to
   the rocprof + graph-decode combination, not the wide route. Pattern used
   throughout: one request per launch.

## Measurement status

| cell | state |
|---|---|
| Gate 1 equivalence, 18 shapes | PASS (bit-identical, caches unchanged) |
| Gate 2 hipGraph replay B512/B8192 + canaries + growth-invalidation | PASS |
| Gate 2 retained-recording replay | PASS (kept separate from hipGraph) |
| Gate 2 ingress under capture_mode → FA2 symbol | PASS |
| Gate 3 W route positively confirmed (serve path) | YES — 16 whole-chunk calls at WG[1024,4,1], block 128, GridY 4, plus 16 wide tails; B277/B91 tails prove the no-`%16` predicate in-daemon |
| Gate 3 Q control | traced (256 seg FA2 + 16 incumbent tails) |
| Gate 3 F arm | traced (16 qresident WG[8,4,16], block 768, VGPR 216) |
| Gate 3 LDS | source-grounded (65536 launch literal); rocprof `LDS_Block_Size` reads 0 for all kernels (static-only column) |
| Gate 4 matrix ABBA (12 runs, one build) | DONE — W-vs-Q pp8192 pooled +0.17% (blocks +0.02%/+0.32%); W-vs-F −27.7%; decode identical |
| Early W pair-1 pp8192 2653 | SUPERSEDED early single (pre-ABBA); ABBA W range 2622–2626 is the number |
| Bench-workload whole-chunk route (VRAM high-water probe) | NOT MEASURED (probe interrupted by time-box; bench W≈Q therefore has two live readings: true ~0–1% whole-chunk effect, or bench running segmented) |
| Gate 4 TTFT pairs | NOT MEASURED |
| Gate 4 c24 ×2 refs ×3 arms | NOT MEASURED |
| Gate 4 battery 5/5 + verbatim texts | NOT MEASURED |

## Verdict

NONE (per time-box; no verdict is rendered). The measured shape so far: W
reaches F nowhere (pp8192 −27.7%, attention 622 ms vs 185 ms, plus a 424 ms
non-attention tier gap), and W-vs-Q is +0.17% pooled on bench with the
bench-route caveat above. Funding decision is Main's.
