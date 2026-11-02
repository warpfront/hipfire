# Dense 27B IU4 prefill: GPU + NPU co-op integration (IEF15)

> **Historical research note: superseded by the 0.4.2 serving route.** This file records a design study (no hardware run) for splitting dense 27B IU4 prefill GEMM columns between the GPU and the NPU, whose mixed-mode numeric contract uses IEF15 on the NPU columns from the 2026-10 NPU research cycle. It is not current guidance, and any figure in it is a dated fixture-bound measurement or an explicitly labelled historical estimate, not a product claim. The shipped 0.4.2 NPU route is the Flash-Next gfx1151 trunk input projection described in [npu-trunk.md](npu-trunk.md): `HIPFIRE_QWEN4_NPU_TRUNK=off|auto|on|emulate` (default `auto`, effectively `off` on every architecture except gfx1151), calibration schema 4, BF16-direct only. There is **no I32 serving contract** (no int32 partial, combine or fold exists in the product route). The out-projection program exists but is **not wired** into serving; `qsa.o` and the GDN o-projection are **not shipped**. Environment-variable names and flag syntax below are historical proposals or experiment switches, not supported configuration.
>
> Nothing here was implemented as a serving route; the measured prototypes that followed ([coop-down27.md](coop-down27.md), [coop-epi-fuse.md](coop-epi-fuse.md)) were killed. The `W:` source references point at the 0.4.1 beta checkout. The GPU-side switches named below (`HIPFIRE_V2B_ADDEPI`, `HIPFIRE_V2B_ZBA_SCATTER`) are developer-only GPU kernel selectors of that checkout and are not NPU configuration.

Design only, 2026-10-03. No hardware, builds or tests were run for this document. Scope is dense 27B pp8192 on Halo (gfx1151); Flash-Next is out of scope.

Path conventions: `W:` = the 0.4.1 hipfire beta checkout (read-only). Bare paths are relative to the NPU research workspace (this directory holds its design notes; `logs/` and `tools/` are as in that workspace). `FC` = [fold-contract.md](fold-contract.md). `D27` = `docs/dense27b.md`. `LOG` = `logs/coop/coop-w2-0fe69f9.log`.

**MEASURED** means a recorded observation with a log citation. **ESTIMATE** is derived arithmetic or an assumed parameter. Every modelled quantity below is ESTIMATE, including each time, tok/s and saving figure.

**New-IEF geometry status.** The TM64/TN32 IEF15 NPU tile has no measured rate, tile cost or wave geometry yet; nor has the column-windowed current-fold GPU entry (§1). Everything about them below is pending. Rates are borrowed from the V8 busy rate on a different kernel (see §4).

**Contract: mixed mode (project decision, 2026-10-03).** G0 is **KILLED on measurement**: the IEF15 GPU route ran at 1.565–1.571× the incumbent (`exp/ief15-g0` @ 7559ee957; `FC` §1), so the GPU does **not** adopt IEF15. The plan is mixed: the **GPU keeps the incumbent fold** (`fmaf(RN(sc·d), C_e, sum)`, `D27` §2) on its columns, and the **NPU runs IEF15** (`FC` §3) on its own columns. Consequently:
- The **fixed column-split map** is **part of numeric identity** (`FC` §4): bind exact sorted per-projection ranges, paired g/u cuts, K segments, output layout, contract/final-pack version and model/weight identity into NPU program/tape, graph-capture, prefill-cache and run keys. No runtime load/thermal/adaptive repartition within that identity; a capture under one map never replays under another.
- **Fallback fails closed or recomputes the same columns under IEF15.** If the NPU path is unavailable, either the step errors (no silent switch) or the NPU-owned columns are recomputed under IEF15 on another engine (e.g. the measured-slower GPU IEF15 entry) so the output bytes are unchanged. Falling back to the incumbent fold on NPU columns would change the numeric identity and is forbidden.
- Modelled payoff of this mode: `tools/npu/fold-model/payoff.json` (md5 f844e93bee923c9b32d7a5d8f19f4ebd), §4.

## 1. Column split per GEMM

NPU owns **whole output features** (weight rows), statically per GEMM, as a leading window `[0,n)`. GPU owns the rest under the incumbent fold. Current V2B admission needs 256-row cuts, and gate_up needs 256 paired hidden features = 512 combined rows (`FC` §7); this document's own model searches the coarser 512-row / 1024-combined grid. NPU wire output is f32 `[token][local_feature]` (`FC` §3); placing it into a full-strided shared C arena requires a proved placement adapter, otherwise compact-output GPU scatter must be priced. NPU/GPU SET regions are disjoint and line-aligned. Each command's arena base, layout and range are fixed in its identity (ring v1: `docs/npu/railgun-npu.md:62-95`).

| Family (calls/step) | M×N×K, GPU g ms/call | Callsite (`W:crates/hipfire-arch-qwen35/src/qwen35/prefill.rs` unless noted) | NPU window, fast cut | Epilogue contract |
|---|---|---|---|---|
| GDN qkv (48) | 8192×10240×5120, 11.463 | `:5743-5760` fold, `:5778-5796` non-fold; `gemm.rs:31957` | n=2048 | SET f32 |
| GDN z+β+α (48) | 8192×6240 useful (6400 padded)×5120, 7.604 | `:5602-5617,5743-5760`; scatter `gemm.rs:32046-32105`; padding rule `:31571-31574` | n=1536, Z rows only | SET Z. Logical 6240 = 6144 Z + 48 β + 48 α. Dispatch pads to 6400, so 160 rows are discarded. NPU never owns β/α or padding |
| FA q (16) | 8192×12288×5120, 13.722 | `:8108-8123`; `gemm.rs:32630-32655` | n=2560 | SET. `wq.m = 2·q_dim` includes the output gate (`batch.rs:101,350`). Windows must hold **complete head windows** of Q+gate. [INFERENCE] 5 heads × 512 rows for head_dim 256; confirm against `batch.rs:101,350` |
| FA k,v (32) | 8192×1024×5120, 1.386 | `:8108-8123` | none (n=0) | stays on GPU |
| o_proj (64) | 8192×5120×6144, 6.706 | `:243-262,8816-8817` | n=1024 | SET `delta`, then the GPU deferred `RN(x+delta)` fold |
| gate_up (64) | 8192×34816 (two 17408 halves)×5120, 38.547 | `:7428-7456,9658-9686`; `gemm.rs:33869-34000` | n=7168 total = G[0,3584) + U[0,3584) | NPU emits **separate f32 pre-SiLU g and u**. The GPU applies the unchanged f32 SiLU `silu(g)*u` (`W:crates/peacemaker-isa/src/kernels/iu4_v2b.rs:753-758`). No Flash BF16 path |
| down (64) | 8192×5120×17408, 18.939 | `:7829-7836,10049-10056`; `gemm.rs:35772-35781` | n=1024 | NPU delta f32. The GPU does `RN(old + delta)` **once** into `x_batch` |

(All numbers are the measured Halo trace of `D27` §1 except the windows, which are the ESTIMATE optimum of §4.)

**Epilogue details.**
- **o_proj.** The existing GPU route already SETs a delta and records `residual_fold_pending` (`W:gemm.rs:20860-20985`, kill switch `HIPFIRE_V2B_ADDEPI=0`). The next RMSNorm lands `RN(x+delta)` (`W:gemv.rs:3448-3454`); any other reader flushes (`W:gemm.rs:21342-21367`). Eligibility is `W:prefill.rs:223-238`. An NPU window is just a column slice of the same delta buffer; the same flush covers it.
- **down.** The existing ADD is `gemm_mq4g256v2_mmq_add_prequant_iu4` (`W:gemm.rs:21206`). The NPU part must not ADD in place. It writes its f32 delta; the GPU adds `RN(old+delta)` once, as one extra pass (model: 12 B per element).
- **Z.** `HIPFIRE_V2B_ZBA_SCATTER=0` fallback, or `gemm_zba_v2b_scatter`, is hipcc-only (no PM entry, `W:gemm.rs:743-757`). The GPU part (rows `[n,6400)`) needs a **current-fold column-windowed entry**, preserving the β/α scatter and the 160-row discard; it must not use the killed GPU IEF15 fold.
- **Current-contract partials fallback** (NPU int32 `C_e`, GPU folds) is a separate **write bound ~3.3 TOPS** (**MEASURED**, `D27` §4, w1 `logs/dense27b/w1-be0be55.log`). It is not part of IEF15, and there is **no claim of current-f32 parity for IEF**.

**Current GPU range API gap.** Nothing in the tree accepts a feature window with a full output stride:
- V2B Y is `[N][M]` with stride exactly M (`W:iu4_v2b.rs:30-31,363-371,684-686`). There is no `ldy`, column offset or column count in any launcher.
- A weight row offset works by pointer (`A + r0*row_bytes`, r0 a multiple of 256), but Y is then compact `[N][M']`. ADD reads old Y at the same compact stride.
- A token window needs an `ldx` argument, because the Xq epoch stride is `N*72` (`W:iu4_v2b.rs:362,554`).
- Eligibility is shape-gated (`W:gemm.rs:20522-20552`): `m%128, n%128, K%256`; V2B needs `m%256, n%256` and `(m/256)*(n/256) >= cu_count`.
- Gate/up can only cut equal windows of G and U (`W:gemm.rs:33901-34000`).
- Today's substitute is a "compact subcall" (offset A, compact output, extra pass). The design requires a **column-windowed current-fold GPU entry** with `(feature_window, ldy)` and, for gate_up, equal G/U windows, numerically identical to the incumbent kernel on its columns. No such entry exists and none is claimed; its speed versus the full-width incumbent is unmeasured (the payoff model assumes `gpu_fold_ratio` 1.0).
- The current A4/V2B fast path is eager-only (`W:gemm.rs:20523-20535`), not graph-capturable. Graph/replay stays unchanged for the current route; **the windowed entry must explicitly admit capture/replay**, keyed by the split map.

**V8 K and N slicing (existing NPU kernel, pre-IEF).**
- V8 `MAX_KC=160` chunks of 64 (K ≤ 10240; `crates/peacemaker-npu/src/kernels/gemm_array.rs:313-314,614-615`). Down K=17408 is therefore **two K segments of 8704 (136 chunks each)** that **preserve the same int64 state**. Per-segment f32 rounding is forbidden (`FC` §3: sum integer states before rounding). The state spills to DDR between the two commands (**ESTIMATE** 16 B per output in H, §4).
- With 8192 tokens, V8 waves = 16·ceil(n/512) ≤ 256 (`gemm_array.rs:313,616-618`, `design_v8` `:1865`) ⇒ **n ≤ 8192 features per command**; larger windows slice N (`j = ceil(n/8192)`).
- The new TM64/TN32 IEF array has **no measured rate or wave geometry**. If its wave is inferred as 256×256, the 256-wave cap gives max 2048 columns per command. That is a sensitivity (§4), not an existing API.
- New-core memory (`FC` §8 M0): TM64/TN32 = 36 KiB live + 25.5 KiB S/D at E136 + 192 B a/b = **61.6875 KiB** (rounded 61.7 KiB). The frozen metadata offsets are `0x9000..0xf600`, a/b `0xf600..0xf6c0`; layout/locks/routes remain a proof gate, not measured capacity credit.

## 2. Buffers: ownership, layout, lifetime, memory type

Transport (**MEASURED**): anonymous host pages, pinned by amdxdna as a **userptr BO** (`crates/railgun/src/npu/mod.rs:385-411`: `pin_user_pages_fast(FOLL_WRITE|FOLL_LONGTERM)`) and `hipHostRegister`ed for the GPU (`crates/npu-tools/src/bin/npu-coop.rs:12-18,76-88`). The three-way alias check passes in all four directions: cpu→gpu, gpu→cpu, npu-map==cpu, npu-map→cpu (`LOG:4,24,43`).

**HIP VMM dma-buf export is NOT a live alias (MEASURED, negative).** After import, GPU writes are not seen through the dma-buf, dma-buf writes are not seen by the GPU, and re-export returns the old bytes (`logs/coop/coop-w2-45aef0b.log:4-8`, `logs/coop/coop-w2-c6193a6.log:4-9`; `report.md:732-759`). Never use an exported HIP VMM allocation for A, D, C or the ring. An immutable pre-written B export is not a counterexample. The fixture was 4 KiB arenas for A and C in the empty ring (`LOG:5`), 15.7 MB / 3.1 MB in the first GEMM fixture (`LOG:25`). It did **not** cover dense IEF sidecars, production arena sizes, concurrent GPU load or graph/replay.

| Buffer | Writer → reader | Layout | Lifetime / memory type | Ownership rule |
|---|---|---|---|---|
| MQ4 weights (QT44) and A4 `xq` (`block_i4_128`, 72 B) | load / GPU producer → GPU, packers | `[K/128][token]` token-fastest (`W:scratch.rs:669-686`) | model / per-layer; device memory | **Immutable** under IEF15; same-device ordering. Stale-handle check `W:dispatch.rs:3670-3678` |
| Weight sidecars `S` (u16 `[local_feature][K/128]`), `a` (i16 `[local_feature]`) | CPU load → NPU B packer / NPU | `FC` §3 ABI | model lifetime; written once before use | Only NPU-owned weight windows. Immutable; no GPU current-fold consumer |
| Activation sidecars `D` (u16 `[K/128][token]`), `b` (i16 `[token]`) | GPU metadata pass → NPU packer / NPU | `FC` §3 | per layer/projection | Generated **after every A4 write** for admitted NPU work; GPU current-fold entry reads original Xq, not D/b |
| Packed NPU B (int8-expanded window of weights) | CPU → NPU | NPU-packed | model lifetime; NPU BO | ESTIMATE **4,982,833,152 B** across layers at the fast cuts (Σ n·K·calls, int8). Immutable |
| Resident S/a for NPU windows | CPU → NPU | `2·n·(K/128+1)` B/call | model lifetime | ESTIMATE **79,462,400 B** total |
| Packed A (+ D/b) arena | GPU pack kernel → NPU shim DMA | int8 expanded, V8 A stream | per slot; **userptr + `hipHostRegister` pages** | GPU compute done and writes visible **before** seq publish; no overwrite until NPU done |
| C arena (f32, pre-SiLU / pre-residual) | NPU DMA → GPU | token-major f32, full stride | per slot; same pinned pages | NPU C SYNC then DONE; GPU acquire/invalidate before load; NPU/GPU SET regions disjoint |
| Ring header, slot lines (`seq` last), done lines | CPU once; GPU publish; NPU done | `docs/npu/railgun-npu.md:62-95` | session; same page class | GPU sole publisher, NPU sole done writer |
| PDI/TXN/CMD BOs, tape keys | CPU → firmware | `docs/npu/railgun-npu.md:134-149` | program lifetime | patch only declared words |

**Layout consequences (ESTIMATE arithmetic, M=8192).**
- Full-strided C needs **4·M·N bytes physical per call**, not 4·M·n. The NPU's window sits inside the full output rows of the pinned buffer: qkv 320 MiB, Z 195 MiB (logical 6240), q 384 MiB, o 160 MiB, gate+up 1088 MiB (544 MiB each), down 160 MiB. This is what makes "other" epilogue passes zero in the model, but it also moves the **GPU's own output writes and the consumers' reads onto registered host pages**. That GPU store/load bandwidth is **unmeasured**; the model's zero extra pass for qkv/z/q/o is optimistic.
- Compact alternative (NPU writes `[token][n]`): gate_up 224 MiB max (8192·7168·4); down A 136 MiB (2 × 8192·8704 int8) + C 32 MiB + i64 state 64 MiB. It needs the GPU pass the model charges as `epi`.
- Two-bank ownership (double buffering across layers) doubles these **transients**, not the resident B.
- Packed A arena bytes before any packing replication: 8192·K int8 (e.g. 40 MiB at K=5120).

Memory-kind limits: A/C must be **userptr + hipHostRegister**; the CPU re-arms only. GPU poll loads are uncached `glc dlc`, publish stores are `glc slc dlc` + `vscnt 0` (`LOG:16-17`, `report.md:711-720`). Whether the GPU may write the full-stride output through registered pages at full speed is a transport gate in §5.

## 3. Synchronization timeline per layer

```text
GPU: norm/rotate (A4 codes) → D/b pass → pack NPU-window A,D/b → release → publish seq(k)
GPU:                           run GPU window (current-fold entry, own region) ─┐
NPU:                           POLL seq → IEF15 window GEMM → C SYNC → DONE(k)
GPU:                           wait done(k) (equality) → acquire → epilogue (SiLU / delta add) ┘
```

Each NPU command is **one kick + one wait transaction (2 directions)**. Independent GPU work must be issued before the GPU waits, because the wait stalls its queue.

| Count (per step) | Commands | Directions |
|---|---:|---:|
| All seven families offloaded (theoretical) | 336 pairs | 672 |
| Same, down K-split in two segments | 400 split jobs | 800 |
| Fast cuts (kv stays GPU; down split) | 368 = 240 other + 128 down segments | 736 |
| Slow cuts (down and kv stay GPU) | 240 | 480 |

Per layer at the fast cuts: GDN layer = qkv + zba + o + gate_up + 2 × down = 6 commands (48 layers); FA layer = q + o + gate_up + 2 × down = 5 (16 layers). 288 + 80 = 368. Nslicing adds jobs beyond 400 if a window exceeds 8192 features.

**MEASURED GPU-driven ring timing** (`LOG`, GPU realtime clock 99.81 MHz; S8/R4, p50):

| Item | Value | Source |
|---|---:|---|
| publish → NPU done → GPU observe, empty ring | p50 3.13 µs (min 2.73) | `LOG:14` |
| done → GPU observe upper bound | 0.56 µs | `LOG:15` |
| uncached poll load | 0.24 µs | `LOG:16` |
| publish store (seq) | 0.20 µs | `LOG:17` |
| publish-to-publish interval | 5.77 µs | `LOG:18` |
| CPU-producer ring (historical) | 4.5 µs | `report.md:622-627` |
| GEMM 512×1280×2560: publish→done / whole copy+GEMM+read | 195.93 µs / 276.46 µs = 12.11 TOPS | `LOG:32,37` |
| GEMM 512×2560×640: publish→done / whole | 109.57 µs / 175.43 µs = 9.58 TOPS | `LOG:51,56` |
| exactness | GPU-read C == eager 32/32 (both shapes) | `LOG:38,57` |

The 3.13 µs is a round trip, not 3 µs each way. The model charges **6 µs per kick/wait transaction** (ESTIMATE, the 5.77 µs interval rounded up; `j·0.006 ms`). The step totals 400 × 6 µs = **2.4 ms**, against the old 20 µs relay figure of 8 ms (ESTIMATE). A first-run start of up to 1.6 ms is the `max` row in these logs and is not in the model. The GEMM rows are int8 V9 fixtures, so they say nothing about IEF rate. The kernel poll proof is not a PM4 `WAIT_REG_MEM` hardware proof, and its poll-interval field of 4 is **not** 4 µs (`docs/coop-flashnext.md:97` records it as unmeasured).

**Release / acquire / slot reuse (`docs/npu/railgun-npu.md:117-132`).**
- Slot free: wait `done[s] == seq − nslots` (equality, never `>=`; wrap-safe).
- Release: the packing kernel's writes are complete and visible before the seq store; seq is written last.
- Acquire: wait `done == seq` with uncached loads, then read C. The NPU writes the done line only after every C channel's SYNC.
- CPU re-arm: `patch_seq` of declared TXN words on the retained command BO, rounds queued ahead. A persistent command needs `S mod nslots == 0` and is bounded by the 60 s job timeout.

**Program/tape identity and limits.**
- The IEF15 program has a **distinct PDI and TXN**, hence distinct `ProgramKey`/`ConfigKey` (`docs/npu/railgun-npu.md:136-149`). **No compatibility promise** with existing V8/V9 ring programs, which `FC` §3 also forbids.
- The **split map is bound into both identities**: the NPU tape `ProgramKey` already covers the per-family PDI/TXN (window fixed in the build), and the GPU side's capture/replay key must carry the same map id; a capture recorded under one map never replays under another.
- P1 limits: one shape per PDI per persistent command, **fixed arena offsets per slot** (the GPU writes operands into the slot; it does not pass addresses), no dynamic per-slot pointers (P2, not built).
- A command that follows a different `ConfigKey` takes the full TXN (≈150 µs firmware, `docs/npu/railgun-npu.md:38-48`). Consecutive families have different K and N, so lean-TXN reuse is partial. The model prices a **50 µs per command prologue** (lean ≈50 µs/GEMM left, `report.md:780`; ESTIMATE for IEF). The conservative 215 µs bound is a §4 sensitivity (215 µs/submit measured, `report.md:766`).
- Current A4/V2B fast-path selection is eager-only; incumbent graph routes are not globally declared unsupported. The new current-fold windowed co-op path has no graph/replay proof yet.
- **No heterogeneous step executor exists.** P1 runs one shape/PDI per persistent command (`docs/npu/railgun-npu.md:92-96`), and `crates/railgun/src/npu/tape.rs:1-3` is a pure hash/cache helper (keys, program cache, replay check), not a scheduler. Proposal (not built): pre-queue one retained per-GEMM command per dependency-ordered kick, all under a compatible common IEF PDI, with GPU release/done at each boundary. S=1 / nslots=1 permits ordered commands; the `S mod nslots == 0` multi-round rule applies only when S>1 is used.
- **Deadlock rule.** Never queue an S>1 command of one family whose future kick depends on another family's command queued behind it (the NPU would poll a seq the GPU cannot publish yet). Queue only in actual dependency order.
- The full-step PDI transition and submit cost across families is **unmeasured** and is **excluded from the model credit** beyond the 50 µs / 215 µs prologue bounds. The measured ring fixtures (`LOG`) are single-shape and do not implement a mixed-K ordered schedule. Gate a measurable ordered mixed-K schedule (§5 item 6) instead of assuming it.

## 4. Baseline, model and results

**MEASURED baseline** (`D27` §1, M=8192, 2nd prefill): step 6810.690 ms = **1202.8 tok/s**. GEMM sum 5287.408 ms (77.6%), 398.9595 TOP useful with Z logical 6240 (the old note's 399.6 uses padded 6400). Remainder 1523.282 ms.

**NPU rate.** V8 int8 busy 26.94–27.53 TOPS (**MEASURED**, M=4096, `D27` §4). Proxy P0 = 26.9. IEF cost 1.15–1.33× the GEMM (**ESTIMATE**, `D27` §9; instruction census `FC` §6): P = 26.9/1.15 = **23.391304** (fast) and 26.9/1.33 = **20.225564** (slow). β = 0.853 (22.97/26.94, `D27` §6). α = 1.113 at NPU gap 0 (**drift-flagged**, `D27` §6, +4.13% GPU-alone drift); alpha 1.101/1.059/1 are sensitivities. γ = **current-fold strided GPU cost ratio**, assumed 1 or 1.03; it is not the killed 1.57× GPU IEF penalty.

**Excluded rates.** The lean-TXN expert rates (30.18 / 32.88 TOPS, `report.md:780-781`) were measured at M=4096 expert shapes, not dense K. The G80 28.75 TOPS has N=1280 only. Neither is imported. The 50 µs prologue replaces the old 215 µs only as a conservative extra bound (V8P is already an elapsed rate, so adding a prologue is deliberate pessimism, not a measured pure-compute figure).

**Mixed-contract payoff (ESTIMATE, FoldContract model).** `tools/npu/fold-model/payoff.json` (md5 f844e93bee923c9b32d7a5d8f19f4ebd), contract mode `mixed-current-gpu-ief15-npu` (GPU fold ratio 1.0: incumbent GPU fold unchanged), Halo baseline 6810.69 ms:

| assumed NPU raw TOPS | NPU traffic 2.44 B/kop (V8): gain, NPU ops share | 3.05 B/kop (25% retile): gain, share |
|---:|---|---|
| 19.0 | +3.36%, 17.6% | +2.64%, 17.6% |
| 21.0 | +5.26%, 20.2% | +4.05%, 20.2% |
| 23.0 | +5.38%, 21.1% | +4.01%, 20.2% |
| 26.9 | +6.81%, 24.9% | +4.01%, 20.2% |
| 30.2 | +6.81%, 24.9% | +4.01%, 20.2% |

The same file's `full-ief15-transferred-g0-penalty` Halo rows at 19–23 TOPS are **−18.50 to −21.97%** (`FC` §7), not a promoted route; their 1.57× GPU penalty comes from measured G0. The mixed model retains a 55 GB/s port cap, 0.215 ms submit, 0.020 ms GPU ring and drift-flagged β/α; the measured GPU roundtrip (3.13 µs, `LOG`) is below its 20 µs. The tables below are this document's **independent compact-A / literal-V8-packing cross-check**, with γ=1 assuming current-fold GPU speed. They are conditional mixed estimates, not measurements or proof that the strided entry or A reuse exists.


**Model (all ESTIMATE).** f = n/N; O = 2·M·N·K.
- H = 1000·O/(P·β·1e12) [+ down: 1000·16·M·N/55e9, int64 state spill]
- j = ceil(n/8192) [down ×2]
- meta = j·(2n(K/128+1) + 2M(K/128+1))/55e9·1000
- pack = (M·K·72/128 + M·K)/200e9·1000 (compact-A bound)
- epi = {gate_up 6·M·n, down 12·M·n, other 0}/200e9·1000
- **t = max(α·γ·(1−f)·g, f·H + j·0.050 + meta) + pack + epi + j·0.006**, and t = γ·g for n=0 (no offload overhead). n is minimised over {0} ∪ 512-multiples (gate_up: 1024 combined, matched halves).
- step = remainder + Σ calls·t.
- GPU-side stream generation, cache flush, real SiLU compute and strided-output DMA are unmeasured beyond the bandwidth floors above, so t is an upper credit. The new TM64/TN32 retile is a separate unmeasured control.

**Per-family contribution and isolated step** (isolated = base − that family's saved ms alone; 8192 tok/step). Gain columns distinguish **tok/s** from **time**:

| Family | Fast: n (f) | t ms/call | saved ms/step | isolated step ms | tok/s | Δtok/s | Δtime |
|---|---|---:|---:|---:|---:|---:|---:|
| qkv | 2048 (20%) | 10.540335 | 44.287910 | 6766.402 | 1210.688 | +0.655% | −0.650% |
| zba | 1536 (24.615%) | 6.855880 | 35.909746 | 6774.780 | 1209.191 | +0.530% | −0.527% |
| q | 2560 (20.833%) | 12.424477 | 20.760364 | 6789.930 | 1206.493 | +0.306% | −0.305% |
| kv | 0 | 1.386000 | 0 | 6810.690 | 1202.815 | 0 | 0 |
| o | 1024 (20%) | 6.370238 | 21.488742 | 6789.201 | 1206.622 | +0.317% | −0.316% |
| gate_up | 7168 (20.588%) | 36.165167 | 152.437312 | 6658.253 | 1230.353 | +2.289% | −2.238% |
| down | 1024 (20%) | 18.899022 | 2.558618 | 6808.131 | 1203.267 | +0.038% | −0.038% |

| Family | Slow: n | t ms/call | saved ms/step | isolated step ms | tok/s | Δtok/s | Δtime |
|---|---|---:|---:|---:|---:|---:|---:|
| qkv | 2048 | 10.540335 | 44.287910 | 6766.402 | 1210.688 | +0.655% | −0.650% |
| zba | 1024 | 7.408091 | 9.403649 | 6801.286 | 1204.478 | +0.138% | −0.138% |
| q | 2560 | 12.847155 | 13.997525 | 6796.692 | 1205.292 | +0.206% | −0.206% |
| o | 1024 | 6.440411 | 16.997725 | 6793.692 | 1205.824 | +0.250% | −0.250% |
| gate_up | 7168 | 37.021032 | 97.661927 | 6713.028 | 1220.314 | +1.455% | −1.434% |
| kv, down | 0 | γ·g | 0 | 6810.690 | 1202.815 | 0 | 0 |

Savings are additive in the full step (remainder is common).

**Full-step totals** (ESTIMATE; α=1.113, γ=1, 50 µs prologue):

| Case | step ms | tok/s | saved ms | time reduction | **tok/s gain** | commands |
|---|---:|---:|---:|---:|---:|---:|
| Baseline (MEASURED) | 6810.690 | 1202.8 | – | – | – | – |
| Fast, P=23.391304 | 6533.247 | 1253.894 | 277.443 | 4.0736% | **+4.2466%** | 368 (736 dirs) |
| Slow, P=20.225564 | 6628.341 | 1235.905 | 182.349 | 2.6774% | **+2.7510%** | 240 (480 dirs) |

**Sensitivities** (slow / fast step ms; tok/s gain %):

| Case | Slow step | Fast step | tok/s gain slow / fast |
|---|---:|---:|---|
| α=1.0 (no contention) | 6301.277 | 6165.247 | +8.08% / +10.47% |
| α=1.059 | 6496.768 | 6375.760 | +4.83% / +6.82% |
| α=1.101 | 6604.895 | 6498.250 | +3.12% / +4.81% |
| α=1.113 (primary) | 6628.341 | 6533.247 | +2.75% / +4.25% |
| γ=1.03 on all GEMMs | 6708.532 | 6619.903 | +1.52% / +2.88% |
| prologue 215 µs | 6651.418 | 6543.726 | +2.39% / +4.08% |
| replicated V8 A (int8 A repeated per 512-feature N wave: `gemm_array.rs:675-699`) | 6795.134 | 6768.103 | +0.229% / +0.629% |
| inferred 256-wave, 256×256 array, 2048 max columns, replicated A | 6810.690 | 6810.690 | **no positive credit** (n=0 is best for every family) |

The replicated-A rows replace `pack` with `(M·K·72/128 + M·K·ceil(n/512))/200e9`. The 256×256 row uses `ceil(n/256)` replication and `j = ceil(n/2048)`. Current V8 packing repeats A for every N wave (`gemm_array.rs:675-699`); the compact-A model assumes the new IEF array can reuse one A stream, which is unmeasured. The +4.2% fast result therefore **depends on a packing design that does not exist yet**.

**Halo +10% / +15% answer.** No promise. The only row above +10% tok/s is the no-contention optimistic case (α=1, fast cuts): +10.47% tok/s, which is **9.48% time**. That α is contradicted by the drift-flagged measurement (α 1.059–1.113). No row reaches +15% in tok/s or in time. Throughput gain and time reduction are not interchangeable: +10.47% tok/s ≙ −9.48% time. The plan: Halo ≥10% is killed as a promised result (`FC` §1).

**gfx1152 / Krackan (conditional ESTIMATE, not a port proof).** Public spec: AMD's [Ryzen AI 7 350](https://www.amd.com/en/products/processors/laptop/ryzen/ai-300-series/amd-ryzen-ai-7-350.html) lists Radeon 860M **8 CUs, up to 3.0 GHz, two memory channels, LPDDR5x-8000 / DDR5-5600, 50 NPU TOPS and 66 overall TOPS**. Two 64-bit channels imply theoretical **128 / 89.6 GB/s**, not sustained bandwidth. [RDNA3 WMMA](https://gpuopen.com/learn/wmma_on_rdna3/) gives 1024 IU4 ops/CU/clock: peak 24.576 TOPS, assumed 70% efficiency ≈17.2 useful TOPS. The assumed .225 throughput ratio versus Halo scales GPU g by s=4.444444 (`FC` §7); γ remains the current-strided cost ratio. NPU rate/β/α transfer from Halo, and the remainder scales r=2 (bandwidth) or 4.444 (compute).

| Remainder r | Base ms (tok/s) | Slow / fast step ms | Slow / fast tok/s | tok/s gain | time gain |
|---|---:|---|---|---|---|
| 2 | 26546.155 (308.6) | 16297.805 / 15565.421 | 502.644 / 526.295 | 62.882% / 70.546% | 38.606% / 41.365% |
| 4.444 | 30269.733 (270.6) | 20021.384 / 19288.999 | 409.163 / 424.698 | 51.187% / 56.927% | 33.857% / 36.276% |

`FC` §7's **mixed** model reaches +48.91–72.06% gfx1152 tok/s under its 256-row cuts, .215 ms submit and 55 GB/s cap. Its separate **full-IEF15** sensitivity transfers the killed Halo 1.57× GPU cost and yields +29.29–48.58%; that is not a gfx1152 measurement or a revived route. gfx1152 is **excluded from the relevant IU4 prefill/A4/V2B admissions** (`W:feature_flags.rs:1023-1026,1097-1100`, `W:gemm.rs:20535-20551,31544-31567`, `W:iu4_v2b.rs:78-82`); explicit admission and a bundle are required. Retile traffic, shared power and memory capacity (27B plus ~5 GB NPU B) can erase the gain. Vendor-rated TOPS and Halo data do not promote gfx1152.

## 5. Gates and kill points, in order

1. **G0 (GPU IEF15 entry): KILLED** — measured 1.57× the incumbent (`exp/ief15-g0` @ 7559ee957). Replacement gate **G0m: column-windowed current-fold GPU entry.** Build it with `(feature_window, ldy)` (§1); prove byte identity with the incumbent kernel on the GPU columns for every family; measure its real-shape time at the planned split (KILL if the GPU side at `(1−f)` columns costs more than the payoff model's `gpu_fold_ratio` budget). No NPU kernel work is credited before it passes.
2. **Quality / parity of the mixed contract.** Default-OFF incumbent execution retains its text/bundle/A4/dispatch and seeded replay identity. GPU-owned columns under the new current-fold entry must byte-match incumbent arithmetic; NPU columns must match the CPU IEF15 reference (`FC` §3), with one rounding after all K segments. Bind the exact map, segments and contract version into every capture/cache/run identity (`FC` §4). Unsupported or unavailable NPU work fails closed; any explicitly provided same-contract recomputation must preserve those columns' bytes. Synthetic-activation IEF evidence (`FC` §5, NRMSE ≈3.15e-5) is **not real-model KLD**; quality is checked end-to-end for each fixed map. The new windowed entry must explicitly pass graph/replay without silently falling through another numeric route.
3. **Packing and metadata cost, measured.** D/b pass, A pack (compact vs replicated A), epilogue passes (SiLU, `RN(old+delta)`), and full-stride GPU writes into pinned pages. The result (replicated V8 A → +0.2–0.6%) decides whether continuing is worthwhile.
4. **Transport at production size.** Userptr arenas with A, D/b and f32 C at the sizes in §2, repeated rounds with stale-poison negatives, GL2 release-before-seq / acquire-before-C, concurrent GPU load, GPU stores to registered pages. A VMM export or CPU-producer ring is not this gate.
5. **New IEF NPU program (N0), after mixed current-fold admission.** Legal 61.7 KiB map, simulator exactness (extremes, ties, subnormal, E=5/20/40/48/136, K-split state), declared new-design ≤90 s silicon window. **Proposed rate checkpoint: attained dense rate ≥19 TOPS** (lowest mixed row, +2.64–3.36%); below it, recompute payoff before continuing. NPU bytes must equal the CPU IEF15 reference for its columns.
6. **State and slicing.** Down K-split int64 state across two commands, N slicing past 8192 (or the new wave cap), distinct IEF PDI/tape keys, `S mod nslots == 0`, config-key transitions between families, full-step exact rejoin with Z/FAq/gate_up/o_proj/down epilogues, graph/replay of the new entry.
7. **Full step and contention.** Interleaved alone/concurrent pairs with matched binary/prompt md5, thermal protocol, re-measured α/derate (§6 of `D27` is drift-flagged). Go only for measured net step savings; the +4.2% fast estimate is a conditional reference to compare against, not a result or a hard ceiling.

Open: new-core IEF rate and wave geometry, A-stream reuse, GPU store/load speed on registered pages, real prologue per family transition, and the real α under IEF traffic.
