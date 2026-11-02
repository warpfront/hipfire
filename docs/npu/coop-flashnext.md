# Flash-Next GPU + NPU co-op integration (historical design)

> **Historical research note: superseded by the 0.4.2 serving route.** This file records a deprioritized (2026-10-03, project decision) design study for sharing Flash-Next MoE experts between the GPU and the NPU from the 2026-10 NPU research cycle. It is not current guidance, and any figure in it is a dated fixture-bound measurement or an explicitly labelled historical estimate, not a product claim. The shipped 0.4.2 NPU route is the Flash-Next gfx1151 trunk input projection described in [npu-trunk.md](npu-trunk.md): `HIPFIRE_QWEN4_NPU_TRUNK=off|auto|on|emulate` (default `auto`, effectively `off` on every architecture except gfx1151), calibration schema 4, BF16-direct only. There is **no I32 serving contract** (no int32 partial, combine or fold exists in the product route). The out-projection program exists but is **not wired** into serving; `qsa.o` and the GDN o-projection are **not shipped**. Environment-variable names and flag syntax below are historical proposals or experiment switches, not supported configuration.
>
> This co-op expert design was never built as a serving route; the IEF15 numerics it relies on are not an admitted contract. The opt-in flag named below, `HIPFIRE_IU4_FOLD_CONTRACT`, is absent from the source tree: it was a proposed name that was never implemented and is retired, not a supported opt-in. See also [coop-dense27b.md](coop-dense27b.md).

Design only, 2026-10-03. Canonical model: `qwen3.8-flash-next-gptq3.mq4`, with symmetric MQ4G256V2/MQ4G128V2 routed experts; not the older asymmetric Flash-Next artifact. Hipfire references below are to the 0.4.1 hipfire beta checkout, so line numbers need not match 0.4.2; NPU references are relative to this repository. `FN` means the Flash-Next Halo measurement ledger `fn-halo-2500.md` (not included in this repository). **MEASURED** means recorded silicon/profiling observations, not a co-op result. **ESTIMATE** covers all derived arithmetic, transferred baselines, proposed costs and extrapolations. Public specifications are identified separately as **SPEC**, not measurements.

`FC` means [fold-contract.md](fold-contract.md): the frozen IEF15 contract (its §2) and the G0 GPU-speed slice (its §3). **REPORTED-MEASURED** means a Halo measurement reported externally and not reproduced here; the cited coop-w2 lines under `logs/coop/` were checked against the recorded logs. IEF15 is a dark opt-in experiment accepted for study (`FC:3`); it is not today's numeric contract and not an admission.

The integration seam is viable only as a design, not a shipped route. Its live transport now has a **MEASURED fixture proof** (§3: anonymous host pages via NPU userptr BO + `hipHostRegister`), but exported HIP VMM memory is measured snapshot-only and withdrawn. The dense-IEF kernel, placement and quality gates remain open. Under the frozen IEF15 contract, granting a free numeric fold and zero transport cost, the fixture-only COOP-W1 projection (original model, IEF15 metadata excluded) is **ESTIMATE 4.47 ms / 0.10%** step-time reduction for V9 and **16.51 ms / 0.37%** for G80 gate_up + V9 down at the M4 derate. Adding the optimistic read-once IEF15 sidecar-metadata surcharge (§6) gives the **metadata-adjusted ESTIMATE 4.04 ms / 0.089% (V9) and 16.06 ms / 0.355% (G80 + V9 down)**; wave-padded D/b gives 3.94 ms / 16.00 ms. These are not timings of the TM64/TN32 IEF15 kernel and claim no free fold. Neither supports a +10%/+15% Halo claim. Exact epoch-partial output is a current-contract parity fallback, not a speed path. Flash-Next is deprioritized; dense-27B integration lives in `coop-dense27b.md`.

## 1. Where the NPU plugs in

**Canonical admission.** The registry selects `qwen3.8-flash-next-gptq3.mq4` (`registry/models.json:1241-1243`), but dispatch does not match its basename. The loader calls typed HFQM admission (`crates/hipfire-loader/src/admission.rs:255`): quant type **44** is MQ4G256V2/group-size 256; quant type **53** is MQ4G128V2/group-size 128 (`crates/hipfire-arch-qwen4/src/artifact.rs:28-29,222-231,269-278`). The expert pair is pinned in `crates/hipfire-arch-qwen4/src/weights.rs:36-41`; down K640 cannot use G256. At load, every layer checks both expert pointer tables and headers before setting `symmetric` (`crates/hipfire-arch-qwen4/src/gpu_forward.rs:919-942,2124-2140`). Symmetry means finite scales/zeros and `zp == -8*sc`, not just a matching dtype (`kernels/src/qwen4_moe_sym_check.gfx1201.hip:7-10,34-39`). Only this verified state declares `Qt44Qt53GroupedSymmetric` (`crates/hipfire-dispatch/src/families/moe.rs:108-115`; `crates/hipfire-arch-qwen4/src/gpu_forward.rs:379-406`). The header verification is a **load-time host/device sync**, distinct from the GPU-only per-layer counts.

The IU4 prefill arm requires this capability, grouped path2, BF16 round-trip recipe, F32 normalized input and eligible batch/architecture (`crates/hipfire-dispatch/src/pipeline/qt44_qt53_prefill.rs:579-588`). It is default-on for gfx1151, admitted for gfx1151/gfx1201 with the two-candidate A4 producer; batch floor is source-defined 512 (`crates/rdna-compute/src/gemm.rs:44739,44899-44925`). Decode does not take this arm (`crates/hipfire-arch-qwen4/src/gpu_forward.rs:1408-1410`). **gfx1152 is not currently in that architecture whitelist**; the later portability section does not imply automatic admission.

| Canonical expert tensor | Source-defined payload/scale layout | IU4 epochs and kernel |
|---|---|---|
| GU MQ4G256V2 / qt44, N1280 K2560 | 136 bytes per 256 weights: two independent `[f16 sc,f16 zp]` K128 header dwords, then 128 nibble bytes. Epoch h header at `(h/2)*136+4*(h%2)`. | **20 K128 epochs**, `gemm_qwen4_moe_gate_up_silu_iu4_sym`; G256 is storage/rotation width, not a K256 fold epoch. |
| Down MQ4G128V2 / qt53, N2560 K640 | 68 bytes per 128 weights: one `[f16 sc,f16 zp]` dword, then 64 nibble bytes. Epoch h header at `h*68`. | **5 K128 epochs**, `gemm_qwen4_moe_down_iu4_sym`; two-epoch loop plus one odd tail. |

Layouts: `crates/rdna-compute/src/dispatch.rs:80-93`; epoch addressing `crates/peacemaker-isa/src/kernels/qwen4_moe_sym.rs:239-240,353-358,499-510`; launch guards `crates/rdna-compute/src/gemm.rs:45382-45441`. Therefore the exact-partial byte arithmetic in §5 applies unchanged to this canonical G256/G128 pair.

The per-layer IU4 route is:

| Stage | Inputs → outputs | Source |
|---|---|---|
| Router projection, softmax/top-10 | normalized input → router logits → `topk_indices`, `topk_weights` | `crates/hipfire-dispatch/src/pipeline/qt44_qt53_prefill.rs:119-153`; `crates/hipfire-dispatch/src/pipeline/sealed_moe.rs:2086-2095` |
| Stable expert grouping | top-10 flat slots → `sorted_slot_index`, `inverse_perm`, `expert_tile_ids`, padded expert counts/offsets | `crates/hipfire-dispatch/src/pipeline/qt44_qt53_prefill.rs:591-610`; `kernels/src/qwen4_moe_scatter_stable_top10.hip:29-108` |
| `qwen4_moe_rotate256_i4` → `gemm_qwen4_moe_gate_up_silu_iu4_sym` | token input → gate `xq` → `y_gate_up_grouped` | `crates/hipfire-dispatch/src/pipeline/qt44_qt53_prefill.rs:663-678` |
| `qwen4_moe_rotate128_i4` → `gemm_qwen4_moe_down_iu4_sym` | grouped activated BF16 → down `xq` → `y_down_grouped` | `crates/hipfire-dispatch/src/pipeline/qt44_qt53_prefill.rs:892-915` |
| `moe_down_combine_grouped_top10_bf16in` | grouped down, inverse permutation, original IDs/weights → residual | `crates/hipfire-dispatch/src/pipeline/qt44_qt53_prefill.rs:1015-1030` |

A flat slot is `token*10 + topk_rank`. Grouping is stable by `(expert, flat_slot)`, with each expert padded to a multiple of 16. The GPU prefix kernel produces **true M** at `chunk_prefix[e*(chunks+1)+chunks]`, but **`expert_token_counts[e]` is pad16 M**, not true M; offsets scan those padded counts (`kernels/src/qwen4_moe_scatter_stable_top10.hip:52-77`). The final scatter writes `sorted[p]=slot`, `inverse[slot]=p`, padding `sorted[p]=-1`, and one expert ID per 16-row tile (`:80-108`). Counts become known when the prefix kernel completes, before scatter/layout production. They are **GPU-only in this route**: no count download or host synchronization occurs in the staged pipeline. `grouped_rows` is a host-known allocation/grid bound, not an observed routing count (`crates/hipfire-arch-qwen4/src/program.rs:755-756`; `crates/hipfire-dispatch/src/pipeline/moe_program.rs:1605-1608`). Do not add a per-layer readback to choose the NPU group.

`block_i4_128` is `{f32 d, i32 s, u8 qs[64]}`: **source-defined 72 bytes** holding one K128 epoch (`kernels/src/block_i4_128_quant.hip:42-47`). Gate `xq` is epoch-major `[20][batch]`, addressed by `slot/10`; down `xq` is `[5][batch*10]`, addressed by the **original flat slot**, not grouped row (`crates/rdna-compute/src/gemm.rs:45272-45309,45382-45441`; `kernels/src/qwen4_moe_rotate128_i4.hip:30-34,67`). `y_gate_up_grouped` is row-major BF16 `[p][640]` after SwiGLU; `y_down_grouped` is BF16 `[p][2560]`. The GPU GEMMs read tile IDs and sorted slots, not the counts/offsets (`crates/peacemaker-isa/src/kernels/qwen4_moe_sym.rs:9-33`). The gate/up kernel already includes SiLU; the separate activation/unscatter steps are skipped for this route (`crates/hipfire-dispatch/src/pipeline/qt44_qt53_prefill.rs:749-772,857-859`). Moving gate/up to the NPU therefore requires a GPU epilogue, not merely calling the existing activation step.

`hipfire-xdna` already has raw amdxdna allocation/import, PDI+TXN loading, START_CU submission and timeline-syncobj waiting. It is **single-flight** and uses CPU cache flushes; imported BOs must never receive `SYNC_BO` (`crates/hipfire-xdna/src/lib.rs:24-46,937-962,1018-1041`). It does **not** connect the MoE route to a GPU producer/ring/tape, and has no HIP-VMM export wrapper. Sidecar/registry plumbing exists, but the loader still supplies `xdna: None` (`crates/hipfire-loader/src/lib.rs:2603-2605`). New integration must reuse the proven NPU transport mechanisms, not mistake this older probe crate for retained co-op dispatch.

## 2. Hand off a static expert subset through tape and ring

**Ownership policy.** Select a static expert set per layer at model load, initially a contiguous range. Use offline routing statistics to balance row mass and padded-wave cost; do not assume expert IDs are equally popular. `f_rows = sum(M_e for NPU experts)/81920` and `f_experts = E/512` coincide only in the uniform-M ESTIMATE below. Preserve the original route/permutation. Give GPU GEMMs a separate tile-ID view with NPU-owned tiles replaced by `-1`; retain the original view for producer/scatter/combine. Existing GPU early exits support this view, but building/selecting it is new integration (`crates/peacemaker-isa/src/kernels/qwen4_moe_sym.rs:19-23`).

**GPU producer.** Keep rotate/quantization on the GPU. The original A4 codes and 72-byte `block_i4_128` blocks are immutable, address-stable inputs that IEF15 never rewrites (`FC:22`). A layout producer gathers `sorted[offset[e]+r]`, decodes the signed A4 codes without requantizing, and writes zero-padded signed-int8 A into the live-shared slot arena (§3). It also gathers the matching IEF15 activation sidecars `D u16[epoch][token]` (each ≤32767) and `b i16[token]` into the same arena order. A GPU D/b pass produces them after every A4 write, leaving Xq bytes untouched (`FC:21,53`). Padding rows get zero A and output +0 (`FC:24`); nonfinite or negative input scales must be rejected before a slot is published (`FC:17`). The producer must reproduce `ArrayDesign::pack_in`, not copy the 72-byte GPU sidecar wholesale; the exact D/b arena layout belongs to the unwritten TM64/TN32 packer (`FC:65`) and is open. V9 packs column segments, M/N waves, K64 chunks and alternating eight-row halves (`crates/peacemaker-npu/src/kernels/gemm_array.rs:659-699,1696-1698`). G80 packs four injecting-column segments `(wave, chunk, E/O half)` (`crates/peacemaker-npu/src/kernels/gemm_g80.rs:349-363`). A fused rotate+pack producer is an option only after parity with the existing quantizer; initially use a separate pack pass and charge its measured envelope later.

**Weights.** At load, expand symmetric signed int4 weights losslessly to int8 and pack each selected expert's B once; the original nibbles stay the immutable source. Build the IEF15 weight sidecars once on the CPU: unsigned `S u16[feature][K/128]` (sign bit unused, each ≤32767) and `a i16[feature]`, with `a` the smallest integer exponent that fits (`FC:15,20`; load-time CPU packing per `FC:53`). GU is `[1280][20]` and down `[2560][5]` per expert; gate and up carry separate S/a (`FC:37`). The S/a sidecars are immutable, model-lifetime data that the NPU streams like B. GPU-owned experts need the same contract; mixing current and IEF15 numerics inside one layer is not covered by `FC` (`FC:27` forbids it for unsupported shapes), so I treat a GPU IEF15 entry for the remaining experts as required **[INFERENCE]**. Keep packed B and S/a resident in shared host-backed DRAM for the model lifetime. “Resident” does **not** mean all layer weights fit on-array: each group still streams its B through the NPU DRAM port. G80 B order is column/chunk/even-odd N tile with 8×8 blocks (`crates/peacemaker-npu/src/kernels/gemm_g80.rs:366-382`); V9 uses the corresponding `pack_b` (`crates/peacemaker-npu/src/kernels/gemm_array.rs:748-777`). Keep GPU weights for the remaining experts and any admitted capacity fallback.

**Commands.** One `experts::grouped` TXN per projection/group: first expert FULL, remaining experts LEAN, one POLL/DRAIN and one DONE around the **entire group**, not around every expert (`crates/peacemaker-npu/src/kernels/experts.rs:12-31,79-88,119-146`). Per-layer gate_up and down have distinct program/config keys and slots. Retain their prepared commands before routing; no CPU derives commands from live M. A step tape records their order; `ProgramKey` hashes PDI, canonical TXN and declared dynamic words, `ConfigKey` hashes PDI, and tape identity excludes binding values (`crates/railgun/src/npu/tape.rs:13-61`). Sequence changes must be declared `Seq` patches; collision, stale-key and fresh-encode checks remain mandatory (`docs/npu/railgun-npu.md:131-146`). Same ConfigKey permits lean tails; switching gate_up↔down does not. The mixed-projection configuration schedule is a gate, not a claim that P1's informational `prog` field can switch a PDI. P1 has one PDI per command; alternation needs legal pre-armed configuration transitions or another proven retained scheduling arrangement. Existing single-flight hipfire-xdna alone cannot supply it; do not assume two concurrent full-array contexts or free PDI reloads.

**Distinct cache key.** IEF15 changes the program/output-contract identity (`FC:25`). The IEF15 gate_up and down programs get their own PDI/TXN and their own `ProgramKey`/`ConfigKey`, and the tape identity declares `fold_contract=ief15`, output `BF16[grouped_p][feature]`, the new core geometry (TM64/TN32, §5) and E. No int8 or int32 ring program may share or reinterpret their keys. Ring v1 layout and semantics are unchanged.

**Variable M, fixed arenas.** Build for the group's maximum admitted M: V9 covers up to 512 with one 512-row wave; G80 uses `round_up(max_M,256)` and either one or two waves. The landed bench computes `wave_m = design.waves()*design.wave_m()` and passes that padded row extent to `grouped` (`crates/npu-tools/src/bin/npu-experts.rs:78-81,143-146`). All experts in that group pay its padded extent. The GPU uses current true M only to gather/zero-fill A and D/b; under option (a) there is no C scatter (§5). Changing M does not patch the program. Empty experts get all-zero A and no useful output, but still pay the static body. Do not infer a small-wave time from average M when one hot expert forces the larger design. Counts over the arena capacity must stay GPU-owned through a device-side ownership decision shared by both projections and the C consumer; the corresponding preplanned NPU body gets dummy zeros and its C is ignored. No truncation, host count relay or in-flight TXN rebuild is allowed.

P1 reads **only `seq`**. Offsets and shapes are fixed at build time: reserve a stable A/D region per `(layer, expert, projection, in-flight bank)`, one C arena per `(layer, projection, bank)`, and fixed B and S/a offsets, and copy into them (`docs/npu/railgun-npu.md:62-73,91-100`). The C arena spans the **entire original grouped-row BF16 output** `[grouped_rows][N]`, so GPU combine can read it directly. Its size, `grouped_rows*N*2` B per layer/projection/bank, is a capacity gate to quantify, and `grouped_rows` is only a host-known bound (§1). Dynamic `a_off/b_off/c_off/m/prog` slot fields are informational, not executable descriptors. CPU `patch_seq`/retained submission re-arms each round, queued ahead without standing between publish and done (`crates/peacemaker-npu/src/kernels/ring.rs:96-100,375-390`). Preserve the power-of-two slot layout, equality waits, no zero/sentinel sequence and slot-reuse discipline. Reuse arenas only after both NPU completion and the GPU C consumer; NPU DONE alone does not prove the GPU has finished reading.

## 3. Buffer ownership and visibility

**Transport: exported HIP VMM is withdrawn; anonymous userptr pages are the measured fixture path.** **REPORTED-MEASURED (Halo), negative:** host-pinned and uncached VMM allocations exported with `hipMemGetHandleForAddressRange` and imported by amdxdna (64 KiB–16 MiB) alias only the bytes present at export. After import, GPU writes are invisible to the NPU/CPU dma-buf view, dma-buf writes are invisible to the GPU, and re-export returns the old bytes. Local logs show `MISMATCH` in every probed kind and size, stopping before any NPU command: `logs/coop/coop-w2-45aef0b.log:4-8` and `logs/coop/coop-w2-c6193a6.log:4-9` (originals on the measurement host). A cache flush alone cannot repair a snapshot alias, so ring/live A/D/C must not use an exported HIP allocation. A stable immutable B export (bytes final before export) is not evidence of live sharing.

**MEASURED fixture proof, positive (`logs/coop/coop-w2-0fe69f9.log`).** Ring, A and C are anonymous host pages (4096 B each) pinned by an amdxdna **userptr BO** (`Device::userptr_bo`) and registered with `hipHostRegister` (`HipRuntime::register_host`). Aliasing passes in all four directions: cpu→gpu, gpu→cpu, npu-map==cpu and npu-map→cpu (`:4`). A GPU-driven empty ring measures publish→done p50 **3.13 µs** (`:14`), done→GPU-observation upper bound **0.56 µs** (`:15`), uncached poll load **0.24 µs** (`:16`), publish store **0.20 µs** (`:17`) and GPU run interval **5.77 µs** (`:18`); these are p50 values and the max outliers reach about 1.6 ms. Serialized CPU-free fixtures (GPU A copy + publish + NPU GEMM + done + GPU C read) were exact, 32/32 C reads equal to eager: **12.11 TOPS** at 512×1280×2560 (`:37-38`) and **9.58 TOPS** at 512×2560×640 (`:56-57`). This proves a live GPU↔NPU transport for the existing int8 fixtures. It is **not** a dense-IEF proof: no D/b sidecars, no BF16 grouped-p placement, no hipGraph/PM4 replay, no production kernel or arena size, and no overlap or contention result. Control-line caching and maintenance at production scale still need their own checks.

| Buffer (live-shared rows use the §3 anonymous-page class) | Writer → reader | Lifetime / memory kind | Maintenance / ownership rule |
|---|---|---|---|
| router logits, top-k IDs/weights, counts, offsets, sorted/inverse/tile IDs | GPU router/grouping → GPU pack, GPU GEMMs, combine | per layer; existing GPU device buffers | ordinary same-device dependencies; never download counts |
| gate/down GPU `xq` (A4 codes, 72-B blocks) | GPU rotate/quantizer → GPU GEMM, D/b pass and layout producer | per layer; existing device scratch | immutable under IEF15; same-device ordering; gate and down use different sidecars |
| activation sidecars `D`/`b` | GPU D/b pass → layout producer | per layer/projection; device scratch | generated after every A4 write (`FC:53`); no stale reuse after a producer write |
| packed A + D/b arenas | GPU layout producer → NPU shim DMA | fixed per-layer/projection/expert bank; anonymous host pages, userptr BO + `hipHostRegister` (fixture-proven for A at 4 KiB, not for D/b or production size) | compute completion + GL2 writeback **before** seq publish; no overwrite until NPU consumed |
| packed B and weight sidecars `S`/`a` | CPU load/pack → NPU; GPU copy of S/a for GPU-owned experts | model lifetime; written before NPU use; immutable | CPU publish/flush once; export is acceptable only as an immutable snapshot; no imported-BO `SYNC_BO` |
| C arena: BF16 pre-SiLU g/u or final down, the entire original grouped-row span | NPU DMA → GPU SwiGLU (g/u, in place at `p`) or GPU combine (down, direct) | fixed per-layer/projection/bank; same anonymous-page class (fixture-proven for int8 C reads, not for BF16 grouped-p placement) | NPU all-channel C SYNC then DONE; GPU waits, then acquire/GL2 invalidate before loads; NPU-owned and GPU-owned rows disjoint; placement is a gate (§5) |
| fallback: int32 epoch partials | NPU DMA → GPU fold/unpack/scatter | separate arena; current-contract fallback only | same C ordering rules; not part of IEF15 |
| ring header | CPU once → NPU/GPU | session lifetime; anonymous host pages (fixture-proven, 4 KiB) | initialize/publish before retained jobs |
| slot lines (`seq` last) | GPU PM4 → NPU poll | session lifetime; GPU uncached store measured in fixture (`:17`); production control mapping still to confirm | confirmed memory write after A release; GPU is sole publisher |
| done lines | NPU DMA → GPU `WAIT_REG_MEM` | session lifetime; same class; fixture observed done within 0.56 µs upper bound (`:15`) | NPU is sole writer; equality on this slot's generation; do not poll a stale cached line |
| SwiGLU output (`y_gate_up_grouped`, BF16 `[p][640]`), combine order scratch | GPU SwiGLU/GPU GEMM → rotate128/combine | per layer; device memory by default | NPU-owned and GPU-owned row ranges disjoint; rejoin before consumers |
| PDI/TXN/CMD BOs, tape keys | CPU builder/re-arm → firmware | retained session/program lifetime; NPU DEV/CMD BOs | patch declared sequence words only; never mutate an in-flight instruction BO |

The frozen ring is header at byte 0, slot `s` at `64*(1+s)`, done `s` at `64*(1+nslots+s)`; no old single-done-line ABI or `>=` wait (`docs/npu/railgun-npu.md:62-73,114-129`). Ring v1 is unchanged by IEF15 (`FC:25`). A release-memory **value** helper is not by itself the required GL2 writeback: hipfire's helper only emits the fence; a separate conservative global boundary performs compute idle and `ACQUIRE_MEM` GCR `0x0c380` (`crates/redline-rocr/src/pm4_gfx10.rs:325-341,459-493`). Validate exact gfx1151 packet/mapping behavior, including acquire after NPU stores, before weakening these boundaries. The coop-w2 fixture shows a working GPU-driven path on 4 KiB anonymous pages; it does not validate production arenas, gate/up+down alternation or concurrent GPU load.

## 4. Per-layer synchronization timeline

```text
GPU: router/top10 → stable grouping → rotate256 (A4 codes unchanged) → D/b pass → pack selected A + D/b → release → publish GU seq
GPU:                                  run GPU-owned GU (same IEF15 contract) ----┐
NPU:                                   POLL → grouped IEF15 GU → BF16 g/u at original p → C SYNC → DONE GU
GPU:                                      wait GU; acquire; SwiGLU on NPU-owned rows in place ┘
GPU:                         rotate128 → D/b pass → pack selected down A + D/b → release → publish DN seq
GPU:                                  run GPU-owned down -------------------------┐
NPU:                                   POLL → grouped IEF15 down → BF16 down at original p → C SYNC → DONE DN
GPU:                                      wait DN; acquire ┘ (no unpack/scatter)
GPU:                                original grouped-top10 BF16 combine reads the C arena directly → next layer
```

The GPU keeps SwiGLU (current BF16 boundaries, applied in place to NPU-owned g/u rows) and rotate128. The SwiGLU pass over NPU-owned rows is new GPU work with unmeasured cost. A final-BF16 gate/up seam means **two BF16 pre-SiLU projections**, not pretending the current NPU already emits activated `[640]` rows. Down needs no GPU unpack or scatter; combine reads the C arena. Under the current-contract fallback (§5 (b)) the wait stage would also carry fold, unpack and scatter. There are **two kick/wait transactions, four directional hand-offs per layer**: **ESTIMATE 96 transactions / 192 directions across 48 layers**, for one group per projection. Multiple groups multiply slot waits; grouping is not permission to report one transaction for many separately ring-gated commands. Independent GPU work must be issued before the CP completion wait: `WAIT_REG_MEM` stalls its queue, not the whole GPU (`crates/redline-rocr/src/pm4_gfx10.rs:496-512`).

**MEASURED** CPU-producer empty-ring publish→done p50 is 4.5 µs, not a separate 4.5 µs in each direction (`report.md:622-627`). Charge **ESTIMATE 96*4.5 = 0.432 ms** per step; the tables use a conservative **192*4.5 = 0.864 ms** bound. First-slot p99 is **MEASURED 51–73 µs**, and submit→first-done **50–55 µs**; queue rounds early rather than omit re-arm latency. Those measurements do not include a GPU producer/cache boundary.

`WAIT_REG_MEM` uses equality, full mask and a source-defined poll-interval field of **4**, whose wall-time conversion/cost is **unmeasured**, not “4 µs” (`crates/redline-rocr/src/pm4_gfx10.rs:499-510`). Keep `p` = incremental poll/detection overhead per wait, excluding GEMM execution, in the cost model:

| Poll p, ESTIMATE | 96 completion waits | 96 completion + 96 reuse waits | Conservative ring + completion polling |
|---:|---:|---:|---:|
| 0.5 µs | 0.048 ms | 0.096 ms | 0.912 ms |
| 2 µs | 0.192 ms | 0.384 ms | 1.056 ms |
| 10 µs | 0.960 ms | 1.920 ms | 1.824 ms |

Producer release, consumer acquire, packing, D/b production, the GPU SwiGLU pass, configuration switching and any extra host counts are **not measured by the 4.5 µs probe**. Charge them separately. Existing glue is already in the base; only genuinely incremental glue is added. A poll wait's entire blocked duration is not extra overhead when it is already accounted by `max(GPU,NPU)`.

## 5. Rejoin and the EpilogueContract seam

**Rejoin under option (a).** The IEF15 C arena is the entire per-layer/projection/bank original grouped-row BF16 span, `C[p*N + feature]` with `p = offset[e] + localr`, indexed exactly like `y_down_grouped`. GPU combine therefore reads it directly: `p = inverse_perm[token*10+rank]`, BF16 `Y[p*hidden+feature]`, contributions ordered by ascending expert ID, not by device completion or router score (`kernels/src/moe_down_combine_grouped_top10.hip:90-121,139-166`). **No C conversion, unpack or scatter pass runs on option (a).** Gate/up g/u rows are consumed in place at the same `p` by the GPU SwiGLU pass. GPU-owned and NPU-owned rows are disjoint. P1 stays fixed-base; placement metadata maps original `p = offset[e]+localr` to NPU-owned output rows. **Open gate:** the actual NPU output/DMA placement for dynamic grouped offsets is new and unimplemented. Existing V9/G80 packed C does not do it, and the informational ring fields must not be reinterpreted to do it (§8 gate 3). Preserve original top-k IDs/weights, BF16 RNE, padding zeros (+0) and residual/_zinit behavior.

`EpilogueContract` identifies scale representation, epoch order, rounding/non-finite semantics, output encoding and row addressing independently of tape/ring:

| Option | Contract and status | Rejoin |
|---|---|---|
| (a) **Frozen IEF15**: NPU emits final BF16 rows (`FC` §2) | Sidecars are unsigned `S u16[feature][epoch]`, `a i16[feature]`, `D u16[epoch][token]`, `b i16[token]`, with S and D ≤32767 (`FC:15-21`). Original nibbles and A4 codes are immutable. `I = Σ_e i64(S)*i64(D)*i64(C)` is exact, then `Y = RNE_binary32(I*2^(a+b))`: one rounding (`FC:17`). Output is BF16 RNE, with g and u separate and pre-SiLU for gate/up and final for down, `[grouped_slot][feature]` (`FC:24`). GPU retains the current BF16 SwiGLU boundaries. Ring v1 is unchanged and the program/output identity is new (`FC:25`). Historical proposal: an opt-in `HIPFIRE_IU4_FOLD_CONTRACT=ief15` selector defaulting to `current` (`FC:27`); that variable was never implemented, is absent from the source and env docs, and is retired. Frozen for the dark experiment, **not an admission**: payoff and quality are gate-conditional (`FC:3`). The NPU kernel is unwritten. `docs/dense27b.md:185-206` is the earlier IEF candidate, not the IEF15 specification. | Direct: no conversion or scatter. GPU SwiGLU consumes g/u in place; combine reads the down C arena directly. |
| (b) **Current-contract fallback**: exact int32 per-K128-epoch partials, GPU fold. Separate from (a); not IEF15. | Keeps today's ascending `fmaf(RN(f32(sc_e)*d_e), C_e, sum)` and BF16 RNE/SiLU (`crates/peacemaker-isa/src/kernels/qwen4_moe_sym.rs:25-33,518-528`). Keep original f16 weight sc and **f32 activation d**; neither is replaced by an approximate scale, and no IEF15 sidecars are used. | GPU performs the original fold and SwiGLU, then a grouped unpack/scatter, because native packed partials do not have the grouped layout. |
| (c) int-exact int32 whole-K GEMM | Exact integer dot product alone is **not IU4 parity**: epoch-dependent scales and rounding cannot be reconstructed from a single unscaled K sum. Current throughput fixtures prove their int8-shift epilogue, not (a)/(b). G80 currently rejects I32 because Cout aliases accumulator storage (`crates/peacemaker-npu/src/kernels/gemm_g80.rs:461-467`). | Benchmark/control only; no admission to production combine. |

The NPU's emulated exact fp32 fold costs **ESTIMATE 4–5.5× GEMM** (`docs/dense27b.md:187`); exact partials were **MEASURED write-bound near 3.3 TOPS** in the dense study (`docs/dense27b.md:171-177`). Neither is hidden inside the grouped-int8 timings.

**The frozen contract is not current-fold byte equivalence.** No IEF15 candidate reproduces today's fold bytes (`FC:9`). GPU/NPU parity means byte equality under the **new** contract. Default-route byte identity for flag absent or `current` is a separate hard gate. Shipping IEF15 numerics additionally needs model-level quality admission (`FC:59,92`). New core geometry (frozen design decision, not derived from `FC:69`) is **TM64/TN32**, not TM64/TN64: live tiles are 36 KiB (I 16 KiB, C 8 KiB, A ping/pong 8 KiB, B ping/pong 4 KiB). Metadata at the maximum E=136 is `64*136*2` (S) + `32*136*2` (D) + 192 (a/b) = 26,304 B = 25.7 KiB. Total **61.7 KiB** of the 64 KiB, leaving 2,368 B for controls. Flash-Next's E=20/5 needs far less metadata, but the geometry is sized for the shared maximum.

For **ESTIMATE M=160**, the current-contract fallback (b) has the following exact-partial traffic **optimistic useful-row floors** (decimal bandwidth; no wave/N padding):

| Projection | Exact partial bytes/expert, ESTIMATE | NPU write at MEASURED ~55 GB/s, ESTIMATE time | GPU read at ESTIMATE ~240 GB/s, ESTIMATE time |
|---|---:|---:|---:|
| gate_up | `160*1280*20*4 = 16,384,000 B` | 297.89 µs | 68.27 µs |
| down | `160*2560*5*4 = 8,192,000 B` | 148.95 µs | 34.13 µs |
| pair | 24,576,000 B | 446.84 µs | 102.40 µs |

At the optimistic G80 split `f=0.111216`, **ESTIMATE 67.17 GB** of partials cross each way per step: at least **1,221.31 ms NPU writing** and **279.88 ms GPU reading** in aggregate, before GPU fold instructions. These cannot simply be added to measured GEMM time because write/compute may pipeline; they are nevertheless binding lower bounds. Even granting **free NPU compute**, replacing H by these write-only floors in §6 gives **ESTIMATE −21.94 ms** savings at its optimal split; including serial GPU reads after group DONE gives **−32.63 ms**, before padding/fold/pack contention. At the proposed G80 split the write-only floor already costs **ESTIMATE 412.28 ms more than baseline expert buckets**. The exact fallback kills the projected gain, rather than a small epilogue tax rescuing it.

## 6. Expert share and step model

**Canonical profile found, not carried over from the asymmetric artifact.** `FN:12` records `qwen3.8-flash-next-gptq3.mq4`, md5 `be007fc3219e9f6cdb1d4dfa8380625f`, and **MEASURED/observed 48/48 symmetric expert layers**. `FN:10-11` identifies the 4j GPU binary. Its synchronized pp8192 step and GU/down buckets are the appropriate measured baseline for this design. They remain provisional two-process traced observations, not an accepted median or a co-op measurement (`FN:3,24,29`).

**Symmetry evidence, two separate items.** (i) The existing model profile `FN:12`: **MEASURED/observed 48/48 symmetric expert layers** (above). (ii) An **externally reported observation**: all 553 inspected canonical GPTQ3 tensors are symmetric. I have no log or artifact citation for (ii), so it adds no citable evidence here and does not replace (i) or the load-time check (§1).

Canonical input ledger, **MEASURED**, `base=4517.110 ms`, `G_gu=485.886 ms`, `G_dn=324.005 ms`; second process `5062.900 / 550.992 / 362.839 ms` (`FN:24,29,41-43,66`). The two samples are not a median, and the approximately 1929 tok/s CLI denominator has no interchangeable bucket trace (`FN:18`). Model dimensions are source-defined: 48 layers, 512 experts, top-10, hidden 2560, intermediate 640 (`FN:35`; `report.md:79-80`). **ESTIMATE** uniform routed M is `8192*10/512=160`; work per step is `25.7698 TOP` gate_up and `12.8849 TOP` down. These buckets are gptq3 observations; all co-op splits/savings computed from them are ESTIMATE.

**MEASURED** grouped V9 timings, int8-shift fixtures, ring off (`report.md:637-655`):

| Group E | GU µs/expert at M160 | down µs/expert at M160 | Across measured M64…512 |
|---:|---:|---:|---|
| 8 | 217 | 130 | GU 216–234; down 129–131 |
| 32 | 210 | 116 | GU 208–210; down 113–116 |
| 64 | 201 | 116 | GU 201–203; down 115–116 |

Prepared lean **MEASURED 30.2/32.9 busy TOPS at M4096** (`report.md:613`) and eager **~360/~280 µs per expert** (`report.md:579-587`) are context, not expert-co-op rates.

<!-- COOP-W1 -->
**COOP-W1 filled with MEASURED grouped G80 gate_up**, not single-submit extrapolation. These are original-model fixture rows with IEF15 metadata excluded. Entry is V9 → G80 µs/expert; all checked experts exact against eager/CPU **fixture** oracles, ring off. Source `logs/coop/coop-w1b-bd3ac0e.log`:

| E | M64 | M128 | M160 | M256 | M512 | Source lines |
|---:|---:|---:|---:|---:|---:|---|
| 8 | 219.0 → 141.7 | 223.3 → 140.2 | 217.0 → 140.8 | 219.9 → 140.4 | 217.6 → 172.3 | `:440,850,1232,1614,1922,2230,2426,2622,4094,4197` |
| 32 | 209.3 → 124.7 | 210.5 → 122.4 | 208.6 → 128.7 | 209.8 → 125.7 | 208.1 → 154.8 | `:2818,3014,3117,3220,3304,3388,3444,3500,4230,4263` |
| 64 | 201.1 → 124.0 | 201.3 → 124.2 | 201.7 → 124.1 | 201.3 → 124.0 | 200.1 → 155.1 | `:3603,3706,3762,3818,3865,3912,3945,3978,4284,4305` |

M160 G80 useful rate is **MEASURED 7.45 / 8.15 / 8.45 TOPS** for E8/E32/E64 (`:2230,3388,3912`); M512 uses two waves. Down has no G80 design. The older **MEASURED single-submit G80 289/292/295/333 µs** at M64/160/256/512 (`report.md:680-683`) does not predict these grouped rates.

### Equation and arithmetic

Use `docs/levers-joint.md:244-248` with the now-measured expert times, not its old M4096 rate assumptions:

```text
H_gu = 48*512*u_gu/1000 = 24.576*u_gu ms
H_dn = 48*512*u_dn/1000 = 24.576*u_dn ms
step(f) = base - (G_gu+G_dn)
        + max(alpha*(1-f)*G_gu, f*H_gu)
        + max(alpha*(1-f)*G_dn, f*H_dn) + h
```

**ESTIMATE** uniform row/expert split and linear GPU bucket scaling; use `h=0.864 ms` conservative ring charge, and add measured packing/cache/epilogue/configuration cost and `96*p/1000` polling later. There is no stolen controller core: P1 uses the firmware interpreter, so the old `31/32` factor does not apply (`docs/npu/railgun-npu.md:57-60`). H already includes grouped TXN/submit overhead. Primary rows assume no NPU contention loss; a separate sensitivity uses the **MEASURED old-M4 rate ratio** `beta=22.16/25.60=0.865625` (`report.md:469-470`) via `H/beta`.

**MEASURED** M4 GPU slowdown is +10.1% at roughly 34 GB/s (`report.md:462-470`; `docs/levers-joint.md:288`); using **ESTIMATE alpha=1.101** for these phase-local groups transfers that observation. Packed traffic gives **ESTIMATE** V9 GU ~43.0 GB/s, G80 GU ~34.3 GB/s, V9 down ~39.5 GB/s at their fixture times; G80 does not justify assuming near-zero derate. Dense traffic points are **MEASURED but drift-flagged/unreliable**: +2.2/+5.9/+11.3% at 5.3/16.0/46.9 GB/s (`docs/dense27b.md:140-148`). Those alpha values are sensitivities, not calibrated gptq3 curves.

The piecewise-linear minimum is found by evaluating `f=0,1` and each phase balance `f_x=alpha*G_x/(H_x+alpha*G_x)`. Do not automatically choose gate_up's balance: V9 is GU-limited, G80+V9 becomes **down-limited** at Halo rates. For the selected V9 E64 proxy, **ESTIMATE** `H=4939.776/2850.816 ms`, `f=0.097714`, phase maxima `482.687/321.872 ms`; `4517.110-809.891+804.559+0.864=4512.642 ms`. For G80 E64 GU, `H=3049.882/2850.816`, `f=0.111216`, maxima `475.464/317.056`; step `4500.603 ms`. The formula remains `H_gu=24.576*u_G80`; changing the measured COOP-W1 GU rate below the down balance buys no further saving without improving down.

<!-- COOP-W1 -->
| Rates used, MEASURED fixture µs/expert (original model, IEF15 metadata excluded) | Optimal f / experts, ESTIMATE | Step ms, ESTIMATE | Saved ms, ESTIMATE | Step-time reduction, ESTIMATE |
|---|---:|---:|---:|---:|
| V9 E8 217 / 130 | 9.12% / 46.7 | 4518.481 | −1.37 | −0.03% |
| V9 E32 210 / 116 | 9.39% / 48.1 | 4516.026 | 1.08 | 0.02% |
| V9 E64 201 / 116 | 9.77% / 50.0 | 4512.642 | 4.47 | 0.10% |
| G80 E8 GU 140.8 + V9 down 130 | 10.04% / 51.4 | 4510.210 | 6.90 | 0.15% |
| G80 E32 GU 128.7 + V9 down 116 | 11.12% / 56.9 | 4500.603 | 16.51 | 0.37% |
| G80 E64 GU 124.1 + V9 down 116 | 11.12% / 56.9 | 4500.603 | 16.51 | 0.37% |
| V9 E64, second FN trace | 10.94% / 56.0 | 5046.015 | 16.88 | 0.33% |
| G80 E64 + V9 down, second FN trace | 12.29% / 62.9 | 5032.400 | 30.50 | 0.60% |

These are **optimistic transfers**, not measurements of groups with exactly 50/57 experts or of BF16/fold-contract kernels. The measured int8 C epilogue is not final BF16; changed C traffic is another unmeasured cost, not covered by a free fold assumption. Benchmark E is amortization regime, not a fractional group physically selected by the router. At uniform M, sending exactly 64 experts (`f=0.125`) with E64 timings predicts **ESTIMATE −164.80 ms V9 / −15.42 ms G80+V9**. Exactly 48 with E32-proxy timings gives **ESTIMATE 0.93 ms** before group-size penalties; actual E48 rates need measurement. Static IDs must be chosen from true routing histograms and measured useful/padded load, not the decimal optimum rounded to a large existing group.

### IEF15 metadata read bytes and adjusted H

Useful sidecar metadata read per M160 expert is `2*N*epochs + 2*N + 2*M*epochs + 2*M` B: S (2 B per feature-epoch), a, D (2 B per token-epoch) and b. These are **read-once lower bounds** on useful rows with no per-core duplication, which can raise them. Times are **ESTIMATE** at the same ~55 GB/s assumed in §5; they are not a measured sidecar-read rate.

| Projection (N, E) | S/a B | D/b B | Total B | Time at 55 GB/s |
|---|---:|---:|---:|---:|
| GU (N1280, E20), M160 | 53,760 | 6,720 | 60,480 | 1.099636 µs |
| down (N2560, E5), M160 | 30,720 | 1,920 | 32,640 | 0.593455 µs |
| V9 GU, M512 wave-padded | 53,760 | 21,504 | 75,264 | 1.368436 µs |
| G80 GU, M256 wave-padded | 53,760 | 10,752 | 64,512 | 1.172945 µs |
| V9 down, M512 wave-padded | 30,720 | 6,144 | 36,864 | 0.670255 µs |

Proposed adjusted rate, with `delta_meta_x` in µs/expert:

```text
H_x = 24.576*(u_x + delta_meta_x)/beta + newtile_x_fold_cost   [ms]
```

The metadata term is an **additive, conservative serial service surcharge**, not a measured latency. `newtile_x_fold_cost` is the actual TM64/TN32 IEF15 tile and fold cost, and it is **unmeasured and set to 0** in the rows below, which makes them optimistic. The step formula, `alpha=1.101`, `h=0.864 ms`, `base=4517.110`, `G_gu=485.886` and `G_dn=324.005` are unchanged. Fixture-only rows use the MEASURED V9/G80 kernels, which are unchanged and unaware of the sidecars.

| Case, ESTIMATE | u_gu / u_dn used, µs | f / experts | H_gu / H_dn ms | Step ms | Saved ms | Reduction |
|---|---|---:|---|---:|---:|---:|
| V9 E64, fixture only (metadata excluded) | 201 / 116 | 9.77% / 50.0 | 4939.776 / 2850.816 | 4512.642 | 4.47 | 0.10% |
| V9 E64, metadata-adjusted | 202.099636 / 116.593455 | 9.72344% / 49.8 | 4966.800663 / 2865.400739 | 4513.070043 | 4.039957 | 0.0894368% |
| G80 E64 + V9 down, fixture only | 124.1 / 116 | 11.12% / 56.9 | 3049.882 / 2850.816 | 4500.603 | 16.51 | 0.37% |
| G80 E64 + V9 down, metadata-adjusted | 125.199636 / 116.593455 | 11.07123% / 56.7 | 3076.906263 / 2865.400739 | 4501.051926 | 16.058074 | 0.3554944% |
| V9, wave-padded D/b | 202.368436 / 116.670255 | 9.71178% / 49.7 | 4973.406683 / 2867.288187 | 4513.174023 | 3.93598 | 0.0871% |
| G80 + V9 down, wave-padded D/b | 125.272945 / 116.670255 | 11.06475% / 56.7 | 3078.707896 / 2867.288187 | 4501.109721 | 16.00028 | 0.3542% |

These are **not** timings of the TM64/TN32 IEF15 kernel and claim no free fold. Gain-ceiling check, **ESTIMATE**: the model's GPU-owned expert phases total about 805 ms (V9) or 793 ms (G80 + V9 down), so a GPU-owned-expert slowdown above about **0.50% (V9) or 2.03% (G80 + V9 down)** erases the adjusted saving. This assumes the slowdown applies uniformly to those phases and that GPU-owned experts must also run IEF15 (§2). The 1.03× G0 threshold (§8) is therefore a kill gate, not a budget.

### Sensitivity and target rates

All rows below are **ESTIMATE**, original-model fixture-only (IEF15 metadata excluded), use the first FN trace, retain h, and omit incremental epilogue/packing:

| alpha / NPU beta | V9 f / saved ms / time reduction | G80+V9 f / saved ms / time reduction |
|---|---|---|
| 1.000 / 1.000, no contention | 8.96% / 71.66 / 1.59% | 10.21% / 81.79 / 1.81% |
| 1.022 / 1.000, low-traffic sensitivity | 9.13% / 56.92 / 1.26% | 10.41% / 67.45 / 1.49% |
| 1.059 / 1.000, medium-traffic sensitivity | 9.43% / 32.26 / 0.71% | 10.74% / 43.49 / 0.96% |
| 1.101 / 1.000, M4-based primary | 9.77% / 4.47 / 0.10% | 11.12% / 16.51 / 0.37% |
| 1.101 / 0.865625, NPU also derated | 8.57% / −6.24 / −0.14% | 9.77% / 4.48 / 0.10% |
| 1.113 / 1.000, high-traffic sensitivity | 9.87% / −3.44 / −0.08% | 11.23% / 8.84 / 0.20% |

The earlier IEF cost model's **ESTIMATE 1.15–1.33×** NPU GEMM multiplier (`docs/dense27b.md:202-206`), if applicable to both projections, reduces the G80 primary saving to **ESTIMATE +4.84…−5.98 ms**, before GPU cost. Do not give IEF15 those rates without measuring its actual epilogue. Whole-step rather than phase-local contention additionally charges **ESTIMATE `0.101*(4517.110-809.891)=374.43 ms`** and reverses the sign. Stop NPU traffic outside the expert overlap.

**+10%/+15% on Halo is not reachable at these rates.** Here “gain” means time saved/base; tok/s improvement is separately `base/step-1`. Even with zero contention the best current proxy is below 2%. The expert-only Amdahl ceiling is **ESTIMATE 17.93%** of the first base (second trace 18.05%); reaching 15% time reduction would require nearly eliminating this bucket, not adding a 30-TOPS M4096 coprocessor.

For a theoretical same-f, phase-balanced pair, solve `q=(G_gu+G_dn-target*base-h)/(G_gu+G_dn)`, `f=1-q/alpha`, and `u_x=q*G_x/(24.576*f)`:

| Halo target | Required shared f, ESTIMATE | Required GU/down µs per M160 expert, ESTIMATE | Useful GU/down TOPS, ESTIMATE |
|---|---:|---:|---:|
| 10% less step time | 59.93% | 14.56 / 9.71 | 72.0 / 54.0 |
| 15% less step time | 85.26% | 3.76 / 2.51 | 278.6 / 208.9 |
| 10% more tok/s | 55.32% | 17.58 / 11.72 | 59.6 / 44.7 |
| 15% more tok/s | 75.35% | 7.12 / 4.75 | 147.2 / 110.4 |

These assume both projections improve, alpha remains 1.101 and the numeric epilogue is free. With present V9 down, even **zero-time gate_up** is capped at the same **ESTIMATE 16.51 ms** primary saving. Expanded-int8 B alone for GU costs **ESTIMATE 59.15 µs/expert** at MEASURED 55.4 GB/s (`docs/g80-feasibility.md:533-540`), already above the 10% requirement. Native int4 B may lower that floor but is not the measured grouped design. A smaller-M wave is a separate kernel, not reduced `tiles`: paper TM32/TM64 waves need new strides, epilogues and DMA lengths (`docs/g80-feasibility.md:483-531`).

## 7. gfx1152 / Krackan Point

Public **SPEC**, not measured co-op: [AMD Ryzen AI 7 350](https://www.amd.com/en/products/processors/laptop/ryzen/ai-300-series/amd-ryzen-ai-7-350.html) lists Krackan Point, Radeon 860M **8 graphics cores/CUs**, up to **3.0 GHz**, **two memory channels**, LPDDR5x-8000/DDR5-5600, **50 NPU TOPS**, and **66 overall TOPS**. Its two 64-bit DDR channels imply a **128-bit aggregate bus**; theoretical bandwidth is **ESTIMATE 128.0 GB/s LPDDR5x-8000 or 89.6 GB/s DDR5-5600**, not sustained NPU bandwidth. [AMD's CES announcement](https://newsroom.amd.com/news/amd-announces-expanded-consumer-and-commercial-ai/) gives Halo Max+ 395 **40 graphics CUs / 50 NPU TOPS** and Krackan 350/340 **50 NPU TOPS**. Thus the NPU stays large while the iGPU shrinks. The vendor-rated peak NPU share is **ESTIMATE `50/66=75.8%` of overall advertised TOPS** on 350; that includes CPU+GPU in the denominator and is not an IU4 useful-rate comparison.

For the same Flash-Next expert workload, take **ESTIMATE GPU expert time scale s=40/8=5** and reuse Halo NPU expert fixture times/alpha without claiming Krackan clocks, port bandwidth or power parity. Halo effective useful expert rates inferred from the measured canonical buckets are **ESTIMATE 53.04/39.77 TOPS**; dividing by s gives Krackan **10.61/7.95 TOPS**. The NPU's useful-rate contribution to the combined GU/down throughput is **ESTIMATE 33.0%/36.2% for V9**, **44.3%/36.2% for G80+V9**, not the 75.8% vendor-rated share. The same-f optimum includes alpha and can differ from these per-phase shares.

Use `G'_x=s*G_x` and `base'=r*(base-G_gu-G_dn)+s*(G_gu+G_dn)`, where r models everything else. **All results are ESTIMATE**, alpha=1.101, unchanged NPU rates, h=0.864 ms before extra group waits and epilogues:

| Krackan scenario | Estimated base ms | V9 NPU share / saved ms / time reduction | G80+V9 NPU share / saved ms / time reduction |
|---|---:|---|---|
| 860M, s=5; other work r=2 | 11463.893 | 35.13% / 1156.28 / 10.09% | 38.49% / 1306.05 / 11.39% |
| 860M, all work scales r=5 | 22585.550 | 35.13% / 1156.28 / 5.12% | 38.49% / 1306.05 / 5.78% |

This supports a larger *conditional* gain than Halo, not a measured 10–15% promise. CU scaling is not a bandwidth model: smaller memory bus, shared power, actual expert formats and changed cache/occupancy can invalidate it. More NPU experts require multiple measured-size groups; re-evaluate group/tape overhead. A canonical model must also fit resident memory including duplicate packed NPU B; **ESTIMATE ~15.24 GB** of V9-padded B for 57 experts/layer across 48 layers is extra to the GPU/model holdings. The historical Flash-Next M4 load was roughly **MEASURED/configured 125 GB** (`report.md:257-259`); do not assert that artifact fits a normal Krackan laptop. Capacity and gfx1152 kernel/transport admission precede any performance extrapolation.

The Krackan rows above are fixture-only and exclude IEF15 metadata. The §6 metadata adjustment is not recomputed for them.

## 8. Gates, kill points and open questions

Run in this order; thresholds are proposed **ESTIMATE engineering decisions**, not observed results. This document runs no hardware, builds or suites. The flash-next route is deprioritized (see the note at the top); these gates record what any future attempt would need.

1. **G0 GPU-speed kill, before ANY NPU work (`FC` §3 G0, `FC:33-47`).** Build the GPU-only IEF15 epilogue and the real-shape microbench on byte-identical A/Xq against the incumbent. It runs exclusively on Halo with at least 3 fresh processes. **KILL, with no NPU kernel, transport, ring or co-op work started, if the weighted real-shape GPU GEMM time is >1.03× the incumbent**, or if correctness fails or spills appear. Any single named GEMM >1.03× needs maintainer approval before proceeding (`FC:47`). Report the full GPU-alone pp8192 step once G1 exists, including the D/b sidecar producer and metadata passes (`FC:47,59`). G0's shapes are dense; the Flash-Next grouped GU/down GPU IEF15 entry is a separate follow-on slice (`FC:57`). I propose applying the same >1.03× rule to it, and note the §6 break-even slowdown of about 0.50% / 2.03%. Passing G0 authorizes no payoff claim.
2. **Quality, parity and admission.** Confirm canonical gptq3 expert headers and symmetric route and repeat the synchronized pp8192 GU/down/glue ledger. Kill any +10%/+15% Halo claim at the present rates. Parity oracle: `tools/npu/fold-model/src/lib.rs::{grid,fixed,round_scaled}` CPU → simulator → silicon, covering extreme C, cancellation, ties, subnormal and zero, padding and BF16 output. NPU bytes must equal GPU bytes under the new IEF15 contract. Default-route byte identity for flag absent or `current` is a separate hard gate, and IEF15 is not current-fold byte equivalence. Model-level quality admission is separate. The current-contract fallback (b) keeps today's numerics but, if it is the only exact choice, kills the speed route. There is no integer-dot “exact” shortcut.
3. **Layout, placement and capacity proof.** TM64/TN32 legal memory map (36 KiB live + max-E136 metadata 25.7 KiB = 61.7 KiB) with slot/RAW proof. GPU-produced A and D/b must match the NPU packer bytes for V9 and both G80 extents. **NPU output/DMA placement from dynamic grouped offsets into the original-`p` BF16 C span is a new, unimplemented explicit gate**: prove it with disjoint NPU/GPU rows, empty and hot experts, changing per-step M, and no overflow, truncation, duplicate or omitted rows. Quantify C-arena capacity (`grouped_rows*N*2` B per layer/projection/bank). Reject any capacity policy needing per-layer CPU count synchronization. Measure the incremental D/b, pack and SwiGLU envelopes.
4. **Production transport and coherence proof.** Exported HIP VMM buffers are REPORTED-MEASURED snapshot-only (§3) and must not carry ring/A/D/C. The anonymous-page class (NPU userptr BO + `hipHostRegister`) passed the int8 fixture proof (`coop-w2-0fe69f9.log`), which is not a dense-IEF proof. Remaining: production-size A/D/BF16-C arenas, poison/stale-cache negatives across repeated rounds, GL2 release-before-seq, control-line mapping/maintenance at scale, all-channel done ordering, acquire-before-C, consumer-safe arena reuse, and behavior under concurrent GPU load and hipGraph/PM4 replay. A CPU-producer ring and an immutable B export are not this gate. Never use imported-BO `SYNC_BO`.
5. **Retained schedule proof.** Grouped ring on silicon, mixed GU/down PDI/configuration transitions, fixed arena offsets, replay keys (distinct IEF15 keys), declared seq patches, queued re-arm without mutating in-flight TXN, round/slot reuse, timeout and teardown. P1's 60 s job bound and `S mod nslots == 0` multi-round condition remain (`docs/npu/railgun-npu.md:91-100`). Kill any schedule that deadlocks by waiting before issuing peer GPU work or depends on informational slot descriptors being executable.
6. **Single-layer then whole-step exact rejoin.** Compare GPU-only versus co-op at unchanged route IDs/weights and expert-order combine. Exercise gate/up BF16→SwiGLU→rotate128→down→combine end to end, including `_zinit`. Verify retained replay with mutated inputs, not just graph enablement. No double credit for existing glue.
7. **Matched contention/net-speed gate.** Thermally stable, interleaved alone/concurrent comparisons; phase-confine NPU traffic, measure GPU and NPU derate and real poll/cache/config costs, including IEF15 metadata and the actual tile/fold cost. Require positive net synchronized step savings beyond run-to-run noise; for a claimed +10%/+15%, measure that target on the canonical model. No presumed alpha curve from the drift-flagged dense run.
8. **Krackan-specific admission.** Verify memory capacity, gfx1152 IU4/native kernel admission, live-shared transport, NPU rate and shared-power derate before transferring Halo rates. Then fresh whole-model quality/parity and synchronized timing. Vendor-rated TOPS is not an acceptance gate.

Open questions: whether the measured anonymous-page transport holds at production arena sizes, with D/b sidecars, BF16 grouped-p outputs and concurrent GPU load, and what control-line maintenance it needs; NPU output/DMA placement for dynamic grouped offsets into the original-`p` BF16 span; the D/b arena layout and the TM64/TN32 packer; actual TM64/TN32 IEF15 tile/fold cost and sidecar read cost including per-core duplication; whether the GPU-owned experts can run the same IEF15 contract inside the 1.03× budget; the C-arena capacity; canonical routing skew/maxima; actual E50/E57 or mixed-group rates; safe retained GU/down configuration scheduling without hidden host serialization; GPU polling wall-time and release/acquire cost at scale; a representative repeated gptq3 baseline; packed-weight capacity on Krackan. None is resolved by the int8 fixture exactness, the 12.11/9.58 TOPS serialized transport fixtures, or the CPU-producer 4.5 µs probe.
