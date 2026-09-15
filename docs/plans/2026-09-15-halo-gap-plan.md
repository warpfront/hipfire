# Strix Halo prefill gap — plan of record

Date: 2026-09-15. **Author: Astra. `@reviewer` (Sol) is an eligible independent reviewer and owns the final technical veto.** Maintainer approval remains required.

Source base: `/home/kaden/ClaudeCode/warpfront/wt-lloyd`, `cadd315a1`; the parent subsequently corrected the predecessor's F STATUS at `6a64c0803`. Source citations below name the audited worktree lines, not the stale line numbers in the predecessors. Read in full: `2026-09-15-halo-gap-levers-max.md`, `2026-09-15-halo-prefill-plan.md`, and `2026-09-15-halo-prefill-levers.md` in this directory. This document supersedes their prospective execution order, not their recorded admissions/rejections.

**Planning evidence only:** no GPU program, build, formatter, linter, or project-wide test was run; no source was edited. CPU address enumeration and chunk-partition calculations below were run. All future speed ranges are **unmeasured estimates**, not promises or results. The supplied 600/582/395 t/s baseline is the campaign input.

## 1. Order and immutable contract

Execute: **C0 latent FA2 blob fix → A0 calibration → A5 grid order → F1 repaired oracle and F2 1024 host/cadence envelope → B6 fill issue structure → G1 sequential GDN constants → T0 tails → S1 co-scheduling screen → W2-versus-S1 Sol-spec decision.** This is the max brainstorm's first-week order, with the mandatory correctness fix in front and the host half of F made explicit (`docs/plans/2026-09-15-halo-gap-levers-max.md:592-604`). F2 has its own early kill; a failed chunk-size experiment must not block B6/G1/T0 at N512.

Keep admitted A2/B1/C2. Do not retry A1: caller evidence is 256 VGPR/40 spills. Do not revive C1's existing F32 chunked wrapper: its recorded CS32/Q8-fast ratio was 10.5 (`docs/plans/2026-09-15-halo-prefill-plan.md:431-438`). G2 is a different, deferred algebra kernel, **default-OFF** if ever built.

All admitted A/B/G1/T0 work and any future S1 implementation require **bit-identical outputs and state** against the accepted predecessor. F2 also requires bit identity; reducing the number of Q8-EF commits is not an allowed way to make it fast. Diagnostic stale-input/no-fold kernels must never feed model state. A failure is a rejection, not a reason to widen tolerance.

### Wall targets, not profile denominators

| Row | Current rate | Current wall | Target rate | Target wall | Required saving |
|---|---:|---:|---:|---:|---:|
| pp512 | 600 | 853.333 ms | 620 | 825.806 ms | **27.527 ms** |
| pp2048 | 582 | 3518.900 ms | 710 | 2884.507 ms | **634.393 ms** |
| pp8192 | 530, supplied context | 15.457 s | not supplied | — | no row-clear claim |
| pp32768 | 395 | 82.956962 s | 566 | 57.893993 s | **25.062969 s** |

Profile attribution supplied by the caller: IU4 about 669.6 ms/chunk, GDN about 58.6 ms, FA2 about 8.2 ms at pp512, 64 remaining standalone quantizers about 9.9 ms. The rounded GDN count×average does not exactly equal 58.6 ms; archive raw totals before replacing the ledger. Long-context FA2's 27–29 s is an extrapolation, not a measured pp32768 bucket (`docs/plans/2026-09-15-halo-gap-levers-max.md:20-30`).

Do not treat mean `full_set_occ3` time as gate/up time or assert a 16/32-clock IU4 issue rate from it. A0 must size that denominator. Do not turn an analytical cache-reuse model into a guaranteed DRAM traffic reduction.

## 2. Independent audit of the two headline claims

### 2.1 F1 oracle bug: CONFIRMED; old correctness and timing verdicts withdrawn

The actual deleted file was inspected with:

```
git show 203864992:crates/hipfire-runtime/examples/tmp_fa2_n1024_oracle.rs
```

Historical evidence: its `upload_i32` at **`203864992:crates/hipfire-runtime/examples/tmp_fa2_n1024_oracle.rs:160-163`** casts the i32 slice to bytes and calls `upload_raw`; both second-half arms use `sub_offset(HALF, HALF)` at **`:260` and `:338`** (first halves at `:259,337`). The worktree implementation creates `DType::Raw` (`crates/rdna-compute/src/dispatch.rs:3588-3611`); Raw has one-byte elements (`crates/rdna-compute/src/dispatch.rs:410-458`); `sub_offset` multiplies BOTH offset and accessible length by dtype size (`crates/rdna-compute/src/dispatch.rs:251-270`).

Consequently the old second-half pointer advanced **512 bytes, not 2048**, and read original positions **[128,640)** for Q/output rows [512,1024). Its view also advertised only 512 accessible bytes for 512 i32 reads. CPU reproduction on packed i32 0..1023 returned endpoints **128,639**, versus **512,1023** for the corrected byte offset; all 512 corrected values equaled the intended row. This is a verified addressing bug, not a GPU parity result.

The FA2 entry has no N512-specialized math: `q_base=blockIdx.x*8`, the body masks `qr<batch_size`, and positions drive gmin/gmax (`kernels/src/attention_q8_0_fa2_gqa.gfx11.hip:163-206,527-548`). This supports reopening F; it does **not** prove N1024 parity. The reported max-abs pattern is consistent with the wrong causal bounds, but this static audit does not reproduce those exact floating-point errors. **Discard all old F1 timing samples**, including the purported “nearly fair −7% at 32K”: different causal work is not a valid speed comparison.

Use **Raw byte views with explicit `4*row` and `4*count`**, not a fictitious `DType::I32`. The current dtype sizing is enumerated at `crates/rdna-compute/src/dispatch.rs:410-459`; the safe recipe is in F1 below.

### 2.2 FA2 fill serialization: source premise CONFIRMED, claimed latency count UNPROVEN

The Q8 K `w<8` loop has `#pragma unroll 1`, one four-byte `memcpy`, conversion and two LDS stores (`kernels/src/attention_q8_0_fa2_gqa.gfx11.hip:279-297`). V has the same pragma and two four-byte loads per iteration before its four stores (`kernels/src/attention_q8_0_fa2_gqa.gfx11.hip:326-343`). fwht3 K has a separate `m<4` non-unrolled packed-three-byte loop (`kernels/src/attention_q8_0_fa2_gqa.gfx11.hip:227-257`).

**Refute the stronger inference that these source lines establish 27 `vmcnt(0)` waits, one outstanding load/thread, or 60–75% fill time.** LLVM can schedule loads across instructions/iterations; two V loads can overlap, scale loads need not each create a wait, and wait-counter ISA spelling varies by target. Only disassembly and matched diagnostic timings establish the generated dependency structure. Source contains independent per-code operations, not a loop-carried numeric dependence. Removing the pragma permits rather than guarantees memory-level parallelism.

**Unrolling is safe with respect to logical LDS write order:**

- K slot `(k,b)` writes `k*128 + ((16*b + 2*w + p + S(k&15)) & 127)`, `w=0..7,p=0..1`. For a fixed k, rotation is a bijection on all 128 dwords; all eight b slots are disjoint.
- V slot `(m,b)` writes `d*16 + ((m+S(d&15))&15)`, `d=32*b+4*w+c`. Each fixed d gets exactly one write for each m=0..15.
- CPU enumeration produced **4096 stores, 4096 unique addresses, range 0..4095 for each plane**. Unrolling changes no logical writer. No consumer reads until the fill barrier; no next-tile writer proceeds until turnover (`kernels/src/attention_q8_0_fa2_gqa.gfx11.hip:345,500-502`). Preserve both barriers and the initial causal mailbox protocol.
- Per-instruction lane addresses remain the same if only the w loop is expanded, so the existing bank mapping is not made aliasing or unordered. **Do not assert conflict-free producer writes**: the even swizzle's b64 conflict argument describes the compute-side fragment reads, not every producer store (`kernels/src/attention_q8_0_fa2_gqa.gfx11.hip:51-57,372-387,477-491`). Compiler vectorization may alter the bank transaction shape; inspect emitted stores and measure it.

## 3. Source ownership and frozen composer interfaces

Do not allow two composers to edit `kernels.rs`, `gemm.rs`, `attention.rs`, `norm.rs`, or `prefill.rs` concurrently. The following are **disjoint ownership sets**, not merely disjoint functions. Owners may prepare independent kernels/oracles concurrently; a single host integrator serializes shared host mutations in the order in §1. Hardware timing is serialized on the Halo. No composer runs shared validation; the main seat runs it once after the accepted stack lands.

| Composer slice | Exclusive writable files when implementing | Frozen deliverable |
|---|---|---|
| H: host/ABI integration | `crates/rdna-compute/src/{attention,gemm,norm,kernels}.rs`; `crates/hipfire-dispatch/src/families/attention.rs`; `crates/hipfire-arch-qwen35/src/qwen35/{prefill,batch}.rs`; admission documentation | C0 serializer fix, exact selectors/ABIs below, F2 cadence and capacity, integration of accepted A/B/G kernels. Only H edits these shared files. |
| A0: diagnostics | new throwaway `crates/hipfire-runtime/examples/tmp_halo_iu4_calibrate.rs` and its example-local HIP diagnostic inputs | Calibration report and byte-identical production source hash; no production edit. |
| A: IU4 | `kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip`; new throwaway `crates/hipfire-runtime/examples/tmp_halo_iu4_oracle.rs` | A5 then T0 kernels; existing public APIs unchanged; H implements listed host edits. |
| B: attention | `kernels/src/attention_q8_0_fa2_gqa.gfx11.hip`; rebuilt `crates/hipfire-runtime/examples/tmp_fa2_n1024_oracle.rs` | Corrected F1 oracle then B6; KT32/ABI/plane sizes unchanged. H owns launcher envelope. |
| G: recurrence | `kernels/src/gated_delta_net_q8_fast.hip`; `crates/rdna-compute/examples/gdn_chunk_parity.rs` | Individually gated DPP/prefetch/R8 candidates; H owns distinct module registration and ordinary-prefill caller. |
| S: overlap screen | new throwaway `crates/hipfire-runtime/examples/tmp_halo_overlap.rs` | Raw-kernel two-stream timing, no scheduler/product mutation. |

These future temporary files are explicitly authorized experiments, not permanent interfaces. Remove rejected variants immediately and throwaway harnesses after archived evidence. Keep only meaningful regressions (the positions-view bounds and blob ABI bugs qualify). H updates existing docs/changelog after admission, not speculative performance claims. Request scoped LSP references before modifying exported symbols. This audit's LSP request returned no references despite the visible same-file FA2 call, so the callsite inventory was also checked textually; do not trust an empty result from a server attached to another worktree.

**Cross-architecture invariant:** optimized bodies use compile-time exact-`__gfx1151__` guards and/or uniquely named exact-gfx1151 modules. Non-Halo preprocessing and routes retain baseline behavior. No new flags are required for A5/B6/G1/T0; compare baseline/candidate binaries or temporary oracle symbols. A5 is selected only for eager full tiles; non-full tiles and capture/replay retain their old mapping. G1's compact/decode siblings remain byte-untouched. F2 raises only gfx1151's verified envelope; other gfx11 direct/dispatch caps stay 512 and gfx12 is unchanged (`crates/rdna-compute/src/gemm.rs:19049-19070`; `crates/rdna-compute/src/attention.rs:3243-3286,3867-3896`; `crates/rdna-compute/src/kernels.rs:6314-6363`).

## 4. Metadata BEFORE execution/timing — hard resource gate

The **first executable-artifact check in every kernel unit** is code-object metadata, before correctness launches as well as before timing. Compile only that unit's code objects with the same JIT flags/source prelude as production. Archive each exact entry's `.vgpr_count`, `.sgpr_count`, `.vgpr_spill_count`/compiler spill report, `.private_segment_fixed_size` (scratch), `.group_segment_fixed_size`, kernarg offsets/size and allocated-VGPR granularity from the kernel descriptor. A missing spill field is **unknown**, not zero; resolve it using compiler resource output/disassembly before proceeding.

Use ROCm `llvm-readobj --notes <object.hsaco>` and `llvm-objdump --disassemble <object.hsaco>` on the exact object, with the production compile diagnostics. Code-object names/cache are source-hash based (`crates/rdna-compute/src/compiler.rs:78-89`). Archive `hipfire profile ... --json` as supplemental evidence only. That profiler reads the first `.kd` symbol, reports fixed group segment, and its simple occupancy formula does not incorporate launch dynamic LDS or waves/WG correctly for these claims (`crates/rdna-compute/src/profiler.rs:315-350`). **Never call its module-level occupancy number proof of two blocks/CU.** Sum fixed+dynamic LDS and calculate block/wave residency from actual launch geometry and allocated register counts. Hardware model: 2 SIMD/CU, 1536 VGPR units/SIMD, 16 waves/SIMD, 65536 B LDS/CU (`crates/rdna-compute/src/profiler.rs:67-74`). Include hardware allocation granularity and architectural block limits; arithmetic ceilings alone are necessary, not sufficient.

| Unit / entry | VGPR hard ceiling (allocated) | LDS hard ceiling / exact layout | Spill and scratch |
|---|---:|---|---|
| C0/F1/F2, unchanged FA2 | no increase vs its compiled baseline; also ≤384 for the modeled two-WG case | fixed+dynamic = 32768 B, block128 | both 0 |
| A0 register probe | ≤192, 8 waves/WG, minimum two-WG register capacity | mode a/b 0; mode c exactly 2048 B | both 0 |
| A0 no-load/no-fold twins | ≤192; record differences vs baseline (otherwise attribution confounded) | same dynamic 30720 B | both 0 |
| A5 full set/add | ≤192, no increase vs allocated baseline | dynamic 30720 B, block256 | both 0 |
| B6 both K modes | ≤384 AND retain two-WG register capacity at block128 | dynamic 32768 B, no new plane | both 0 |
| G1 DPP R4 | ≤64 (or baseline allocation if lower) | static 2048 B, block32 | both 0 |
| G1 prefetch R4 | ≤96, minimum 16-wave/SIMD register capacity | static 2048 B | both 0 |
| G1 R8 | ≤96 | static 4096 B, block32; LDS caps at 16 WGs/CU | both 0 |
| T0 z M64 | ≤192 for two blocks; **≤128** required to claim three | dynamic 19968 B, block256 | both 0 |
| T0 qkv union | ≤192 | dynamic 30720 B, block256 | both 0 |
| S1 screen | each incumbent unchanged and individually admitted | simultaneous IU4+FA2 total 63488 B/CU; calculate combined VGPR demand | both 0 |
| W2 future Sol prerequisite only | ≤300 for the **whole kernel**, not different fictitious allocations per role | exactly 61440 B, block320, 10 waves | both 0 |
| G2 future Sol prerequisite only | ≤192 at block128 (conservative) | ≤28672 B, all arrays itemized in its separate spec | both 0 |

The current IU4 source comment records 190 VGPR/zero spills, but this is not a fresh object measurement (`kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip:48-51`). A `__launch_bounds__(256,3)` spelling does not create three resident blocks when LDS requires two. Reject spills immediately: no timed “maybe it still wins” branch.

## 5. C0 — repair FA2's latent Q8 kernarg blob first

**Owner H; 1 hour; correctness-only, predicted wall delta 0.**

Confirmed mismatch: the pointer-array ABI has q,k,v,**out**,positions followed by four i32 and one f32 (`crates/rdna-compute/src/attention.rs:3931-3952`). Its blob omits out (`crates/rdna-compute/src/attention.rs:3972-3982`), although the HIP entry requires it (`kernels/src/attention_q8_0_fa2_gqa.gfx11.hip:527-537`). Generic production ingress excludes capture/recording (`crates/rdna-compute/src/attention.rs:3269-3270`); the Q8 direct launcher itself does not contain the fwht3 eager-only rejection (`crates/rdna-compute/src/attention.rs:3854-3929,4038-4043`). Thus “dormant in ordinary ingress” is precise; “unreachable anywhere” is not.

Insert `b.push_ptr(out_ptr)` between v and positions. Keep the existing blob mechanism and eager gates. For a shared, testable packing function, use a **private attention.rs helper** with the same ten typed arguments and invoke it from the existing closure; no public ABI refactor. Natural alignment is provided by `KernargBlob` (`crates/hip-bridge/src/kernarg.rs:43-47,84-107`). Correct argument payload offsets: q0,k8,v16,out24,pos32,nh40,nkv44,hd48,bs52,scale56; 60 argument bytes, with only tail padding required by the actual emitted ABI. Do not confuse compiler implicit arguments or rounded kernarg size with an extra user parameter.

Transition: caller owns immutable Q/K/V/positions and writable out → serializer records exactly those same addresses → blob launch has the same owners/writers as params launch. No tensor allocation or persistent-state transition changes.

Gate: CPU ABI regression with distinct sentinel pointer values must detect the old missing output slot and verify the consumer argument offsets. After the metadata gate, a Q8 direct params-versus-production-helper-blob smoke on identical guarded tensors must produce identical output and unchanged input/guard bytes. Do **not** test the malformed old blob on a GPU just to recreate memory corruption. No capture/replay enablement, fwht3 replay, or performance prerequisite. An ABI/parity failure blocks the campaign until corrected; the timebox triggers escalation, not leaving the known bug behind.

## 6. A0 — measure the IU4 denominator before designing a replacement

**Owner A0; one half-day total: 2 h rate/clock, 2 h attribution. Diagnostic only, wall delta 0.**

Production operation to match is `__builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(false,A,true,B,C,false)`, with the eight-WMMA integer chain and per-half f32 fold unchanged (`kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip:185-210`).

Freeze three probe entries, block `[32,8,1]`, grids 80/160/320 WGs (40 CU target), runtime loop counts 1024/4096/8192:

1. **dependent:** one live int32x8 accumulator; four WMMAs per loop iteration into that same chain;
2. **independent:** four int32x8 accumulators; one WMMA each in the same loop iteration;
3. **independent+LDS:** same four chains, two `ds_read_b64` operand loads per WMMA from a per-wave 256-byte region. Region bytes `[wave*256,(wave+1)*256)` hold two 128-byte fragment arrays, indexed by `(lane&15)*8`; lanes separated by16 intentionally broadcast. Initialize once, barrier once, then no LDS writes during timed work.

Global runtime operands must be nonzero, nibble-valid and shared identically by paired lanes; final accumulators are written to a checksum buffer so LLVM cannot erase the loop. Keep loop arithmetic bounded against signed i32 overflow, inspect instruction counts, and report both checksum validity and instruction count. Use internal `s_memrealtime` timestamps and external HIP events; convert realtime ticks using the documented target timebase, not assumed SCLK. Record sustained SCLK telemetry concurrently. If timestamp support/timebase cannot be established within the box, retain externally timed WMMAs/µs/CU and mark cycle claims unknown rather than substituting boost clock. Event APIs exist at `crates/hip-bridge/src/ffi.rs:1457-1509`.

Report total issued wave-level WMMA instructions / elapsed µs / 40 CU, dependent/independent ratio, LDS co-issue ratio, and sustained clock distribution. **Do not divide by lane count twice.** Accept a plateau only if grid160 vs320 and loop4096 vs8192 throughput differ ≤5%, with ≤3% repeated paired spread. Otherwise mark A0 inconclusive and do not select a kernel redesign from it.

Attribution twins preserve the production 128×128 layout: Y bytes `[0,9216)`, X `[9216,30720)`, X row stride168 B and Y row stride72 B; eight compute/loader waves. Production has load-X/Yh0 → barrier → dot h0 → barrier → load-Yh1 → barrier → dot h1 → barrier (`kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip:232-262`).

- **no-global-load twin:** group0 fills normally; later groups reuse initialized X/Y, retain all barriers and WMMA/fold work. Its output is intentionally invalid. For h1 reuse the existing stale Y; do not introduce an uninitialized half.
- **no-fold twin:** preserve loads, barriers and WMMA work, but replace the f32 fold with a cheap observable checksum of each integer accumulator. Inspect ISA to prove WMMAs remain; quantify checksum overhead. Register/occupancy changes make the result only a bound, not an exact subtraction.

Compare each against production at gate/up `(M,K,N)=(17408,5120,512)` and down `(5120,17408,512)`, 100 interleaved event samples after warmup. Define exposed-load bound `E=(production-no_load)/production`. W2 is not eligible unless **E≥25%** on the weighted target and the object budgets hold. A0 determines whether H1-like (~16 clock) or H2-like (~32 clock) issue behavior is supported; a value between them remains a measured value, not forced into a hypothesis. Delete diagnostics on completion; no product patch is an A0 deliverable.

## 7. A5 — column-adjacent full-tile grid

**Owners A+H under disjoint ownership; 1 h swap, at most 1 additional h for one 2×2 swizzle.**

Current full-tile selection and launch are `Gpu::gemm_mq4g256v2_mmq_prequant_iu4`, `[ceil(M/128),ceil(N/128)]`, block `[32,8]` (`crates/rdna-compute/src/gemm.rs:19039-19123`). Kernel origin is x=row,y=column (`kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip:225-226`).

For exact gfx1151, eager, full tiles only, add distinct `*_full_{set,add}_occ3_col_gfx1151` entries with the unchanged seven-argument ABI. Factor only the internal body origin into explicit `row_tile,col_tile` arguments; baseline entries supply x,y. New entries supply y,x; host grid becomes `[N/128,M/128,1]`. `FULL=false`, gfx1100, replay/capture, the quantizer and arithmetic are untouched. Keep load loops, eight wave roles, `sum[64]`, LDS offsets `[0,9216)`/`[9216,30720)`, fold order and epilogue verbatim.

State proof: read-only A/Xq → each logical block sees identical operands → identical private sums → exactly the same Y element, once; add reads only its unique destination. Block ordering introduces no global reduction or state dependency (`kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip:264-287`). Column adjacency is a locality hint, **not a guarantee of CU placement or one DRAM fetch**.

Gate: metadata first; full set/add oracle; gate/up and down separately, then weighted in-model IU4. Require **≥2% weighted IU4 reduction**, no measured down regression (paired difference >1% is a definite reject; spread ≤1% required to call it unchanged), no family >2% regression, and wall transfer. If down loses, exactly one bounded alternative: flatten full row/column tiles into 2×2 macrotiles, `macro=floor(linear/4)`, `row=2*floor(macro/(C/2))+(linear&1)`, `col=2*(macro%(C/2))+((linear>>1)&1)`, grid `[R*C,1,1]`; eligible only even R,C, otherwise baseline. No padding blocks or changed arithmetic. Same gate. Failure/repeated noise beyond the box removes the candidate.

Conditional estimates: pp512 **13–67 ms**, pp2048 **54–270 ms**, pp32768 **0.9–4.3 s**. Bank zero until wall measurement. Rebase F2/T0 against whichever mapping was actually admitted.

## 8. F1/F2 — repaired N1024 oracle, then complete host/cadence cutover

### F1: direct attention proof

**Owner B oracle, H temporary launcher envelope; 1 h repair + 2 h rerun budget. No kernel edits.**

Rebuild the historical harness, but use `upload_raw(position_bytes, &[4*N])` and `sub_offset(4*row_start,4*row_count)` for both views. Q/out F32 views still use element offsets `row_start*6144`. Assert the byte pointer delta, accessible length, and every downloaded i32 read through the actual view equals `start+row_start+b`. To prove the kernel-read contract, include a tiny copy-positions debug entry that reads `const int*` exactly as FA2 does into a host-checked echo; its budget is ≤32 VGPR, zero LDS/spill/scratch. Its negative control must detect the historical wrong view **on the host before launching an undersized view**. No reinterpretation of numeric f32 values into positions.

Freeze old-FA2/new-FA2 fixtures `(batch,start,end)`:
`(1,0,1),(7,26,33),(64,0,64),(128,384,512),(512,0,512),(512,7680,8192),(512,32256,32768)`, plus N1024 `(1024,0,1024),(1024,7168,8192),(1024,31744,32768)`. Run Q8-K and fwht3-K. At N1024 compare one full launch with two N512 calls against identical complete KV, separately cloned Q/out. Both FA2 modes must match output bits; fwht3 must also match post-launch Q bits. K/V, positions and all guards must remain unchanged. Refresh Q **before every** fwht3 timing iteration, outside the timed interval: the prologue mutates it once (`kernels/src/attention_q8_0_fa2_gqa.gfx11.hip:551-624`). For N≤512 the accepted FA2 is the bit-exact baseline; any incumbent comparison is additional evidence, not an assumption that historically different attention algorithms are bit-identical.

Use a temporary lab-only unchecked/direct helper or a temporary exact-gfx1151 direct cap1024 while production ingress remains capped512. Do not retain an unchecked public escape hatch. Metadata first, then oracle, then new interleaved direct timings. Any mismatch after fixing position validation ends F2 before product work. Report all old F1 samples invalid.

### F2: ordinary eager 1024 envelope with 512 macrosegments

**Owner H; at most 2 days implementation/state proof plus 2 h performance screen.**

All four cap sites must migrate together after F1: Q8 generic gate `crates/rdna-compute/src/attention.rs:3274-3275`; Q8 direct cap `:3889-3896`; fwht3 direct cap `:4054-4061`; **fwht3 production dispatch** `crates/hipfire-dispatch/src/families/attention.rs:1655-1671`. Freeze `max_fa2_batch=if arch=="gfx1151" {1024} else {512}` at those sites; preserve existing minimum/step/context/shape/tree/eager predicates. No N2048 admission.

Model-default ingress is currently exact gfx1151/all dense MQ4V2 (`crates/hipfire-arch-qwen35/src/qwen35/prefill.rs:500-537`). The **new 1024 default is narrower**: ordinary single-GPU sequential Q8-EF, all dense MQ4V2, no per-token requant, no recording/capture, no tree/tape/spec/independent semantics, no caller-owned scratch ceiling below1024, and no hidden-ring route in this first envelope. Resolve this call-local decision in `forward_prefill_batch_with_pbs_opts_inner` instead of globally changing an arch-only getter: the latter cannot inspect DeltaNet state or request semantics (`crates/hipfire-arch-qwen35/src/qwen35/prefill.rs:1122-1174,1394-1423,4071-4118`). Ineligible requests retain512 and their old branches; an explicit >512 developer override must not bypass the new target's state contract. Existing non-target override semantics remain untouched.

PBS construction already sizes tensors from max_batch and transactionally frees allocation failures (`crates/hipfire-arch-qwen35/src/qwen35/batch.rs:193-239,261-270`). Reuse it, no per-layer allocation or speculative tape. Verify every field including attention Q/out, gate/up, token/position capacity and IU4 sidecar `[K/128,N]`; owned scratch and effective chunk size remain bounded by `pbs.max_batch` (`crates/hipfire-arch-qwen35/src/qwen35/prefill.rs:1394-1423`). One N1024×17408 f32 intermediate is **71,303,168 B**; N2048 would be142,606,336 B but is not admitted here. No PBS ABI change is expected; H may fix only a demonstrated checked-capacity omission.

**Do not put macrosegmentation into the generic backend GDN wrapper.** Its callers include speculative replay/slot/model-dispatch paths that must not change (`crates/hipfire-arch-qwen35/src/speculative.rs:1837`; `crates/hipfire-arch-qwen35/src/forward_slots.rs:980,1293`; `crates/hipfire-dispatch/src/model_ext/qwen35.rs:174`; `crates/hipfire-dispatch/src/ops/delta_net.rs:183`). Put it at the ordinary sequential Q8 branch of `batch_chunk_delta_net_attn`, currently `prefill.rs:5419-5433`. Leave the separate MoE call at `prefill.rs:7834-7847` unchanged (its owner is `batch_chunk_delta_net_moe`, `:7252`).

Frozen macrosegment loop: repeatedly call the existing `next_prefill_chunk_len(remaining,512)`, advance a local token offset, and call the accepted sequential Q8-fast wrapper for each segment. Use F32 element offsets `offset*n_v_heads*128`, length `count*n_v_heads*128` for q,k,v,out; `offset*n_v_heads`, length `count*n_v_heads` for alpha/beta. State codes/scales/EF are **the same unsliced per-layer tensors** for both launches. Q/K have already been expanded for this branch; preserve that producer path (`crates/hipfire-arch-qwen35/src/qwen35/prefill.rs:5266-5285,5420-5432`). Existing wrapper reserves count frame IDs once per call (`crates/rdna-compute/src/norm.rs:3031-3039`). Under EF, per-layer frame values are not used by the deterministic epilogue; final process frame advancement is still the same sum (`kernels/src/gated_delta_net_q8_fast.hip:266-301`). No attempt to reorder stochastic frame ownership is permitted.

**Tail requirement:** preserve not only Q8 commit boundaries but the existing shape-dependent dispatch of partial chunks. The existing partitioner avoids singleton tails (`crates/hipfire-arch-qwen35/src/qwen35/prefill.rs:2524-2542`). Freeze a private outer selector: `a=next_prefill_chunk_len(remaining,512)`; if a=512 and `next_prefill_chunk_len(remaining-a,512)=512`, return1024; otherwise return a (do not call the second expression when remaining=a). Only two complete old512 chunks may merge. Thus513 stays511+2;1025 stays512+511+2;1537 becomes1024+511+2. Do not globally replace the old partitioner cap with1024: although nested512 splitting preserves GDN cadence, a new1023/513-token GEMM/attention launch can choose a different arithmetic route than old512/511/2 launches. CPU enumeration of every total2..32768 verified that expanding each selected1024 into512+512 reproduces the entire old partition exactly, with zero mismatches. Only a real1024 chunk invokes the two-launch GDN loop; partial old chunks retain one old invocation. Smaller externally imposed ceilings keep baseline behavior. Keep boundary regressions for513/1025/1537.

| Transition | Persistent owner / invariant | Scratch / completion |
|---|---|---|
| enter chunk | KV and each LA convolution/Q8/scales/EF belong to request; previous successful boundary intact | PBS caller-owned or call-owned, no alias escapes |
| pre-GDN projection/conv | conv advances in original token order, independently of Q8 state; future convolution does not consume GDN state | q/k/v/gate/beta for whole1024 are ready |
| first macrosegment | dequant current Q8; advance exactly first512; fold EF/requant once | out[0..512) written; second segment reads this committed Q8/scales/EF |
| second macrosegment | same operation on rows512..1024 | no F32 state crosses the midpoint seam |
| layer completion | state matches old layer at same macrosegment boundaries; KV rows have same position/data | prepared IU4 handle consumed before next reservation |
| outer completion | caller commits hidden staging only after forward success, then advances chunk_start as before | no new rollback claim on device failure |

Source for the live recurrent transition is `kernels/src/gated_delta_net_q8_fast.hip:133-145,258-317`; outer success/hidden commit ordering is `crates/hipfire-arch-qwen35/src/qwen35/prefill.rs:1473-1505`. Convolution's state writes are separate and sequential (`kernels/src/conv1d_silu_split.hip:50-78`). No claim of atomic rollback if a GPU fault occurs mid-launch.

State oracle: compare old512 vs candidate1024 over a full2048-token request, fresh/reset plus nonzero resumed prefix; snapshot each layer's Q8 codes, f32 scales, f16 EF at512/1024/1536/2048 **by layer and logical position**, not global chronological order. Compare KV bytes/positions, final convolution state, hidden/logits and final frame counter. Midpoint conv/KV snapshots for the larger chunk are diagnostic logical-prefix snapshots, not claims that the larger launch physically stops writing at512. Include tails513/1025/1537, caller PBS ceiling512, rejected tree/capture route, and prepared-generation lifetime. Demand exact bytes, not just close logits or KLD.

Promotion: ≥**3% lower wall at EACH pp2048/8192/32768**, pp512 and every decode context within2%, correct FA2 route at N1024, all metadata/state/one-chunk-md5/serve gates. If any fail, restore512 and remove the product1024 route, retaining the repaired oracle/regression evidence. Conditional estimates: 0 / **40–80 ms** / **2.5–3.5 s** for pp512/2048/32768. Those estimates may miss the deliberately stronger3% gate; they do not justify weakening it.

## 9. B6 — FA2 fill memory-level parallelism

**Owners B+H; 30 min ISA screen + 2 h attribution + at most 1 day implementation/oracle.** Start at the accepted F size (512 if F2 rejected).

First dump both K-mode objects, annotate the dynamic full-tile path's global-load issue and waits, and compare a fill-only twin (all waves fill, barrier, no QK/PV) with a compute-only twin (fill once then compute on stale planes). Prevent dead-code elimination using observable checksums; retain same launch/LDS geometry, inspect register differences. Timings are bounds because removing compute changes registers/overlap. If fill-only is <**20%** of production on `(512,7680,8192)`, kill B6 before restructuring. The brainstorm's60–75% claim is not a gate input.

Freeze **two incremental candidates, never a mixed rewrite**:

1. **B6a:** under exact `__gfx1151__`, replace only K-Q8 and V `#pragma unroll 1` with full unroll8 (and fwht3 K with unroll4). Keep outer slot loops and every conversion/store expression unchanged. Compare generated outstanding loads/waits; no scheduling change means skip to B6b, not a performance claim.
2. **B6b:** explicit register staging before conversion. KT32 Q8 mode has `uint32_t k_code[2][8]`, `uint16_t k_scale[2]`, `uint32_t v_code[2][8]`, `uint16_t v_scale[2]`: **32 code dwords + four scale scalars**, not36 four-byte global loads. For every thread issue the two K slots (`idx=tid+128*t,t=0,1`) and both V records (`m=tid>>3,b=tid&7`) with exactly the old valid-position predicates and zero initialization. Then perform K `t,w` conversions/stores followed by V `w,c` conversions/stores in the original order. Never replace an unaligned memcpy with an aligned pointer cast. fwht3 uses `cnorm[2]` and `packed[2][4]` with exact three-byte bounded memcpy, plus the same V staging; do not over-read the final100-byte record.

LDS unchanged: K bytes `[0,16384)`, V `[16384,32768)`, transient gmax/gmin at32760/32764 before first V fill. All four waves fill; waves0..2 retain their48 query rows, `Ofr[16]`, m/l and WMMA order; wave3 remains helper (`kernels/src/attention_q8_0_fa2_gqa.gfx11.hip:125-206,345-502`). Pending arrays die before QK/PV to keep the register live range bounded. No second plane, no helper-only takeover, no Q preconversion, alpha skip, grid reversal or removal of the compute memory clobbers in this unit.

Transition proof: immutable global bytes → private staged bytes (invalid records zero) → same f16 conversion and unique LDS writer → publish barrier → unchanged QK/softmax/PV → turnover barrier → next tile. Load scheduling can change, but the f32 multiplications/f16 casts, softmax and output writer do not. Compiler arithmetic changes are still possible, so the oracle is authoritative.

Gates per candidate: metadata; F1 seven+three shapes in both modes (including post-Q/guards); full-tile Q8 fill **≤6 serialized wait groups/tile** in emitted ISA, or reject the MLP-mechanism claim; then **≥20% attention symbol reduction at pp8192**, no >2% regression at pp512/2048 or other K mode, plus matrix/eval/serve. Count dynamic wait groups, not occurrences in an unrolled disassembly file. Failure at B6b ends this lever. If ISA improves but time does not, record “this change did not expose useful fill overlap”; do not claim the co-resident WG explanation as measured fact.

B6c next-tile register prefetch is **deferred**, not silently included. It needs measured remaining fill and a separate live-range/turnover specification. Conditional B6a/b estimates: **3–5 ms pp512, 50–80 ms pp2048, 11–16 s pp32768**. If B6 does not meet its large gate, bank its measured outcome as zero and resize the32K roadmap.

## 10. G1 — sequential GDN constants, no cadence change

**Owners G+H; at most 1 day total; three individually gated arms in DPP → prefetch → R8 order.**

Shipping selection is plain fast source, `[48,32,1]` ×32 for48 heads (`crates/rdna-compute/src/norm.rs:3000-3018,3064-3068`). DPP currently exists in the compact decode registrations, not the batch source (`crates/rdna-compute/src/kernels.rs:6311-6312,6360-6363`). Do not select a compact-QK kernel: its head divisor changes the input layout.

Frozen integration interface: H adds `Gpu::gated_delta_net_q8_batch_seq_gfx1151_prefill` with the **same typed argument list** as the existing batch method (`crates/rdna-compute/src/norm.rs:2980-2997`). It is called only from the ordinary dense-prefill sequential Q8-EF branch already identified in F2, and from the oracle. A private common launch helper takes a closed internal variant selector; the old public method always selects baseline. New method validates exact gfx1151, eager/not-recording, EF present, default single-end requant, H48/D128 and128..512 tokens; otherwise the caller selects old method before invocation. F2 macrosegments call the new method only when eligible. No changes to forward_slots/speculative/MoE/model-dispatch APIs or compact/masked kernels. A0/F2 do not depend on this future method.

JIT registrations each have a **distinct module AND kernel symbol**, using `HIPFIRE_GDN_KERNEL` and existing source concat convention (`kernels/src/gated_delta_net_q8_fast.hip:25-49`; `crates/rdna-compute/src/kernels.rs:6345-6363`). Freeze names `gated_delta_net_q8_prefill_dpp_gfx1151`, `..._prefetch_gfx1151`, `..._r8_gfx1151`. Use `HIPFIRE_GDN_QK_HEAD_DIV=1`, block32, no two-wave mode. Update both ensure_kernel and launch name consistently, and archive per-variant timer identity; no function-cache collision with `gated_delta_net_q8_fast`.

### G1a DPP

Define `HIPFIRE_GFX1151_GDN_DPP_REDUCE=1` only for the new prefill source. Do not alter the helper: it uses correct identity permlanex16 selectors then row_shl8,4,2,1 (`kernels/src/gated_delta_net_q8_fast.hip:51-84`). Lane0's reduction DAG is the same16,8,4,2,1 addition tree as shipping. Only lane0 is consumed for output; the kv scalar is broadcast with readfirstlane. Upper-lane differences outside the lane0 dependency cone are irrelevant. Preserve the local four-term dot and alpha/delta/update expressions (`kernels/src/gated_delta_net_q8_fast.hip:173-203`). R4 LDS is `[0,2048)`: `S_f4[r*32+lane]`, byte offset `512*r+16*lane`, holds columns lane,lane+32,lane+64,lane+96. One wave owns four rows; there is no inter-wave state sharing.

### G1b one-token prefetch

From the accepted G1a if admitted, otherwise baseline, stage exactly next token's four q, four k, four v-row values, raw gate and beta (14 scalars/lane). Prologue loads token0. Loop `t=0..n_tokens`: if next exists, issue these14 loads into a distinct pending bundle; evaluate current `expf(gate)` at the same semantic point, perform the **unchanged r=0..3** recurrence/output, then promote pending to current. No speculative out-of-range final load; no extra state tile, no movement of EF/requant into the loop. Array indices must be compile-time fixed or scalarized in ISA, not scratch. Record whether LLVM actually issues pending loads before current row arithmetic; otherwise this arm has no demonstrated prefetch mechanism.

### G1c R8

Compile existing `HIPFIRE_GDN_TILE_ROWS=8`, grid `[48,16,1]`, block32, LDS `[0,4096)` with the same `512*r+16*lane` mapping. Keep the existing r loop explicitly non-unrolled and exact per-row arithmetic (`kernels/src/gated_delta_net_q8_fast.hip:215-248`). First screen R8 without the optional prefetch. If prefetch was admitted, a combined R8 arm needs eight v-row scalars (18 pending scalars total), the same96-VGPR gate, and a separate incremental timing; do not accidentally keep a float4 load for eight rows. This is a row-decomposition experiment, not a guaranteed occupancy win: R8 halves WGs but doubles sequential row work.

All arms: persistent Q8/scales/EF → same scale*code prologue → exact sequential updates in private one-wave LDS → exactly one EF-fold/max/round/store epilogue per wrapper call → same persistent output bytes and frame reservation. The epilogue is left verbatim (`kernels/src/gated_delta_net_q8_fast.hip:258-317`). No atomics, no representation/cadence change.

Oracle: extend `gdn_chunk_parity`'s cloned Q8/scales/EF arm (`crates/rdna-compute/examples/gdn_chunk_parity.rs:122-177`) with selectable G1-only mode so it does not intentionally fail on retired C1's speed gate (`:220-234`). Deterministic normalized q/k, nonzero Q8 state and nonzero f16 EF; H48/D128, T={1,33,128,256,512}, near-one alpha (gate−0.001) and mid-decay (gate−0.1) plus varied finite gates/beta and zero-state fixture. Direct oracle may exercise short T beyond product eligibility. Compare **all output f32 bits, signed Q8 bytes, scale f32 bits and EF f16 bits**, and guards; repeat two consecutive512 segments and compare after each. Restore host frame checkpoints between arms using existing checkpoint helpers (`crates/rdna-compute/src/norm.rs:26-39`).

Each item must meet its resource cap and **≥25% incremental median reduction at T512/H48** versus the current accepted GDN path, 100 interleaved event samples, with no >2% T128/T256 regression. Otherwise drop that item; no arithmetic-tolerance fallback. Then require in-model GDN total ≥25% better, wall transfer, one-chunk md5 and serve. T512 supplied reference≈1260µs is context only; compare a fresh matched baseline. Conditional combined estimates: **30–40 ms pp512, 120–160 ms pp2048, 1.9–2.6 s pp32768**; never add independent percentages.

## 11. T0 — z half-height and qkv union, separately admitted

**Owners A+H; one day total; z first, union only after z passes and profiles still justify it.**

### T0a z M64×N128

Add a set-only exact-gfx1151 twin for **M6144,K5120,N multiple128**, including N512 and admitted1024; no generic tile policy. Same seven-argument ABI, block `[32,8,1]`. Grid row tiles96, column tilesN/128, using the admitted A5 mapping if any. Dynamic LDS: Y `[0,9216)`, X `[9216,19968)` =64 rows×42 dwords. This corrects the inherited half-height dimension confusion: X is weight rows; Y is128 activation columns (`kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip:43-53,232-236`).

Freeze wave mapping `row_group=wave/4`, `col_group=wave%4`: each wave owns32 rows×32 columns. In dot/epilogue, `i0=32*row_group`, Y base `16*col_group*18`, `j0={0,64}`, `n={0,1}`, `t={0,1,2,3}`, two WMMAs/t exactly as baseline, then the same fold. `sum` is32 f32/lane, index `((j0/64)*2+n)*8+l`; output local column `j0+16*col_group+tile_C::get_j(l)`, local row `i0+16*n+tile_C::get_i(l)`. The source mapping to preserve is `kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip:170-210,264-287`.

Loader nibble loop: `i=wave; i<64; i+=8`, each lane copies its original nibble dword. Header producer: `i=wave*16+lane/2`, execute only i<64, ksc=lane&1, same half2 conversion/replication; all waves still reach every barrier. Y loader remains256 threads striding2304 dwords. kb/h0/h1 barrier/order unchanged. Do not simply change the128 macro without changing the wave map and sum indices.

Each output retains the same K-half accumulation sequence; only its block/lane owner changes. Require bitwise z output at N128/256/512/(1024 if admitted), nonzero data, zero data, dual-half header differences and guards. Metadata then **≥15% z-shape gain**, meaningful in-model z savings beyond paired spread, no route transfer regression. Two blocks suffice for T0; if VGPR>128, explicitly report no three-block claim. A three-block redesign is T3, not free credit to T0.

### T0b q+k+v single grid

The unprepared IU4 branch currently quantizes once then issues three sets (`crates/rdna-compute/src/gemm.rs:29095-29107`); the C2-prepared branch also issues three (`:29212-29231`), consumed at `crates/hipfire-arch-qwen35/src/qwen35/prefill.rs:6041-6055`. **Both** migrate, sharing one private prequant union launcher. No weight concatenation or output repack.

New HIP entry ABI in order: `(Aq,Ak,Av,Xq,Yq,Yk,Yv,qM,kM,vM,K,N)` (seven pointers, five i32; H packs both params/blob). Eligibility exact gfx1151 eager, qM12288/kM1024/vM1024/K5120, N multiple128. Union logical row tile `r` has ranges q[0,96), k[96,104), v[104,112); select A,Y, localM and local row tile uniformly per WG. Call the unchanged M128 body with that local origin and **local output stride**, not14336. Source epilogue stride is `Y+(col+j)*M+(row+i)` (`kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip:275-283`). No WG crosses a projection boundary. Grid112×N/128, block256, LDS30720 B, existing eight-wave consumer/fold unchanged. A5 mapping applies to union tile IDs; the2×2 fallback is valid because112 and N/128 are even for N512/1024. Prepared generation is checked once and cannot be overwritten before all three projections finish.

Gate: metadata/bitwise q,k,v vs three independent sets, all guards, both prepared and unprepared paths; no additional quantizer, no stale prepared fallback. Require **≥10% combined q+k+v reduction** and **≥0.5% weighted IU4** transfer; otherwise drop union alone. z estimate≈8 ms/chunk and union≈6 ms/chunk are not guarantees; combined conditional estimate **14 ms pp512,56 ms pp2048,0.9 s pp32768**. After F2, tail math changes; gate on the accepted N, not stale N512 round counts.

## 12. S1 — timing-only two-stream co-scheduling screen

**Owner S; 3 hours maximum; no production scheduler edit.**

Use direct, prewarmed raw `HipRuntime::launch_kernel` on two explicit streams (`crates/hip-bridge/src/ffi.rs:1308-1325,1359-1384`), never profiled high-level wrappers that synchronize each launch. Create immutable sidecars/weights and disjoint output/Q/state allocations **before** timing. Gate/up IU4 `(17408,5120,512)` on A, FA2 `(512,15872,16384)` on B; repeat with Q8-fast T512/H48 on B. If F2 admitted, also screen its actual N1024 GEMM/FA2 with two512 GDN calls, preserving state cadence. Record every incumbent code-object budget first.

Freeze synchronization: timing stream records start; A and B wait on that recorded event; enqueue each workload; A/B record done; timing stream waits on both done events and records stop. Synchronize only stop, then measure start→stop. Serial control uses the same start/stop and workload counts with B waiting for A done. All event records precede corresponding waits; reset GDN state outside timing each iteration. Existing event/wait APIs provide this ordering (`crates/hip-bridge/src/ffi.rs:1476-1519`). No default-stream launches in the timed region and no shared `Gpu::active_stream` mutation across host threads.

**Important screen correction:** one long-context FA2 can greatly exceed one GEMM. Its ideal overlap ratio is `max(a,b)/(a+b)` and may already exceed0.7. Therefore report the requested one-GEMM/one-partner ratio **and** a balanced-burst screen: use the measured serial times to choose integer repeat counts making A/B total durations agree within10%, preserving separate inputs/outputs where necessary. Run20 interleaved serial/concurrent macro-iterations after warmup. Report `R=C/(A+B)` and overlap efficiency `(A+B-C)/min(A,B)`, plus each kernel's slowdown and paired spread. This avoids rejecting a physically perfect but unbalanced pair by an impossible threshold.

Kill: balanced R≥**0.90** (or no positive overlap beyond spread) for both FA2 and GDN → drop S1. R≤**0.70** for either partner, with no corruption and corroborating device timeline overlap, is a **Sol-spec trigger**, not a product win. R∈(0.70,0.90) may use the remaining3h for one repeated balanced screen; still above0.70 → defer, not “conditional ship.” Concurrency result must match serial output bits, and GDN codes/scales/EF must match from cloned starts. No model md5/matrix/serve claim is earned by this diagnostic; those are mandatory later for an S1 implementation.

### Correct ownership requirements for the future S1 spec

The max brainstorm's “only cross-chunk dependencies are GDN/KV” is incomplete. LA convolution reads/writes persistent ring state (`kernels/src/conv1d_silu_split.hip:50-78`); `Gpu` has one active stream **and one mutable scratch owner** (`crates/rdna-compute/src/dispatch.rs:664-675`); C2 generations/sidecar allocation are global to that scratch (`crates/rdna-compute/src/scratch.rs:70-117,1403-1444`). Two PBS objects alone do not isolate those buffers.

Sol's S1 spec must name two stream-local complete scratch contexts (PBS **plus** GPU temporary conversions/sidecar generation owners), request-owned per-layer conv/GDN/KV, and events with these invariants:

1. chunk(c+1),layer(l) may consume shared conv/GDN state only after chunk(c),layer(l)'s corresponding write-completion fence;
2. attention may read a causal prefix only after its required KV writes; future disjoint KV rows must not be treated as a committed host prefix;
3. layer input hidden(c,l) waits for its own previous layer, independently of cross-chunk state waits;
4. no prepared pointer or transient Q/output is reused until its stream's last consumer event;
5. host hidden-ring/cache-length/logit publication stays in logical chunk order after completion; failure drains both streams before releasing scratch and does not promise rollback of partial device writes;
6. recording/capture, speculative/tree/independent/multislot, no-EF stochastic, TP/EP and other architectures remain outside the new route.

The current outer loop does all layers of one chunk before the next and commits hidden staging after success (`crates/hipfire-arch-qwen35/src/qwen35/prefill.rs:1419-1505`). A passing pair screen only says co-scheduling may help; it does not prove the proposed dependency graph admits that overlap. Conditional future estimates: **0 pp512,150–400 ms pp2048,5–15 s pp32768**, all banked as zero until product proof.

## 13. Common verification recipe (hardware owner, hipx only)

**Do not execute these commands as part of planning.** Hardware owner runs on **hipx, `ROCR_VISIBLE_DEVICES=1`**, and confirms exact gfx1151 before any measured work. Other-device data cannot satisfy Halo gates.

Every sample archives source/binary/model/reference/prompt md5, device identity, all route flags including unset values, compiler/ROCm, JIT object hashes, warmup policy, raw samples and run order. Baseline is the accepted predecessor, not an earlier campaign stack. Use three fresh-process **ABBA** cycles (six samples/arm); report min/median/max and paired spread. Device microbench warmups are separate from matched fresh-process promotion (`docs/methodology/perf-benchmarking.md:26-38,55-96`). No speed admission if claimed gain is ≤paired spread; one extra cycle is allowed within the unit's box, otherwise inconclusive/reject.

Common environment: `ROCR_VISIBLE_DEVICES=1`, `HIPFIRE_GFX11_MQ4V2_IU4=1`, `HIPFIRE_DPM_WARMUP_SECS=10`, FA2 enabled, deterministic Q8-EF/default single-end cadence, graph/spec disabled for target-path oracle. Set MODEL to the exact `qwen3.8-27b.mq4-xt` artifact and REF to the pinned matching evaluation reference. F2 alone compares explicit512 vs1024; later units use the accepted default. Never use runtime environment reads in production HIP code.

### V0 — resources and unit oracle

Perform §4 for every candidate and diagnostic first, then the unit-specific byte/state oracle. A/T0 baseline set shapes: `(M,K)=(1024,5120),(6144,5120),(10240,5120),(12288,5120),(17408,5120)`; add `(5120,5120),(5120,6144),(5120,17408)`; N128/256/512 and1024 if admitted. Nonzero initialized add destinations, distinguishable half-group scale/zero points, signed activation patterns, guarded buffers. Include an off-envelope M/N tail on the unchanged baseline route to catch dispatch drift. F/B use §8; G uses §10. New kernels must also exercise the applicable numeric channel route per `docs/VALIDATION.md:245-260`; a green unrelated kernel test is not coverage.

### V1 — separate per-kernel attribution, microseconds before/after

For each arm, run separately at P512/2048/8192/32768:

```
cargo run --release -p saddle-lab --example profile_prefill_qwen35 --features deltanet -- "$MODEL" --prefill <P> --warmup 3 --kv-mode q8
./target/release/hipfire profile qwen3.8:27b-mq4-xt --kernel '<changed-symbol>' --json
```

Substitute the four literal P values in four invocations. The profiler resets DeltaNet and aggregates calls/total_us/avg_us (`crates/saddle-lab/examples/profile_prefill_qwen35.rs:123-177,179-239`). Archive exact symbols, calls, total and per-callµs for IU4 set/add/z/qkv, FA2 both modes when relevant, GDN and producer/quantizer counts. F2 halves many chunk launches but **not the512-macrosegment GDN count**. Changed symbol names must not hide traffic in “other.” No unobserved shape claim from a blended symbol average. Attrib profiles are not throughput evidence and cannot measure S1 overlap (`docs/methodology/perf-benchmarking.md:148-169`).

### V2 — production matrix, interleaved A/B

One fresh invocation per sample:

```
./target/release/hipfire bench qwen3.8:27b-mq4-xt --matrix --pp 512,2048,8192,32768 --ctx 128,2048,32768 --runs 1 --warmups 10 --kv-mode q8
```

Apply ABBA to whole invocations, not best rows selected from different processes. No target prompt/decode-context regression >2% unless a unit states a stricter condition. Record all rows, including pp8192. Row clear requires measured wall≤§1 target, not sum of kernel estimates. The supplied halogen denominators are HTTP measurements: also retain like-for-like production done-field prompt runs for any halogen headline; do not equate internal-profile wall with HTTP wall (`docs/methodology/perf-benchmarking.md:110-118,153-159`).

### V3 — eval_hipfire one-chunk md5 for every bit-exact product unit

```
cargo run --release -p hipfire-runtime --example eval_hipfire --features deltanet -- --model "$MODEL" --ref "$REF" --output <arm>.kldseq --kv-mode q8 --kv-v q8 --scoring-mode prefill --max-chunks 1
md5sum <baseline>.kldseq <candidate>.kldseq
```

Arguments are implemented at `crates/hipfire-runtime/examples/eval_hipfire.rs:62-153`. Eval forces graph/prompt-normalization off and may reuse PBS (`:156-185`); record those effective overrides. Compare identical output digests, with identical reference and model, after every C0/A5/F2/B6/G1/T0 admission and later S1. **This is an additional model-level check, not proof that all logits or hidden states are identical**: the file stores per-sequence KLD aggregates (`:5-11,34-35`). Direct byte/state oracles remain mandatory. `--max-chunks 1` is one reference-evaluation chunk, not an assertion of one512-token model chunk. Eval trailing tok/s is never speed evidence (`docs/methodology/perf-benchmarking.md:118`).

### V4 — serve and retained-path isolation

Per `docs/VALIDATION.md:120-141,253-261`, run actual serve harness, reasoning off, exact registry/model identity:

```
python3 scripts/serve_harness.py --model "$MODEL" --tag qwen3.8:27b-mq4-xt --kv q8 --mtp off --dflash off --mode battery --sampling registry --thinking off --max-tokens 512 --max-seq 32768 --seed 0 --show-config
python3 scripts/serve_harness.py --model "$MODEL" --tag qwen3.8:27b-mq4-xt --kv q8 --mtp off --dflash off --mode battery --sampling registry --thinking off --max-tokens 512 --max-seq 32768 --seed 0 --out <arm>-battery.json
```

Require5/5 coherent, normal completions, zero empty/runaway/attractor failures, decode within2%; reviewer reads decoded text. Compare deterministic baseline/candidate text bytes for bit-exact units. For F2 and future S1 add the same command with `--mode chain` (state/prefix-cache surface). A one-request CLI generation is insufficient. Final accepted stack runs `scripts/redline_daemon_harness.py` using the kernel route in `docs/VALIDATION.md:133-135` to prove existing capture/HIP/PM4 paths remain isolated; no new retained-route performance claim. Main owns required shared no-GPU checks and maintainer review (`docs/VALIDATION.md:40-59`), not each composer.

## 14. Predicted ledger and the W2-versus-S1 decision

All savings below are **conditional unmeasured ranges versus the relevant accepted predecessor**, not additive credits. Zero is banked before gates. Recompute remaining bucket totals after F2/B6/G1/T0.

| Ordered unit | pp512 | pp2048 | pp32768 | Decisive gate |
|---|---:|---:|---:|---|
| C0 | 0 | 0 | 0 | correct ABI, no performance prerequisite |
| A0 | 0 | 0 | 0 | stable rate/clock; E bound |
| A5 | 13–67 ms | 54–270 ms | 0.9–4.3 s | ≥2% weighted IU4, down unchanged |
| F1 repair | 0 | 0 | 0 | actual positions and1024-vs-two512 bits |
| F2 | 0 | 40–80 ms | 2.5–3.5 s | ≥3% wall at2K/8K/32K, exact cadence/state |
| B6 | 3–5 ms | 50–80 ms | 11–16 s | ≥20% long-context attention |
| G1 | 30–40 ms | 120–160 ms | 1.9–2.6 s | each arm≥25% incremental GDN |
| T0 | about14 ms | about56 ms | about0.9 s | z≥15%; union≥10% and weighted≥0.5% |
| S1 screen | 0 | 0 | 0 | balanced overlap ratio≤0.70 to request spec |
| Future W2, H1-like | 200–300 ms | 0.8–1.2 s | 13–19 s | A0 then new Sol spec; not authorized code |
| Future W2, H2-like | 100–135 ms | 0.4–0.54 s | 6.4–8.6 s | same, with lower measured ceiling |
| Future S1 product | 0 | 150–400 ms | 5–15 s | separate state-machine spec and product gates |

pp512 needs27.5ms: G1 or a measured A5 can plausibly clear it. pp2048 needs634ms: composer-tier middle estimates do not establish a clear. pp32768 needs25.06s: the outcome is highly sensitive to B6; do not add pre-B6 S1 ceilings to post-B6 attention savings.

**Decision after S1 screen, using measured residuals:**

1. If all rows clear, stop. No redesign for an already closed gap.
2. W2 eligible only with valid A0, exposed-load boundE≥25%, room for a ≥15% measured weighted-IU4 improvement to matter, and no evidence its proposed issue rate exceeds A0's measured rate. Otherwise W2 is abandoned without a kernel.
3. S1 eligible only after balancedR≤0.70 and real overlap evidence; calculate a conservative residual-wall opportunity using **post-B6/G1** partner times and the state fences above. A pair benchmark alone never awards wall savings.
4. If only one eligible, write **that lever's own Sol spec**. If both, choose the one whose conservative measured-bucket opportunity closes the larger remaining deficit; ties prefer S1's unchanged math but only if scratch/state isolation has a concrete design. If neither closes a row at an optimistic measured-bucket bound, report residuals and stop; do not invent another lever.

W2's separate spec starts from8 compute waves+2 loader waves and61,440B: Y0 `[0,9216)`,Y1 `[9216,18432)`,X0 `[18432,39936)`,X1 `[39936,61440)`. Current single-plane arithmetic/source is `kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip:123-210,232-262`. Compute reads X(kb&1),Yh0 while loader stages next-X and Yh1 into freed planes; barrier publishes/read-completes; compute h1 while loader stages next-Yh0; barrier; swap group. Sol must prove warmup/final-group ownership, loader staging/register cap for the **whole320-thread kernel**, and every-wave barrier participation before coding. Day1 object gate≤300 VGPR,0 spill/scratch,61,440LDS; dominant gate/up ≥15% gain over accepted baseline in100 interleaved samples; same bit oracle and ≥15% weighted in-model IU4 then V2–V4. Abandon at any gate, maximum3–5days. This is **not** permission to retry A1's in-place staging.

## 15. Deferred/rejected scope and numerical escape hatch

- **Deferred B6c/B3/alpha-skip/LPT:** next-tile staging needs a measured remaining fill denominator and explicit pending-byte lifetime; Q preconversion needs measured ≥10% Q-conversion cost. Do not bundle any with B6. `x*1==x` is not an unrestricted IEEE bit-identity proof for NaNs/signed-zero paths; any future skip must state its finite-domain contract and oracle. Compute conversion/rescale sites are `kernels/src/attention_q8_0_fa2_gqa.gfx11.hip:389-402,460-464`.
- **G2 chunked GDN:** separate Sol spec only after G1 residual sizing. The current chunked source is a correctness vehicle with one WG/head, host chunk index and scalar global rereads (`kernels/src/gated_delta_net_f32_chunked.hip:12-24,108-154,177-205,226-272`). A new multi-WG on-device-loop algebra changes summation order even if Q8 commit cadence stays512. It stays **default-OFF behind documented `HIPFIRE_GFX1151_GDN_CHUNKED`**, config-owned and exact ordinary gfx1151 Q8-EF only; no silent fallback or default flip in this plan. First-day F32 prototype gate <1e-4 output/state max-abs vs sequential, T512/H48≤700µs, metadata≤192VGPR/28KiBLDS/zero spills+scratch; otherwise kill. Next Q8 stage must meet prior C1.1 gates: no NaN/Inf/guards; relative-infinity output and dequant-state relative-L2 each≤shipping-Q8 error+5e-4; T512 output≤8e-3 near-one and≤5.5e-3 mid-decay; CS16/32 discrepancy≤5e-4; one EF commit per512 macrosegment, same frame total (`docs/plans/2026-09-15-halo-prefill-plan.md:508-516`). WT2 KLD≤matched baseline+0.0005 plus V4, and≥25% **over accepted G1**, not obsolete1260µs. The separate spec must define how CS16 is supported before using that gate. No equality claim from the algebra identity alone (`kernels/src/gated_delta_net_f32_chunked.hip:26-37`).
- **S3 gate/up+SiLU/FWHT fusion:** deferred behind W2 verdict; would touch weight layout and requires identical SiLU/butterfly/quant order. Current producer and shared exact72-byte recipe already exist (`kernels/src/fused_silu_mul_mq_rotate.hip:135-215`; `kernels/src/fused_silu_mul_mq_rotate_awq.hip:66-147`; `kernels/src/fused_rmsnorm_mq_rotate.hip:169-245`; `kernels/src/block_i4_128_quant.hip:21-88`). Do not count C2's already-removed loads again. F16 producer sibling is not a new Halo lever (`kernels/src/fused_silu_mul_mq_rotate_f16.gfx1100.hip:8-26,106-115`).
- **T3 three-block M64:** deferred; needs≤128 allocatedVGPR and measured weighted gain, not merely19,968B LDS. T0's targeted z twin makes no such claim.
- **Rejected/closed:** A1 retries/deeper in-place staging; T2 multi-chain consumer at the current two-block budget; N256 column tile / M256 row tile; T4/T5 residency trades without new measurements; double-LDS at two blocks; standalone header trim; IU8 register-resident/MMQ round2 retries; occ2/occ3 replay of prior tuning; s_setprio; NPU; gfx11 B16 arms; KT64, split-KV and f16-accumulator FA2; prefill graph capture. Production pins KT32 (`crates/rdna-compute/src/attention.rs:3918-3927`); gfx12's distinct partial/WMMA mapping is not a drop-in gfx11 optimization (`kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:14-42,74-108`). C0 does not reopen capture.
- **Lower-bit prefill weight copy:** dropped pending contrary measured byte attribution; IU4 already consumes raw weight nibbles with int4 activations (`kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip:5-25,137-157`). This plan does not establish halogen's internal format or quantify its quality from repository comments; no external-server change is authorized.
- **Conv/remaining quant/KV fusions:** deferred small buckets, not row-closing promises. Convolution preserves load placement because compiler FMA decisions matter (`kernels/src/conv1d_silu_split.hip:44-60`); do not hoist it “for free.”

## 16. Deliverable, cleanup and review veto

**Completed planning checks (CPU only):** all explicit worktree path/range citations resolved within their files (historical oracle checked separately); corrected Raw position views returned all512 intended i32 values; FA2 plane enumeration found4096 distinct writes per plane; pair-only F2 outer partition expanded to the baseline partition for every total2..32768; T0's proposed mapping covered all8192 elements of64×128 exactly once using sum slots0..31/lane; the A5 fallback was bijective for tile grids8×4,48×8,96×4,112×8,136×4. LDS byte totals recomputed to30,720/19,968/32,768/61,440 for baseline IU4/T0/FA2/W2. These checks support the specified indexing and ownership, not emitted-code numerical parity or throughput.

A composer returns: exact source/object identities; metadata before first timing; direct oracle/state bytes/guards; raw per-kernelµs before/after; complete interleaved matrix; one-chunk eval md5; decoded serve battery/chain as applicable; measured delta against the accepted predecessor; and explicit accepted/rejected/inconclusive verdict against the predeclared threshold. Diagnostics return their actual report, not model-validation claims. A reviewer may veto even a nominal speed win for unmatched work, hidden resource regression, incomplete route coverage or state ownership.

After proof, remove invalid diagnostic kernels, unused candidate arms and throwaway scaffolding; update existing docs/changelog and the measured ledger through H. Do not keep rejected environment switches, unchecked launchers, stale cap comments or duplicate quant recipes. The current production state/API is not widened to unrelated architectures, decode, retained replay, tree/independent/multislot, TP/EP, weight/KV formats or speculative verify. Main performs one shared no-GPU validation pass after integration; independent **Sol `@reviewer` may review this Astra-authored plan and owns the final technical veto**. Source audit/CPU enumeration are not GPU correctness or performance results (`docs/VALIDATION.md:24-38`).
