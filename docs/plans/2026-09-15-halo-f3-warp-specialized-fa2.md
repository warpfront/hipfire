# F3: gfx1151 producer/consumer, ping-pong FA2 prefill

Date: 2026-09-15. Plan of record for composition, **not an admission or a measured kernel result**. Worktree: `/home/kaden/ClaudeCode/warpfront/wt-lloyd`, branch `mq4-lloyd`. This planning assignment changed only this document. No GPU program, build, formatter, linter, or project-wide test was run. CPU address/resource/epoch calculations were executed; their limited evidence is in §8. Main owns admission; an independent reviewer owns the final technical veto.

> **STATUS 2026-09-15 — SHELVED by F3.0 attribution (parent-measured, gfx1151,
> batch 512, H24/KV4/D256, 100 interleaved samples, twins within ±10% VGPR of
> production's 200, no spill).** Fractions of shipping FA2 time at
> L1024 / L8192 / L32768: fill 21.3 / 23.2 / 27.4 %; `__expf` 0.0 / −0.2 / −0.7 %;
> PV 8.4 / 8.1 / 6.6 %; **Q reload (global f32 + f16 cvt per 16-key subtile)
> 38.2 / 39.3 / 37.1 %**; QK WMMA 13.3 / 14.4 / 12.8 %.
> The ping-pong half of this design overlaps exp with WMMA, and exp is already
> free on RDNA3.5 — nothing to buy. The producer/consumer half has a fill
> ceiling of 21–27 % but moves fill from four waves to two producers, so it
> cannot approach the ≥15 % admission bar. The dominant lever is Q reload,
> which is a loop restructure inside the shipping kernel (F4: share each Q
> fragment across both 16-key subtiles; pre-rounded f16 Q), bit-exact and
> portable to every gfx11. This document is retained as the audited anatomy
> (§2) and register model (§5); its units are not dispatched.

## 1. Decision, scope, and conflicting premises

**Build one exact-gfx1151 candidate: two producer waves, three consumer waves partitioned into groups of one and two, block160, 16-key stages, a depth-two f16 K/V LDS ring, and a barrier-based two-phase consumer ping-pong.** Keep the existing kernel as the shipping control and fallback. First land a serial-consumer skeleton with the same ownership, buffers, ABI, and per-row arithmetic; only then introduce overlap. This is FA3-inspired scheduling, not a port of NVIDIA TMA, WGMMA, warp-group barriers, or dynamic register repartitioning.

Three distinctions are mandatory:

1. **The named shipping kernel is H24/KV4/D256, not H48/KV8/D128.** Both HIP entries reject other shapes; both host launchers and the repaired oracle agree (`/home/kaden/ClaudeCode/warpfront/wt-lloyd/kernels/src/attention_q8_0_fa2_gqa.gfx11.hip:527-548,600-624`; `/home/kaden/ClaudeCode/warpfront/wt-lloyd/crates/rdna-compute/src/attention.rs:3880-3887,4045-4052`; `/home/kaden/ClaudeCode/warpfront/wt-lloyd/crates/hipfire-runtime/examples/tmp_fa2_n1024_oracle.rs:37-45`). The supplied 6.9–7.0 ms/call at pp8192 and its H48/KV8/D128 shape annotation remain **caller-reported context**, not a matched measurement for this source. Do not silently substitute D128's smaller accumulator/LDS budget. This spec implements the source-compatible D256 replacement and explicitly sizes D128 portability in §5; admitting a new H48/D128 route requires identification of its real shipping comparator and a separate shape admission. Record the model's actual FA-layer launch shape before attributing a wall win.
2. **N512/N1024 means launch-wide query-token batch, not hundreds of rows in one workgroup.** A workgroup still owns eight positions × six heads. The current direct launchers permit N1024 only on exact gfx1151, whereas ordinary Q8 and fwht3 ingress remains capped512 in this snapshot (`/home/kaden/ClaudeCode/warpfront/wt-lloyd/crates/rdna-compute/src/attention.rs:3889-3897,4054-4061,3264-3281`; `/home/kaden/ClaudeCode/warpfront/wt-lloyd/crates/hipfire-dispatch/src/families/attention.rs:1655-1685`). Supplied F1 results are accepted facts: bit-exact N1024, and one1024 beating two512 by 6–8% at L8192/L32768. F2 is live. F3 supports both N values in its oracle and consumes whichever production envelope F2 actually admits; it does not implement F2's recurrent-state/cadence changes or count F2's win again.
3. Use the assignment's **32-CU, wave32, 64-KiB LDS/CU** target when sizing grids. Do not inherit the predecessor's 40-CU/80-slot denominators. Confirm the physical device identity/CU count when hardware execution is eventually authorized; record discrepancies rather than forcing the numbers. Unified memory does not make raw cache loads or host/GPU ownership fences free.

Explicit non-goals: changed KV formats/scales, additional cache writes, f16 O accumulation, split-KV/merge, Q-preconversion sidecars, speculative or retained replay, graph-capture enablement, tree/window/multislot attention, new model shapes, multi-GPU scheduling, changed prefill chunk/state cadence, grid-order tuning, and simultaneous B6 edits. Freeze the accepted predecessor (including any admitted B6/F2 changes) before comparison.

## 2. Audited shipping anatomy

Every statement in this section describes source, not measured ISA timing.

| Item | Current implementation and citation |
|---|---|
| Workgroup/waves | Block128, four wave32s. Waves0–2 compute; wave3 helps. **All four**, not just wave3, perform K/V fill. Each compute wave owns16 packed query/head rows: `head_local=2*wave+(ml>>3)`, `q_local=ml&7`; six heads share each KV fill. Grid `[ceil(N/8),4,1]` (`/home/kaden/ClaudeCode/warpfront/wt-lloyd/kernels/src/attention_q8_0_fa2_gqa.gfx11.hip:125-146,163-170,214-228,259-265,300-305,348-350`; `/home/kaden/ClaudeCode/warpfront/wt-lloyd/crates/rdna-compute/src/attention.rs:3937-3960,3973-3983,4110-4137,4150-4171`). |
| LDS | `KT=32` in production. K starts at dword0; V at `KT*128`; each plane has `KT*256*2=16384` bytes. Total32768 bytes, dynamic. Source also supports KT64, but host pins32. Initial gmax/gmin occupy the last two dwords, later overwritten by V (`/home/kaden/ClaudeCode/warpfront/wt-lloyd/kernels/src/attention_q8_0_fa2_gqa.gfx11.hip:77-90,125-146,173-206`; `/home/kaden/ClaudeCode/warpfront/wt-lloyd/crates/rdna-compute/src/attention.rs:3926-3935,3973-3977,4101-4108,4150-4154`). |
| Swizzle | `S(c)=2*bit_reverse4(c)`; K dword `(key,p)` at `key*128+((p+S(key&15))&127)`; V transpose dword `(dim,key_pair)` at `dim*(KT/2)+((key_pair+S(dim&15))&(KT/2-1))`. WMMA fragments use four aligned b64 loads. This is the existing read-side layout, not proof that all producer stores have zero bank conflicts (`/home/kaden/ClaudeCode/warpfront/wt-lloyd/kernels/src/attention_q8_0_fa2_gqa.gfx11.hip:102-108,276-296,335-341,367-387,470-491`). |
| Private state | Per compute lane: `Ofr[16]` of float8 =128 f32 values, m and l scalars; a float8 score fragment per16 keys. Q is reloaded as f32 float4s and rounded to f16 per QK dimension chunk rather than retained across the key loop. P becomes half16 after pair reconstruction; WMMA accumulates f32 (`/home/kaden/ClaudeCode/warpfront/wt-lloyd/kernels/src/attention_q8_0_fa2_gqa.gfx11.hip:148-170,360-407,438-458,493-494`). |
| Synchronization | One initial causal-mailbox workgroup barrier; exactly **two workgroup barriers per KT tile**: after cooperative fill and after all PV reads. Thus Q8 body count is `1+2*ceil((gmax+1)/32)` for a nonempty valid workgroup. fwht3 adds one prologue barrier. No barrier exists between the two16-key subtiles or between their QK/softmax/PV phases (`/home/kaden/ClaudeCode/warpfront/wt-lloyd/kernels/src/attention_q8_0_fa2_gqa.gfx11.hip:201-212,345-350,498-503,576-597`). |
| WMMA → exp → WMMA | For each16-key subtile:16 QK WMMAs across D256; mask/row-max and xor16 pair max; alpha `__expf`; eight per-lane probability `__expf` expressions; pair sum, m/l update, full-P pair exchange; rescale128 O values;16 PV WMMAs. The same wave executes these phases serially. Different waves may naturally desynchronize, but source has no intentional cross-group phase pipeline (`/home/kaden/ClaudeCode/warpfront/wt-lloyd/kernels/src/attention_q8_0_fa2_gqa.gfx11.hip:350-407,410-497`). |
| MUFU terminology | Source calls `__expf` for alpha and probabilities; alpha's initial-infinity branch and all-masked rows can avoid exp. Actual lowering, special-function issue throughput, and overlap with WMMA need the exact gfx1151 object. Do not infer NVIDIA MUFU behavior, one instruction per source expression, or a fixed exp/WMMA cycle ratio from source (`/home/kaden/ClaudeCode/warpfront/wt-lloyd/kernels/src/attention_q8_0_fa2_gqa.gfx11.hip:428-449`). |
| Q8 dequant | The format is **a scale per32-code block**, not one scale for an entire D256 head row:34 bytes = f16 scale +32 signed i8 codes; eight blocks/head; four heads give1088 bytes/position. K fill multiplies each code by its block scale in f32 and casts to f16. V fill pairs adjacent keys and transposes into LDS. Unaligned code loads use bounded memcpy. K's eight-code-dword loop and V's paired loop have `unroll 1`; this alone does not count machine wait instructions (`/home/kaden/ClaudeCode/warpfront/wt-lloyd/kernels/src/attention_q8_0_fa2_gqa.gfx11.hip:140-142,259-298,300-343`). |
| fwht3 K | K instead uses100 bytes/head: f32 cnorm +96 bytes of packed3-bit codes,400 bytes/position. Decode `cnorm*TURBO_C3_256[code]`, cast f16, and write the same K layout; V remains Q8. K is already rotated: the attention fill does not rotate it again. Before the body, all four waves transform disjoint Q rows in place using `fwht_shfl_forward_256`,48 rows in12 four-wave steps, followed by one barrier. Rotation is not replay-idempotent (`/home/kaden/ClaudeCode/warpfront/wt-lloyd/kernels/src/attention_q8_0_fa2_gqa.gfx11.hip:215-257,551-624`; `/home/kaden/ClaudeCode/warpfront/wt-lloyd/crates/rdna-compute/src/attention.rs:4038-4043`). |
| Causal/tails | `positions[]`, not the profile-only host max-context value, defines gmin/gmax and `seq_len=gmax+1`. Fill zeroes out-of-sequence records. A16-key subtile is full if its final key≤gmin; otherwise each lane pair masks against its own row's position. Missing Q rows are not loaded/stored; an entirely masked row keeps zero probability and normalizes to zero. All waves still participate in barriers (`/home/kaden/ClaudeCode/warpfront/wt-lloyd/kernels/src/attention_q8_0_fa2_gqa.gfx11.hip:163-206,232-238,267-274,308-324,350-358,415-449,510-522`; `/home/kaden/ClaudeCode/warpfront/wt-lloyd/crates/rdna-compute/src/attention.rs:3850-3852,4006-4008`). |
| Output ownership | Lane pair ml owns one query/head row; half0 stores even dimensions, half1 odd, `dc*16+2*j+half`. Each valid element is written once; only fwht3 Q has an input-side mutation (`/home/kaden/ClaudeCode/warpfront/wt-lloyd/kernels/src/attention_q8_0_fa2_gqa.gfx11.hip:505-524,565-597`). |

### 2.1 ABI and exact callsite inventory

Preserve these **flat argument lists and the pointer-array form**; do not replace them with one pointer to a new argument struct or pass synchronization flags/scratch in a new user argument.

| ABI | Flat order and user-payload offsets |
|---|---|
| Q8 | q0, k8, v16, **out24**, positions32; nh40, nkv44, hd48, batch52, scale56. Five pointers + four i32 + one f32 =60 payload bytes. |
| fwht3 | q0, k8, v16, **out24**, positions32, signs1 40, signs2 48; nh56, nkv60, hd64, batch68, scale72. Seven pointers + four i32 + one f32 =76 payload bytes. |

HIP declarations, params vectors and blob packing agree on that order (`/home/kaden/ClaudeCode/warpfront/wt-lloyd/kernels/src/attention_q8_0_fa2_gqa.gfx11.hip:527-537,600-612`; `/home/kaden/ClaudeCode/warpfront/wt-lloyd/crates/rdna-compute/src/attention.rs:3939-3960,4124-4137,4157-4170,17138-17166`). Payload bytes are not a claim about compiler-rounded kernarg segment size or implicit arguments. **C0 is already fixed in this worktree**: `pack_attention_q8_0_fa2_gqa_gfx11_kernarg` includes out and is used by the blob closure; do not reintroduce the older plan's missing-output description as current state (`/home/kaden/ClaudeCode/warpfront/wt-lloyd/crates/rdna-compute/src/attention.rs:3979-3983,17143-17166,17181-17234`).

Exact integration sites:

- `Gpu::attention_q8_0_flash_prefill_wmma` calls `Gpu::attention_q8_0_fa2_gqa_gfx11` after shape/eager checks (`/home/kaden/ClaudeCode/warpfront/wt-lloyd/crates/rdna-compute/src/attention.rs:3225-3237,3264-3281`).
- The `AttnFlashAsym3FwhtBatchedMasked` dispatch's gfx11 arm calls `Gpu::attention_q8_0_fa2_gqa_fwht3k_gfx11`, with no tree bias and Q8 V (`/home/kaden/ClaudeCode/warpfront/wt-lloyd/crates/hipfire-dispatch/src/families/attention.rs:1650-1685`).
- Direct launchers are `Gpu::attention_q8_0_fa2_gqa_gfx11` and `Gpu::attention_q8_0_fa2_gqa_fwht3k_gfx11`; source registration is `ATTENTION_Q8_0_FA2_GQA_GFX11_SRC` and `ATTENTION_Q8_0_FA2_GQA_FWHT3K_GFX11_SRC`. The latter concatenates the K-mode define, turbo common prelude, and kernel (`/home/kaden/ClaudeCode/warpfront/wt-lloyd/crates/rdna-compute/src/attention.rs:3854-3989,4010-4177`; `/home/kaden/ClaudeCode/warpfront/wt-lloyd/crates/rdna-compute/src/kernels.rs:5681-5696`). Registration comments saying “never on a default path” are stale relative to the default-on ingress above; use predicates, not those comments, as route evidence.
- Oracle `run_q8` and `run_fwht3` call those direct launchers and synchronize; `fixture_n1024_q8`/`fixture_n1024_fwht3` currently compare one1024 with two512, not shipping with F3 (`/home/kaden/ClaudeCode/warpfront/wt-lloyd/crates/hipfire-runtime/examples/tmp_fa2_n1024_oracle.rs:451-483,635-725,727-834`).

LSP reference queries for the two absolute launcher locations returned empty despite the visible calls. This was reported as a worktree/server mismatch; this inventory was grounded by direct source search/read. Before implementation, H repeats references with a server attached to this worktree and preserves additional callers it discovers.

## 3. Frozen F3 geometry and ownership

### 3.1 Wave roles and six-head reuse

New source: `/home/kaden/ClaudeCode/warpfront/wt-lloyd/kernels/src/attention_q8_0_fa2_gqa_ws.gfx1151.hip`. Exact `__gfx1151__` compilation guard. Block `[160,1,1]` = five wave32s, with a `__launch_bounds__(160,2)` declaration as a compiler constraint, **not occupancy evidence**.

| Wave | Role | Private ownership |
|---|---|---|
| 0 | Consumer group A | Packed16 rows, heads local0/1 × eight positions; one O/m/l owner per lane pair. |
| 1,2 | Consumer group B | Each packed16 rows: heads local2/3 and4/5 respectively. Two independent per-wave states, not a shared accumulator. |
| 3 | K producer | All Q8 or fwht3 K bytes, dequantization, K-plane stores, and one next-stage raw bundle. No O or softmax state. |
| 4 | V producer | All Q8 V bytes, dequantization/transposition, V-plane stores, and one next-stage raw bundle. No O or softmax state. |

Keep the exact output map `q_row=8*blockIdx.x+(ml&7)`, `head=6*blockIdx.y+2*consumer_wave+(ml>>3)`, and parity ownership of dimensions. Grid `[ceil(N/8),4,1]`: N512 gives256 WGs, N1024 gives512. At the requested32 CUs and two resident blocks/CU these are4 and8 nominal scheduling rounds, before causal skew. Each16-key K/V fill serves all six heads and all eight positions once; **do not split GQA into head-specific workgroups or independently refill K/V for the two groups**.

The 1+2 consumer grouping is deliberate: two balanced three-wave groups plus two producers would be block256, eight waves, and cannot claim two blocks/CU at256 VGPR under the resource model below. Four consumers introduce padded WMMA rows or a second live row tile. Keep all48 useful rows with three consumers; unequal phase work is a measured performance risk, not hidden padding.

### 3.2 Global and persistent state

Before launch: the request owns cache bytes and positions, Q and out are live caller-owned buffers, and upstream writes have completed under the existing stream order. F3 allocates no global scratch and changes no request cache length, GDN/EF state, convolution state, or host transaction boundary.

- Q8 mode: Q/K/V/positions remain byte-identical; only out is written.
- fwht3 mode: K/V/positions/sign tables remain byte-identical; exactly this call's Q rows are rotated once, in place, before any consumer loads them. Five waves visit logical row `r=wave+5*step` with **r<48** and `q_base+r/6<N`. Preserve the signed FWHT-256 helper and floating-point operation order; then every wave reaches one workgroup barrier. The r<48 guard is essential with five waves. Post-Q must match shipping bitwise even though F3 attention outputs may differ numerically.
- No role returns early for an invalid individual query row. All five waves traverse the same workgroup-uniform stage count `T=ceil((gmax+1)/16)` and barrier sequence. Invalid rows only mask arithmetic/stores. Unsupported shape is rejected by the host or a uniform entry guard before any barrier.
- Success: every valid out element has one writer and normalized finite output for finite bounded fixture inputs. Failed launches propagate errors. Never retry shipping after a launched fwht3 candidate has mutated Q; fallback selection happens **before** launch.

## 4. LDS ring, state machine, and softmax overlap

### 4.1 Exact layout: depth two, KT16

All offsets below are bytes; no static LDS allocation is permitted in addition to this dynamic region.

| Range | Contents | Bytes |
|---|---|---:|
| [0,8192) | slot0 K:16 keys ×256 dims ×2 | 8192 |
| [8192,16384) | slot0 V-transpose:256 dims ×16 keys ×2 | 8192 |
| [16384,24576) | slot1 K | 8192 |
| [24576,32768) | slot1 V-transpose | 8192 |
| total | two slots × two planes ×16×256×2 | **32768** |

K dword address within its plane remains `k*128+((p+S(k&15))&127)`. V becomes `d*8+((m+S(d&15))&7)`, m=0..7. Each half16 V fragment is still four aligned b64s; rotation wraps within eight dwords. This is a new physical V stride, not permission to use the KT32 mask or stride by accident.

Use slot1's final two dwords, byte32760/32764, as **initial-only gmax/gmin mailboxes**. Wave0 publishes; all waves barrier/read into registers. First fill touches only slot0; its publication barrier guarantees all mailbox reads have retired before any slot1 fill. Mailboxes have no meaning afterward and are not asynchronously polled flags. No persistent flag, P tile, Q tile, m/l array, or O array occupies LDS.

Producer mapping is explicit:

- Wave3 lane l handles K block slots `s=l+32*i`, i=0..3; `k=s>>3`, `b=s&7`. Q8 stages four scale values and four × eight code dwords. fwht3 stages four cnorm values and four × four3-byte packed loads, each packed word covering8 decoded dimensions. Keep all loads bounded to their100-byte records; no four-byte over-read at a3-byte tail.
- Wave4 lane l handles V pair-block slots `s=l+32*i`, i=0..1; `m=s>>3`, `b=s&7`. Two slots × two adjacent key records stage four scales and32 code dwords. Convert to `(V[2m,d],V[2m+1,d])` f16 pairs.
- `gk>=seq_len` never loads global memory; its staged values are zero. Mask actual attention probabilities using each query position; zero-filled K/V is not a substitute for a causal score mask.

### 4.2 Skeleton protocol: correctness before overlap

The first candidate uses the above five-wave mapping and two slots, but consumers run the ordinary QK → online-softmax → PV sequence together. For t=0..T−1: producers fill slot(t&1); **all160 threads** execute a publication `__syncthreads`; all three consumers consume t in increasing key order; all160 execute a retirement `__syncthreads`. Only then may slot(t&1) be recycled. No prefetch across retirement, no consumer phase skew, no spin waits. This has an elementary fill/publish/read/retire invariant; use it to prove dequant/layout, guards, row ownership, fwht3 prologue and ABI independently of overlap.

### 4.3 Selected ping-pong protocol: two full barriers per16-key stage

Stages are16 keys; `slot(t)=t&1`. Each consumer keeps its own row state; the groups **never combine or exchange m/l/O**. A score fragment belongs to one wave and one epoch. `soft(t)` applies masking, max, exp, m/l update, and P reconstruction; `pv(t)` rescales O by alpha then performs PV. The next QK does not update online state.

Prologue after bounds/Q rotation:

1. Producers load/dequant/store stage0 to slot0; publication barrier P0.
2. Group A computes QK(0). Group B waits at the next barrier. Producers load raw stage1 to private registers if it exists, without writing slot1. Startup barrier S0.

Steady state, for every t=0..T−1, including tails:

| Phase | Group A (wave0) | Group B (waves1,2) | Producers (waves3,4) | Full-workgroup fence |
|---|---|---|---|---|
| Y(t) | soft(t), retaining P(t), alpha(t), O through t−1 | QK(t), retaining S(t), state through t−1 | dequant/store raw(t+1) into slot(t+1), if it exists | **publish(t+1)**: next slot complete; all current QK/soft work finished |
| X(t) | PV(t); then QK(t+1) if it exists, reusing dead P/score temporaries | soft(t); then PV(t) | load raw(t+2) into private registers, if it exists; **no LDS stores** | **retire(t)**: both groups' last K/V reads of slot(t) finished |

At the next Y, slot(t+2)=slot(t) is free because the preceding X barrier retired it. The raw(t+2) bundle is already private. The final X omits future QK/prefetch, drains both groups' PV, and retires the last slot before the epilogue. No partial final-stage consumer skips a fence.

**Where overlap is intended:** Y pairs A's exp/softmax with B's QK WMMA plus producer conversion/stores. X pairs B's exp/softmax with A's PV/next-QK WMMA plus producer global loads. Source staging creates opportunities; it does not prove simultaneous issue, fair lane placement, or a faster instruction mix. Groups are not switched between query owners: their **math phases** ping-pong, avoiding a second O accumulator or inter-group m/l transfer.

### 4.4 State invariants and deadlock argument

| Boundary | Required before | Guaranteed after |
|---|---|---|
| entry → bounds | no consumer observes uninitialized LDS; Q transformation, if any, owns disjoint rows | all waves hold identical gmin/gmax/T; mailbox can later die |
| fill0 → P0 | slot0 is exclusively producer-owned | K0/V0 fully visible, immutable until retire0 |
| startup → S0 | slot0 published | A owns S0; producers own raw1; B state still empty |
| Y(t) → publish | slot(t) readable; A S(t) ready; slot(t+1) free; producer raw(t+1) valid | A owns P/alpha(t), B owns S(t), next K/V readable; no writer touched current slot |
| X(t) → retire | both current planes and next plane immutable | A/B O,m,l incorporate exactly keys through stage t in order; A S(t+1) ready if any; producers own raw(t+2); current slot free |
| final retire → output | no unread stage or outstanding LDS reader | only unique out stores remain; buffers may be released only at ordinary stream completion |

Publication/retirement use HIP workgroup barriers with memory ordering; inspect generated LDS/global wait completion around them. A raw-load prefetch must complete before its following conversion; no compiler-hoisted store may cross retirement. Do not substitute naked `s_barrier` inline assembly without equivalent compiler and memory-order semantics.

All five waves execute finite, uniform phases and the same ordered barrier instances. There is no “wait for a wave that is blocked on my flag,” no inter-workgroup dependency, and no oversubscribed producer workgroup that must become resident to release a consumer. Hardware barrier waits allow runnable waves in the resident workgroup to advance. Thus the selected protocol has **no software circular wait** and needs only ordinary forward progress already required by a well-formed workgroup barrier—not an undocumented fairness guarantee for a spin loop. This is a control-flow proof, not measured hardware execution.

Cost is explicit: ping-pong body has two barriers per16 keys, or **four per32 keys**, versus shipping's two per32 keys (shipping sites: `/home/kaden/ClaudeCode/warpfront/wt-lloyd/kernels/src/attention_q8_0_fa2_gqa.gfx11.hip:345,502`). Q8 F3 total is one bounds + P0 + S0 +2T =3+2T; fwht3 adds its one rotation barrier. Removing fill exposure must pay for the extra synchronization; otherwise reject F3, not the correctness fences.

### 4.5 Why not LDS flags plus s_sleep, or depth three?

A conventional flag alternative would allocate **64 control bytes per slot**: `ready_k`, `ready_v`, `ack_wave0`, `ack_wave1`, `ack_wave2`, plus padding. Epochs are full32-bit stage numbers, not one-bit ready toggles. Initially every word is UINT_MAX. Producer ownership for t<R is free; for t≥R, both producers acquire-load all three acks for exactly t−R before writing their own plane. After all lanes finish plane stores, that producer publishes its ready epoch with workgroup-scope release semantics. Each consumer acquire-observes both ready epochs t, consumes K/V, drains its LDS reads, then release-publishes its own ack t. Invalid query lanes/waves still acknowledge. No cross-wave “group leader” may acknowledge on another wave's behalf without a real completion mechanism.

Polling must be wave-uniform, use real acquire operations rather than a compiler-cached ordinary/volatile load, broadcast the elected lane's result, and only then use bounded `s_sleep` backoff. `s_sleep` is neither a memory fence nor a proof that a waiting producer/consumer gets scheduled. Even with all five waves resident, absence of a dependency cycle is not a guarantee against unfair polling starvation. A separately instrumented watchdog stress and a target memory-order/ISA audit would be prerequisites; a timeout invalidates the attempt, not permission to return partial output.

**This flag variant is not admitted for source-compatible D256 F3:** depth-two payload32768 +128 controls =**32896 B**, so two blocks need65792>65536. Depth-three payload49152 +192 =**49344 B**, also only one block. KT32/depth2 needs65536 B before controls and also violates the two-block requirement. Overlaying live data with polled flags is not a legitimate fix; epochs can be overwritten before the other wave has acquired them. Choose the barrier protocol with **zero permanent flag words**. Do not quietly relax occupancy because flags look more FA3-like.

## 5. Register arithmetic, portability, and metadata-first rejection

### 5.1 Whole-kernel VGPR ceiling

These are **design reservations per lane, not compiler measurements**. Wave roles do not receive different hardware register allocations: one code object allocates the whole-kernel peak for every wave. The role columns describe mutually exclusive live ranges, not quantities to add across different waves.

| Live category | Consumer A/B VGPR reservation | Producer K/V reservation |
|---|---:|---:|
| O:16 float8 accumulators | 128 | 0 |
| one16-key S fragment | 8 | 0 |
| probability scratch `earr[8]` | 8 | 0 |
| m/l, alpha, max/sum and state temporaries | 8 | 0 |
| simultaneously live half16 operands (K/Q or V/P) | 16 | 0 |
| conversion/load temporaries | 8 | 8 |
| addresses, row/position predicates | 24 | 16 |
| raw Q8 code bundle | 0 | 32 |
| four scales/cnorm | 0 | 4 |
| epoch/predicate temporaries | 0 | 4 |
| compiler/allocation headroom | 56 | 32 |
| **sum** | **256** | **96** |

fwht3 K's raw bundle is16 packed dwords rather than32; reserve the same96 ceiling. Its Q rotation occurs before O initialization and must not be inlined into an overlapping O live range. P(t) dies after A's PV before A's QK(t+1), so there is no persistent double score/P tile. Q is not retained D-wide. Stage temporary arrays must scalarize; register-indexed scratch is a rejection, not an implementation of this table.

Use the repository's stated gfx1151 resource model: two SIMD/CU,1536 wave32 VGPR units/SIMD,16 waves/SIMD,65536 LDS bytes/CU (`/home/kaden/ClaudeCode/warpfront/wt-lloyd/crates/rdna-compute/src/profiler.rs:67-74`). At allocated256, floor(1536/256)=6 waves/SIMD. Conservatively charging three waves/SIMD per five-wave block, two blocks need6 waves/SIMD; total LDS is2×32768=65536. This supplies necessary register/LDS capacity for **at least two blocks/CU**, even without assuming perfect five-wave packing. Confirm target allocation granularity, SGPR and workgroup limits, static LDS and actual occupancy before making that claim about an object. **≤256 alone is insufficient** if any other limit breaks two-block residency.

Metadata is the first check after compiling an exact candidate, **before any correctness launch or timing**: exact entry's allocated VGPR (including rounding), SGPR, spills, private scratch bytes, static+dynamic LDS, block/wave count, wave32, kernarg offsets and rounded segment size. Required: allocated VGPR≤256, zero spills, zero scratch, total LDS32768, and verified two-block capacity. Missing spill information is unknown, not zero. Use per-entry code-object notes/compiler resource reports and disassembly, not only profile JSON: the profiler returns the first `.kd` entry and its simple occupancy calculation omits this launch's dynamic-LDS/block arithmetic (`/home/kaden/ClaudeCode/warpfront/wt-lloyd/crates/rdna-compute/src/profiler.rs:315-350`). One bounded live-range correction is allowed; never time a spilling variant hoping it wins.

### 5.2 Item-by-item portability

| Item | Exact gfx1151 action | gfx1100 / gfx1200 treatment |
|---|---|---|
| Selection and resource target | New source/module/entry, only exact gfx1151;32-CU grid accounting and measured RDNA3.5 issue behavior | Leave current dispatch unchanged. gfx1100 uses the shipping gfx11 kernel where already eligible; gfx1200 keeps its existing route, **not** a gfx11 kernel. The existing gfx11 allowlist and separate gfx1201 branch are visible at `/home/kaden/ClaudeCode/warpfront/wt-lloyd/crates/rdna-compute/src/attention.rs:3243-3286`. |
| WMMA fragments | Preserve gfx11 duplicated full half16 row and xor16 lane-pair semantics | Algorithm potentially portable to gfx1100 after independent resource/numeric/timing gates. gfx12 operand/lane mapping must be re-derived; do not compile this body there. Existing gfx11 mapping is documented and implemented at `/home/kaden/ClaudeCode/warpfront/wt-lloyd/kernels/src/attention_q8_0_fa2_gqa.gfx11.hip:23-49,360-404,438-458,466-494`. |
| Producer dequant, causal state, barrier protocol | Exact byte transforms and ownership above | Conceptually portable algorithms, not an authorization to change other targets. Byte rounding, barrier lowering and scratch allocation still need target proof. |
| LDS swizzle, waves, exp overlap | KT16/depth2,5-wave roles, selected barrier schedule | Target performance choices only. No assumed bank/issue-rate/fairness equivalence across gfx1100/gfx1200. |
| Flag polling | Rejected for D256 two-block design | Not a portability fallback. Needs a distinct memory-order/progress proof on each ISA. |
| H48/KV8/D128 | Arithmetic feasibility only: O=8×8=64 f32/lane; consumer reservation192 using the same other allowances; grid `[ceil(N/8),8,1]` | Requires its actual existing cache/rotation ABI and comparator. Never reinterpret D256 fwht3's100-byte records or256-entry sign tables as D128. No new shape route in F3. |

D128 LDS arithmetic, **not a selected build**: KT16/depth2 payload16384 +128 controls =16512; depth3 payload24576 +192 =24768, both allow two blocks by LDS. KT32/depth2 payload32768 leaves no permanent flag room, just like the selected D256/KT16 case. Thus a flag/depth3 design only looks feasible if one silently adopts the unverified D128 premise. That is specifically forbidden here.

## 6. Numeric/state oracle, timing oracle, and admission

### 6.1 Numeric and state gates

F3 is explicitly exempt from the campaign's bit-exact **attention-output** requirement; softmax accumulation order may change. No exemption applies to pointer bounds, causal semantics, output ownership, immutable input bytes, fwht3 post-Q, or unchanged routes.

1. **Serial skeleton:** preserve the shipping per-row16-key math sequence and require exact output/post-Q bits in both K modes on the repaired F1 fixtures. This is a structural debugging gate before permitting any numerical reordering. A layout/protocol bug must not be disguised by a tolerance. The shipping per-row subtile sequence to preserve is `/home/kaden/ClaudeCode/warpfront/wt-lloyd/kernels/src/attention_q8_0_fa2_gqa.gfx11.hip:350-498`.
2. **Ping-pong:** compare every output against shipping and serial controls; report bit equality, max absolute/relative error with a documented near-zero denominator, RMS error, and finite/masked-row counts. Initially retain per-row arithmetic order so unexplained drift receives investigation; F3 need not be bit-exact for final admission. Keep exact K/V/positions/sign-table/guard bytes, Q8 Q bytes, fwht3 post-Q bits, and unique-writer coverage. Constant-V/zero-Q analytic fixtures check real values, not just nonempty output.
3. Include F1's seven N≤512 and three N1024 fixtures; add N8/9 and stage-ending contexts15/16/17/31/32/33, nonmonotonic valid positions within one8-position WG, repeated positions, first/last partial Q group, and a fully masked lane only in a fixture whose positions contract explicitly allows it. Guard both ends of Q/out/cache/positions. Hold the requested max context within cache allocation; never intentionally launch an undersized historical Raw position view.
4. **Model numeric gate:** on the exact pinned one-reference-chunk prefill fixture, run shipping and candidate from identical fresh model/request state, with identical model/reference/token hashes, KV mode and V=Q8, effective PBS N, routing, compiler and other flags. Use `eval_hipfire --model <absolute-model> --ref <absolute-reference> --output <absolute-arm.kldseq> --kv-mode q8 --kv-v q8 --scoring-mode prefill --max-chunks 1`, then repeat with `--kv-mode fwht3`. Exercise the candidate route, not a silent fallback. These modes/arguments exist at `/home/kaden/ClaudeCode/warpfront/wt-lloyd/crates/hipfire-runtime/examples/eval_hipfire.rs:95-137`.
5. Archive **md5 comparison even when it differs**. Decode the actual HFKSEQ output rather than its rounded console summary: current v2 writes a20-byte header then fp64 mean KLD, p99 KLD, mean NLL per reference chunk (`/home/kaden/ClaudeCode/warpfront/wt-lloyd/crates/hipfire-runtime/examples/eval_hipfire.rs:1438-1459`). Require the absolute difference in that fixture's **mean KLD versus the same pinned reference** between F3 and shipping to be **≤1e-4**, with finite valid metrics. Report p99/NLL differences and both full-precision values to Main. This is not “KLD between the two output distributions” and not a per-token logit bound. `--max-chunks 1` limits reference chunks, not necessarily one N512/N1024 attention launch (`/home/kaden/ClaudeCode/warpfront/wt-lloyd/crates/hipfire-runtime/examples/eval_hipfire.rs:352-359`).
6. An md5 mismatch alone does not reject F3; md5 equality alone does not prove state parity. Passing ≤1e-4 is a **necessary numeric gate**; Main decides admission with the independent review, not the composer. Do not auto-promote, widen1e-4, or repin a fixture to hide a failure. The pinned fixture/model/reference identities must come from the campaign artifacts; absent identities block numeric admission, not source work.

F3 does not directly write recurrent state. A numerically different attention output can still affect downstream model state, so do not promise whole-model byte equality. Parent-owned serve battery/chain and request reset/resume checks remain required for admission; graph/retained paths must demonstrably remain on their previous route.

### 6.2 Extend, do not replace, the repaired timing oracle

Owner O extends `/home/kaden/ClaudeCode/warpfront/wt-lloyd/crates/hipfire-runtime/examples/tmp_fa2_n1024_oracle.rs`, retaining the Raw-position view checks and existing one1024-vs-two512 comparison. That current oracle's wrappers synchronize, and its timing uses `Instant` plus those synchronizations with5/10 iterations (`/home/kaden/ClaudeCode/warpfront/wt-lloyd/crates/hipfire-runtime/examples/tmp_fa2_n1024_oracle.rs:451-483,699-723,801-832`). Those timings are useful historical end-to-end evidence, not the new isolated kernel gate.

Add a closed oracle-local `F3Arm { Shipping, Serial, PingPong }` selector. For **each N in {512,1024}, L in {1024,8192,32768}, K mode in {Q8,fwht3}**, positions are `L-N .. L-1`; both arms see the same complete KV prefix and same Q. Compare one shipping N launch against one F3 N launch. Keep the F1 split-launch result separate; never compare F3 one1024 against shipping two512 and attribute the full gain to F3.

Precompile/prewarm all arms. Use at least100 matched ABBA-ordered pairs per cell, one stream, HIP events around just the kernel dispatch, no device synchronize within the event interval. Refactor only oracle launch plumbing or use its raw launch helper; include the in-kernel fwht3 Q rotation in the measured kernel. Restore a pristine preallocated Q buffer **before every arm** in fwht3 mode and synchronize restoration outside the timed interval. No measured allocation/upload, no re-rotation of already consumed Q, no profiler timer inside the event interval. Report host wrapper time separately if desired.

Archive raw times/order, entry/module/object hashes, dimensions, event timing method, clocks/thermal state, residency metadata and numeric status. Serialize all hardware timing on Halo, keep ordinary warmup/CPU contention policy identical for unified memory, and use the campaign's fresh-process ABBA wall protocol as a second gate. Treat warm-cache isolated timing as isolated evidence only.

### 6.3 Admission/kill thresholds and timebox

For each admitted production N and **both** K modes:

- At L8192, require `median(T_F3)/median(T_shipping) ≤ 0.85`—at least15% lower time, a deliberately explicit interpretation of “≥15% faster.” The upper95% paired-confidence bound must also clear0.85; archive the paired estimator/resampling method.
- At L1024, **not slower**: ratio≤1.00 with the upper95% paired bound≤1.00. “Within noise” is inconclusive, not a pass. One extra matched session is allowed within the box.
- L32768 must not regress; report its actual reduction rather than extrapolating L8192. Require output/state/numeric gates for every cell. No K-mode averaging may hide a loss. If N1024 is still lab-only, report it but do not widen production to it under F3.
- In-model pp2048/8192/32768 must show the candidate's actual dispatch count and lower matched FA2 time, and the wall benefit must survive measurement spread. No pp512/decode regression >2%. The supplied 6.9–7.0 ms observation is context, not a substitute for this matched baseline. No claim of closing395→566 t/s without like-for-like production wall evidence.

**Hard timebox: three engineering days /24 working hours of composition plus serialized scoped hardware gates**, apportioned in the final unit list. One bounded resource/live-range repair and one bounded phase/bank adjustment are permitted inside—not after—the box. Reject immediately on spill/scratch, >256 allocated VGPR, loss of two-block capacity, protocol hang, state corruption, or KLD failure. If the final interleaved gate fails or remains inconclusive at expiry, abandon F3 and remove the unadmitted product route. A serial skeleton win is a separately named producer-only result requiring its own admission, not permission to call F3 complete. No open-ended switch to KT8, one-block occupancy, depth3, new shape, or custom unfair spin protocol.

## 7. Risks and expected benefit—not results

**Single biggest risk: phase separation may not produce useful exp/WMMA overlap on RDNA3.5, while the mandatory barrier rate doubles per key.** Compiler scheduling, unbalanced1+2 consumer groups, wave-to-SIMD placement and shared issue resources can erase the intended gain. The ≥15% matched gate, rather than an architectural analogy to FA3, decides.

| Risk | Mitigation / evidence required |
|---|---|
| RDNA wave-scheduler fairness | Selected protocol has no polling and no inter-WG dependency. Flag/s_sleep designs remain rejected; bounded sleeps do not prove fairness. Inspect phase traces on a diagnostic object and ordinary uninstrumented timing separately. |
| New LDS bank layout | CPU b64 bank-start enumeration covers the proposed consumer addresses, not producer store transactions or real bank-service time. Inspect emitted b64 instructions/alignment and measure bank-conflict/stall counters if available. One bounded layout adjustment must preserve32-KiB total and rerun address/numeric gates. |
| Causal-tail load imbalance | Bounds use gmax; rows with lower positions still participate in every phase. No divergent early barrier exit or per-wave key-loop truncation. Measure short contexts, partial16-key stages, irregular positions and end-of-grid tails; no full-tile-only claim. |
| RDNA3.5 special-function rate | Count actual exp lowering and dependent instruction chains; measure rather than importing NVIDIA MUFU rates. A successful latency-hiding schedule cannot increase a saturated shared throughput ceiling. |
| Producers become bottleneck | Two producers perform all conversion work; Q8 and fwht3 have different loads. Stage one bounded raw bundle and overlap its lifetime only with consumer work. Record Y/X phase imbalance; no extra producer waves without new resource proof. |
| Compiler live-range inflation | The256 reservation is not metadata. No simultaneously live next-Q tile, doubled O, or consumer raw-dequant arrays. Reject spills before launching. |
| Capture/ABI and fwht3 ownership | Preserve pointer-array arguments and C0 packer, exact offsets, stable buffer lifetimes; candidate eager-only. Retained/capture stay on existing routes. Do not retry a mutated fwht3 Q buffer after an execution error. |
| Kernel win fails in model | Same accepted F2/B6 baseline, both K modes, actual dispatch proof, serialized hardware and matched wall ABBA. Do not bank isolated-cache gains or double-count N1024. |

**Projected conditional range [unmeasured]:15–30% lower long-context FA2 time (1.18–1.43× kernel throughput), with zero banked until gates.** A regression is also plausible. Under the predecessor's explicitly extrapolated27–29 s FA2 share of an82.96 s pp32768 request, that would save roughly4.1–8.7 s, produce roughly416–441 t/s, and recover only part of the395→566 gap. At the predecessor's roughly0.13 s pp2048 attention estimate, the same fraction saves only20–39 ms (about0.6–1.1% of the3.52 s wall), not the entire gap. Those denominators are estimates, not fresh profiles (`/home/kaden/ClaudeCode/warpfront/wt-lloyd/docs/plans/2026-09-15-halo-gap-levers-max.md:15-30`). Recompute after F2/B6 and after resolving the supplied shape annotation; do not add these values to their earlier projections.

## 8. Planning evidence and frozen implementation interfaces

CPU-only calculations run for this document established:

- K and V each enumerate2048 distinct dword addresses covering exactly0..2047 at KT16/D256; two slots total32768 bytes.
- The three-consumer row/parity mapping writes12288 distinct output elements per eight-position, one-KV-head workgroup =8×6×256.
- For each b64 fragment step i∈{0,2,4,6}, the16 unique ml rows have16 distinct even bank starts for K and V under the proposed swizzle; duplicate lane halves intentionally read identical addresses. This does not measure producer conflicts or hardware throughput.
- A slot-epoch model with all six abstract Y-phase role orderings passed stage counts1,2,3,5,64,512,2048. It verified current-slot preservation, retirement before reuse, next-stage publication and complete final state. It is not a GPU memory-model proof.
- Consumer reservation sums256; producer96; the stated resource model gives6 waves/SIMD at256, enough for two conservatively3-wave/SIMD blocks; LDS limits residency to two.

Frozen host/kernel contract:

- New production entry/module names: `attention_q8_0_fa2_gqa_ws_gfx1151` and `attention_q8_0_fa2_gqa_fwht3k_ws_gfx1151`. Same typed HIP argument lists as their respective shipping entries, no new user arguments. Distinct oracle-only serial entries/modules: `attention_q8_0_fa2_gqa_ws_serial_gfx1151` and `attention_q8_0_fa2_gqa_fwht3k_ws_serial_gfx1151`.
- New source constants in `/home/kaden/ClaudeCode/warpfront/wt-lloyd/crates/rdna-compute/src/kernels.rs`: `ATTENTION_Q8_0_FA2_GQA_WS_GFX1151_SRC` and `ATTENTION_Q8_0_FA2_GQA_FWHT3K_WS_GFX1151_SRC`, using the established concat/prelude convention. Compile-time `HIPFIRE_FA2_WS_PIPELINE=0|1` and `HIPFIRE_FA2_WS_ENTRY=<exact-symbol>` distinguish serial/ping-pong without function-cache collisions; K mode remains compile-time0|3. No runtime environment read in HIP.
- New Rust methods `Gpu::attention_q8_0_fa2_gqa_ws_gfx1151` and `Gpu::attention_q8_0_fa2_gqa_fwht3k_ws_gfx1151` have the same typed argument lists as the respective shipping Rust methods. They validate exact arch, shape, eager ownership, context/capacity and accepted N envelope; use `[ceil(N/8),4,1]`, `[160,1,1]`, dynamic32768. Q8 reuses the existing private kernarg packer. fwht3 preserves its existing inline pack order. Serial is oracle-only raw launch using the same argument values, not a new public unchecked API.
- Shipping methods and entries remain forced-shipping controls; do not hide F3 selection inside them. Only after admission, the two production ingress callsites identified in §2.1 select the new methods for eligible exact-gfx1151 requests. Others keep their previous branches. A build-time comparison can keep shipping selected before admission; do not add a permanent experimental environment-flag family solely for the oracle.
- H exclusively owns all shared Rust integration files; K exclusively owns the new kernel; O exclusively owns the existing throwaway oracle. K does not edit the shipping kernel, H does not change F2 cadence, and O does not change production routing. Interfaces above allow preparation concurrently; shared-file mutation and hardware execution remain serialized. Every composer skips project-wide tests, formatters and linters; Main runs shared validation once after accepted integration.

## 9. Ordered bounded composer units

1. **F3.1 — producer/consumer skeleton, no softmax overlap (K;8 h maximum).** Own only `/home/kaden/ClaudeCode/warpfront/wt-lloyd/kernels/src/attention_q8_0_fa2_gqa_ws.gfx1151.hip`. Add exact serial entries, `fa2_ws_body`, `fa2_ws_load_k_raw`, `fa2_ws_load_v_raw`, `fa2_ws_store_k`, `fa2_ws_store_v`, `fa2_ws_qk`, `fa2_ws_softmax`, `fa2_ws_pv`, and `fa2_ws_rotate_q_256`; these are private implementation symbols, not extra kernarg fields. First make the five-wave fill/publish/read/retire protocol observable with an oracle-only exact-value fill/read checksum; then compose the unchanged per-row QK/softmax/PV math and the five-wave Q rotation. Both modes must exist; no stub alternate entry. Acceptance: source-compatible H24/KV4/D256 contract explicitly recorded; CPU writer/slot enumeration; per-entry metadata first with≤256 VGPR/zero spills/scratch/32768 LDS/two-block capacity; checksum passes for ring wraps and final stages; then exact serial output and post-Q bits on repaired F1 fixtures. H/O may prepare units2/3 concurrently from the frozen interfaces, but no overlap optimization proceeds before this gate.
2. **F3.2 — isolated host/ABI wiring (H;2 h maximum).** Own only `/home/kaden/ClaudeCode/warpfront/wt-lloyd/crates/rdna-compute/src/kernels.rs` and `/home/kaden/ClaudeCode/warpfront/wt-lloyd/crates/rdna-compute/src/attention.rs`. Add the two source constants/new Rust methods and distinct module names from §8; reuse `pack_attention_q8_0_fa2_gqa_gfx11_kernarg`, preserve fwht3 params/blob order, and leave both shipping methods and production ingress selected. Acceptance: checked callsite inventory; Q8 offsets0..56 and fwht3 offsets0..72 match emitted metadata/params/blob with distinct sentinel addresses; direct candidate rejects non-gfx1151 and capture/replay before Q mutation; no source/ABI behavior change for shipping paths. After metadata, scoped Q8 params-versus-blob direct smoke must match on guarded buffers; this is not capture enablement. No speculative launcher or global selector abstraction.
3. **F3.3 — differential and timing oracle extension (O;4 h maximum).** Own only `/home/kaden/ClaudeCode/warpfront/wt-lloyd/crates/hipfire-runtime/examples/tmp_fa2_n1024_oracle.rs`. Extend `run_q8`, `run_fwht3`, `fixture_n1024_q8`, `fixture_n1024_fwht3`, the existing N≤512 fixtures and `main` using `F3Arm`; keep corrected `upload_positions`, `pos_view`, `assert_pos_view`, and the historical negative control. Add the analytic/checksum and boundary cases, input/output guards, fresh fwht3 Q restoration, and non-synchronizing event-timed arm launches. Acceptance: shipping/serial fixture outputs and post-Q parity, exact immutable inputs, no unchecked Raw view, complete12-cell N×L×K-mode event matrix, and F1 split-launch results separately labeled. Existing oracle log label “F3” for fwht3 at `/home/kaden/ClaudeCode/warpfront/wt-lloyd/crates/hipfire-runtime/examples/tmp_fa2_n1024_oracle.rs:798,829` must become unambiguous K-mode/arm labeling, not evidence that this optimization already existed.
4. **F3.4 — consumer ping-pong only (K;4 h maximum; depends on1–3 correctness).** Own the new kernel only; implement the exact P0/S0/Y/X/drain schedule in `fa2_ws_body` and production entry names. Do not change dequant representation, output map, ABI, cache access predicates, or add persistent LDS flags. Acceptance: metadata gate rechecked on both final entries, phase/epoch instrumentation verifies publication/retirement with arbitrary valid tails, no hang under scoped repeated launch stress, all oracle state/guard contracts, all output errors reported, and parent numeric gate≤1e-4 with md5 comparison. Remove phase instrumentation from the measured object. If overlap requires another live O tile, extra permanent LDS words, or unfair flag polling, reject rather than widening this unit.
5. **F3.5 — claim-scoped performance decision (O, serialized Halo hardware owner;3 h maximum; depends on4).** Own the oracle and its external result artifacts only; no production selection. Run the full interleaved matrix and matched in-model attention attribution at the accepted N, with both K modes and device/source identity. Acceptance: numeric gates first; L8192 ratio≤0.85 and L1024≤1.00 with the stated paired bounds, no L32768 regression, no architecture/shape substitution, and all raw results archived. One bounded scheduling/bank correction may be assigned to K within the total24 h, with metadata/numeric gates repeated. Failure or inconclusive timebox expiration abandons F3; projected speedups are not evidence.
6. **F3.6 — admitted routing, fallback/isolation, and final veto (H + Main/reviewer;3 h composition maximum; depends on5 and explicit parent admission).** H owns `/home/kaden/ClaudeCode/warpfront/wt-lloyd/crates/rdna-compute/src/attention.rs`, `/home/kaden/ClaudeCode/warpfront/wt-lloyd/crates/hipfire-dispatch/src/families/attention.rs`, the source registrations, and updates to this plan plus the existing `/home/kaden/ClaudeCode/warpfront/wt-lloyd/CHANGELOG.md`. Change only the Q8 ingress at `Gpu::attention_q8_0_flash_prefill_wmma` and the fwht3 dispatch arm identified in §2.1 to call the new methods on admitted exact-gfx1151 shapes/N/eager requests; preserve shipping fallback and all non-gfx1151/capture/tree predicates. Acceptance: matched pp/decode matrix and parent serve battery/chain, actual route proof, immutable ABI and pointer lifetimes, gfx1100/gfx1200 isolation, no retry-after-mutation, and independent reviewer veto resolved. Main runs shared no-GPU validation once. Only after evidence is archived remove oracle-only serial/checksum/trace scaffolding and throwaway files no longer needed by F1/F2; retain genuine ABI/position/protocol regressions, never delete a sibling-owned active oracle. On rejection remove the unadmitted candidate wiring instead and leave shipping selected; document the measured reason, not an unproven bottleneck explanation.
