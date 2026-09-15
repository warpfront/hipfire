# W2: exact-gfx1151 IU4 prefill GEMM redesign

Date: 2026-09-15. Worktree: `/home/kaden/ClaudeCode/warpfront/wt-lloyd`, branch `mq4-lloyd`. **Plan of record, not kernel admission.** This assignment changes only this document. No source edits, builds, GPU runs, formatters, linters, or project suites were performed. Main owns all Halo execution and integration gates; an independent reviewer owns the final technical veto.

## 1. Decision and prerequisite corrections

**Choose register-prefetched, half-slab turnover with four compute/loader waves and two independent output accumulators per wave.** Retain the 128×128 output tile, original row/column grid walk, 30,720-byte LDS layout, W4A4 representation, and per-output arithmetic order. Target **two resident workgroups/CU**, not merely a launch-bounds spelling. The new block is `[32,4,1]`; every wave loads and computes. There are no dedicated producer waves, second LDS plane, asynchronous barrier protocol, or persistent/global scratch buffers.

This deliberately resolves two conflicts in the predecessor plan:

- The earlier W2 starting point was block320, eight compute plus two loader waves, and 61,440 bytes LDS **per workgroup**. That fits only one workgroup/CU, so it cannot meet this assignment's occupancy≥2 requirement. Its ≤300 whole-kernel ceiling remains a ceiling here, but its block/plane geometry is superseded, not silently described as two-block capable (`/home/kaden/ClaudeCode/warpfront/wt-lloyd/docs/plans/2026-09-15-halo-gap-plan.md:84-100,388`; `/home/kaden/ClaudeCode/warpfront/wt-lloyd/docs/plans/2026-09-15-halo-gap-levers-max.md:184-243`). The new design uses **61,440 bytes for two resident workgroups together**.
- The current `_occ3` entries use `__launch_bounds__(256,3)`, but the documented 190 VGPR and 30,720-byte LDS do **not** establish three resident workgroups. Two require eight waves/SIMD and allocated VGPR≤192; three already fail LDS. Increasing this eight-wave kernel to 250–300 VGPR would lose the second workgroup. Changing to four waves is therefore necessary for the selected budget; this is not an A1 retry that adds pending registers to the unchanged eight-wave body (`/home/kaden/ClaudeCode/warpfront/wt-lloyd/kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip:43-51,232-236,291-297`; `/home/kaden/ClaudeCode/warpfront/wt-lloyd/docs/plans/2026-09-15-halo-gap-plan.md:82,100`).

**A0's load-exposure admission is accepted as parent-measured:** E=25.8% for `(M,K,N)=(17408,5120,512)`, E=25.4% for `(5120,17408,512)`; no-fold bounds 5.2% and 3.0%. However, the independent rate probe did **not** plateau: grid160/320 differed 9.9%, exceeding the old 5% rule. W2 is authorized here by the assignment's E admission, not by pretending the entire older A0 plateau gate passed. The rates below are qualitative scheduling evidence, not a calibrated peak, a clock-count measurement, or an H1/H2 verdict. Retain that limitation in the gate report (`/home/kaden/ClaudeCode/warpfront/wt-lloyd/docs/plans/2026-09-15-halo-gap-plan.md:118-135`; `/home/kaden/ClaudeCode/warpfront/wt-lloyd/crates/hipfire-runtime/examples/tmp_halo_iu4_calibrate.rs:850-875,937-950`).

The requested two-iteration, 8% per-call in-model kill replaces the earlier prospective W2's 15% admission threshold. Do not carry forward its unmeasured 100–300 ms pp512 ledger. **Conditional planning target: 10–18% per-call reduction; measured credit today: zero.**

Non-goals: changed output tile walk/order or grid swizzle; extra weight copies or packing; activation quantizer changes; split-K, alternate fold/reduction, FP16 accumulators; T0 shape fusion; S1 multistream scheduling; F2 chunk-size/state-cadence work; F3 attention work; new architecture routes; capture/replay, speculative or retained-path enablement. FA2 remains an independent H24/KV4/D256 campaign; W2 does not touch attention or infer its geometry from IU4.

## 2. Cited shipping-kernel anatomy

All statements here describe source or explicitly attributed historical measurements, not freshly inspected machine code.

| Item | Shipping implementation and evidence |
|---|---|
| Tile and launch | Output M128×N128, group-loop K256, with two K128 dot/fold halves. K must be divisible by256. Grid `[ceil(M/128),ceil(N/128),1]`, block `[32,8,1]` =256 threads/eight wave32s; x selects rows, y columns. FULL=false bounds/clamps the tails (`/home/kaden/ClaudeCode/warpfront/wt-lloyd/kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip:43-47,225-230,238-261`; `/home/kaden/ClaudeCode/warpfront/wt-lloyd/crates/rdna-compute/src/gemm.rs:19058-19070,19093-19110`). |
| Wave roles/output ownership | All eight waves load and compute; no loader-only wave. `i0=32*(wave/2)`; column offset `16*(wave%2)`; j0=0,32,64,96 and n=0,1 give eight 16×16 subtiles/wave, or32×64 outputs. A lane owns64 FP32 running sums. Local output row is `i0+16*n+2*l+(lane/16)` and column is `j0+16*(wave%2)+(lane%16)`, l=0..7 (`/home/kaden/ClaudeCode/warpfront/wt-lloyd/kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip:82-116,170-188,236,264-287`). |
| LDS planes | `tile_y` holds Xq, not weights: bytes `[0,9216)`, 128 columns×18 dwords, stride72 B. `tile_x` holds A: `[9216,30720)`, 128 rows×42 dwords, stride168 B. A row has32 packed payload words, eight replicated half2 header words, two pad words. Total30,720 B dynamic LDS; no second plane (`/home/kaden/ClaudeCode/warpfront/wt-lloyd/kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip:47-53,132-158,232-234`; `/home/kaden/ClaudeCode/warpfront/wt-lloyd/crates/rdna-compute/src/gemm.rs:19095-19110`). |
| A global loads | A group is136 B: two4-byte `(f16 scale,f16 zero-point)` headers at+0/+4 and128 B packed nibbles at+8. For each of16 rows/wave, lane tx copies one dword at `gp+8+4*tx`, `gp=A+136*(row*(K/256)+kb)`. Adjacent lanes span one128-byte payload, but successive rows are separated by an entire packed K row; this is not a single contiguous 17-KiB CTA read. Metadata ownership is `row=16*wave+lane/2`, half=`lane&1`: one header dword/lane converted through f32 back to half2 and replicated four times. Logical traffic is128×136=17,408 B/group/CTA; payload16,384 B plus headers1,024 B (`/home/kaden/ClaudeCode/warpfront/wt-lloyd/kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip:123-159`). |
| Xq global loads | `block_i4_128` is72 B: f32 d, i32 exact signed code sum s, and64 B packed signed nibbles. Order is `[K/128,N]`; half h of group kb starts at `Xq+((2*kb+h)*N+col0)`. The256 threads copy contiguous dwords, nine/lane/half, into2,304 LDS words. Each half costs9,216 B; both cost18,432 B/group/CTA. Full-tile logical A+Xq traffic is35,840 B/group, excluding cache effects and output traffic (`/home/kaden/ClaudeCode/warpfront/wt-lloyd/kernels/src/block_i4_128_quant.hip:21-29,147-149`; `/home/kaden/ClaudeCode/warpfront/wt-lloyd/kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip:241-259`). |
| WMMA issue structure | For each `(j0,n)` the source initializes one `int32x8_t acc`, then t=0..3, two dependent IU4 WMMAs/t into that same acc: eight K16 steps per K128 half. A is unsigned, Xq signed; intrinsic flags are `(false,A,true,B,acc,false)`. Lanes r and r+16 read identical fragments; source performs four b64 operand reads per two WMMAs. There are eight output chains/wave/half executed successively, **not eight simultaneously live independent chains**:64 WMMAs/wave/half,128/group,1,024/WG/group. Actual scheduler overlap still requires ISA inspection (`/home/kaden/ClaudeCode/warpfront/wt-lloyd/kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip:118-119,170-199`). |
| Fold | After all eight integer WMMAs for one output subtile: `sum += sc_w*d_x*float(acc) + zp_w*d_x*float(s_x)`, with weight header index `4*h`. Every output sees kb increasing, h0 before h1; the integer accumulator resets at each half. The same float expression, cast positions, multiply/add DAG and compiler contraction settings are part of the bit-exact contract (`/home/kaden/ClaudeCode/warpfront/wt-lloyd/kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip:188-210,238-260`). |
| Synchronization/epilogue | Per group: load A+Xq0 → publish barrier → dot/fold0 → read-complete barrier → load Xq1 → publish barrier → dot/fold1 → read-complete barrier. Four barriers/group. Each valid element has one writer at `Y+(col0+j)*M+(row0+i)`; add reads that existing element once and adds the final sum; set overwrites it. No atomic or cross-WG reduction (`/home/kaden/ClaudeCode/warpfront/wt-lloyd/kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip:238-287`). |
| Registers/occupancy | Source records190 VGPR/zero spills from A2; this is not fresh metadata. Full set/add choose `_occ3`, tails choose the base entry. Physical two-WG residency, not three, follows the §1 resource arithmetic; object allocation granularity and actual launch LDS must still be archived (`/home/kaden/ClaudeCode/warpfront/wt-lloyd/kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip:48-51,291-297`; `/home/kaden/ClaudeCode/warpfront/wt-lloyd/crates/rdna-compute/src/gemm.rs:19065-19070`; `/home/kaden/ClaudeCode/warpfront/wt-lloyd/docs/plans/2026-09-15-halo-gap-plan.md:78-100`). |

### 2.1 Host, representation, and ownership contracts

`Gpu::gemm_mq4g256v2_mmq_prequant_iu4` is the only private selection/launch integration point. It currently accepts exact gfx1100/gfx1151, serializes **A0, Xq8, Y16, M24, K28, N32, add36** in both pointer-array and blob forms (40 user-payload bytes, not a claim about compiler-rounded kernarg size), and profiles the selected symbol. Public set/add wrappers pass false/true respectively (`/home/kaden/ClaudeCode/warpfront/wt-lloyd/crates/rdna-compute/src/gemm.rs:19039-19151`). Preserve that ABI and public API.

`Gpu::ensure_int4_mmq_x` delegates stream/capture/replay ownership to scratch. Scratch grows a dedicated72-byte-per-block buffer, bumps `int4_mmq_generation`, and **always** launches `quantize_int4_mmq_ds128`; a stable pointer is not permission to reuse old activations. C2 instead reserves the same buffer/generation and lets the producer fill it. W2 consumes either valid representation without requantization or a new reservation (`/home/kaden/ClaudeCode/warpfront/wt-lloyd/crates/rdna-compute/src/dispatch.rs:2858-2895`; `/home/kaden/ClaudeCode/warpfront/wt-lloyd/crates/rdna-compute/src/scratch.rs:1306-1444`). The production module concatenates the one shared quant recipe and IU4 source; no registration or quantizer copy is needed (`/home/kaden/ClaudeCode/warpfront/wt-lloyd/crates/rdna-compute/src/kernels.rs:3403-3406`; `/home/kaden/ClaudeCode/warpfront/wt-lloyd/kernels/src/block_i4_128_quant.hip:31-88`).

Exact caller inventory, all in `/home/kaden/ClaudeCode/warpfront/wt-lloyd/crates/rdna-compute/src/gemm.rs`:

- Set/add forwarding calls: `gemm_mq4g256v2_mmq_set_prequant_iu4` at19139; `gemm_mq4g256v2_mmq_add_prequant_iu4` at19151.
- `gemm_mq4g256v2_small_tail_set_iu4` fallback:28476.
- `gemm_qkvza_mq4g256v2_wmma`:28516-28534; prepared sibling:28687-28705.
- `gemm_qkv_mq4g256v2_wmma`:29104-29106; prepared sibling:29228-29230.
- `gemm_gate_up_mq4g256v2_wmma`:30235-30238; prepared sibling:30385-30388.
- `gemm_mq4g256v2_residual_wmma`:31778; prepared sibling:31929.

These callers intentionally keep their argument/sidecar contracts and migrate through the single private selector; no new per-family API. LSP returned empty definition/reference results despite visible calls, reported as a worktree/server mismatch. This inventory was recovered with source search/read. H must repeat references with a worktree-attached server before implementation and account for new sibling changes.

## 3. What A0 establishes, and the rejected alternatives

Parent-reported event rates at grid160, loop4096, in wave-level WMMA/µs/CU using A0's40-CU divisor:

| Probe | Rate | Interpretation |
|---|---:|---|
| dependent |69.5| One live accumulator, four dependent WMMAs/iteration. |
| independent |162.5| Four live accumulators, one WMMA each/iteration. |
| independent+LDS |194.0| Same four chains with intended two b64 LDS operand reads/WMMA. |

`69.5/162.5=0.428≈0.43`: dependence materially limits the one-chain issue pattern as well as global-load exposure. `194/162.5=1.194≈1.19`: this probe does not show an LDS co-issue penalty at that operating point. It does **not** prove LDS instructions are free, that additional LDS causes a19% speedup, or that two chains obtain the four-chain rate. The intended probe loops are at `/home/kaden/ClaudeCode/warpfront/wt-lloyd/crates/hipfire-runtime/examples/tmp_halo_iu4_calibrate.rs:62-214`; ratio/count reporting at812-839. Confirm actual LDS instructions before drawing an emitted-ISA conclusion: invariant source reads can be optimized, and the no-plateau limitation in §1 remains. No fixed WMMA cycle or peak-utilization claim is authorized.

The 25% term comes from **exposed serialized fill and associated work**, not a measurement that25% of time is DRAM bandwidth. Shipping fill completes before WMMA begins, with no next-slab register lifetime across compute (source cited in §2). A0's noload twin fills group0 normally, then skips A/Xq fills, retaining barriers and WMMA/fold on initialized stale data. Removing fills also removes their address/conversion/LDS-write work. Its intentionally invalid output cannot be a numeric oracle. The nofold twin substitutes an integer checksum folded into output, so5.2%/3.0% are confounded bounds, not a new arithmetic optimization budget (`/home/kaden/ClaudeCode/warpfront/wt-lloyd/crates/hipfire-runtime/examples/tmp_halo_iu4_calibrate.rs:320-375,377-475,879-950`).

| Alternative | Budget and verdict |
|---|---|
| Eight-wave full next-slab registers | Even half-phased payload staging adds8 A+9 Xq words/lane before headers/addresses:190+17=207, already beyond the192 ceiling for two WGs. Full next A plus Xq is worse. Reject the unchanged-eight-wave form before timing. |
| LDS double buffer, old8+2 roles |61,440 B/WG forces one WG/CU; a second WG would need122,880 B. Role-specific register counts cannot evade one whole-kernel allocation. Reject for this occupancy contract. |
| More independent accumulators alone | An extra int32x8 costs eight lane registers before operand/address growth; it does not overlap global fill. At eight waves,190+8 exceeds192. At four waves, two chains fit the design budget and compensate qualitatively for fewer waves, but do not by themselves solve E. |
| **Selected four-wave half-slab register pipeline** | Keep128×128/30,720 B/grid walk. Double outputs/lane to128, use two output chains, and retain only34 pending payload dwords/lane. Two WGs remain possible below the whole-kernel300 ceiling. Stage payload globally during WMMA; publish into **retired** half-regions only. Headers remain a small exposed turnover operation. |

## 4. Frozen W2 kernel contract

### 4.1 Names, geometry, and output mapping

All additions to `/home/kaden/ClaudeCode/warpfront/wt-lloyd/kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip` are enclosed in **`#if defined(__gfx1151__)`**. Do not change the existing IU4 macros, helpers, entries, non-RDNA branch, or quantizer. New exact symbols:

- `gemm_mq4g256v2_residual_mmq_iu4_full_set_w2_gfx1151`
- `gemm_mq4g256v2_residual_mmq_iu4_full_add_w2_gfx1151`
- Internal `gemm_iu4_body_w2_gfx1151`, `vec_dot_i4_x128_w2_gfx1151`, `prefetch_iu4_half_w2_gfx1151`, `publish_iu4_half_w2_gfx1151`.

Entries use the unchanged seven-argument ABI and `__launch_bounds__(128,2)`. No non-Halo no-op exports. Both set/add retain the runtime add argument and old expression; callers pass0/1. Grid remains `[M/128,N/128,1]` for eligible full tiles, block `[32,4,1]`, dynamic LDS30,720 B. Do not call the existing helper with four threads-y: its eight-wave assumptions are not this mapping.

For wave w=0..3 and lane=0..31:

- `i0=64*(w/2)`, column offset=`16*(w%2)`.
- j0=0,32,64,96; n=0,1,2,3; l=0..7.
- `row=i0+16*n+2*l+(lane/16)`; `col=j0+16*(w%2)+(lane%16)`.
- `sum[((j0/32)*4+n)*8+l]`, exactly128 f32/lane, initialized+0 as before.

Every wave owns64×64 outputs, in four alternating16-column strips. This changes **intra-WG ownership**, not the block/tile walk or output layout. Keep all sum indices compile-time scalarizable; no dynamically indexed scratch array is acceptable.

For each j0, process n pairs `(0,1)` then `(2,3)`. Each pair has two distinct int32x8 accumulators. Loop u=0..7 over K16 in order; issue WMMA for n0 then n1 using the same B fragment. Load only the current two A b64 fragments plus one B b64 fragment, not fragments for all eight K16 steps. Source operand reads become three b64 per two WMMAs, versus four in the shipping source; machine-level counts are a gate, not assumed. Fold each completed chain in n order, then advance the pair. This does not split or reassociate an individual output's dot.

### 4.2 Pending-byte ownership and LDS addresses

One reusable register bundle/lane contains `A_pending[16]` and `Xq_pending[18]`: **34 dwords=136 bytes**, no second register bank. Current operands live in LDS; pending operands live only in registers. Arrays must scalarize in the emitted object.

For a target A group g, half h, each lane loads16 dwords with q=0..15:

- `r=8*q+2*w+(lane/16)`, payload word=`16*h+(lane%16)`.
- Global address=`A+136*((row0+r)*(K/256)+g)+8+4*payload_word`.
- Publish destination=`tile_x[r*42+payload_word]`.

This has one unique writer per2,048 payload dwords of an A128 half. A half-wave reads64 contiguous bytes of one row; paired half-waves read two different rows. **This differs from the shipping128-byte wave-wide row load** and may cost transactions at136-byte group alignment; it is a named risk, not a claimed memory improvement.

For target activation half `(g,h)`, q=0..17:

- `l=128*q+32*w+lane`.
- Load dword l from `reinterpret_cast<const int*>(Xq+((2*g+h)*N+col0))`.
- Publish to `tile_y[l]`.

All2,304 dwords have one writer; a four-wave sweep remains contiguous. Do not repack or reinterpret d/s numerically.

Headers are **not** retained across WMMA. After payload publication has consumed/released pending registers, lane `(w,lane)` owns header row `r=32*w+lane`, loads the target half's dword at `gp+4*h`, and uses the **same half→float→half2 conversion and fourfold replication** into `tile_x[r*42+32+4*h+v]`, v=0..3. Do not replace this with a raw half2 cast: unusual header bits can distinguish conversions. One half's header operation is512 global bytes/WG. Keep both pad words untouched. No full-row/vector store may overwrite the other half's payload or headers.

### 4.3 State machine: warmup, overlap, turnover, drain

Let G=K/256>0. A and Xq are immutable caller/scratch-owned global inputs; Y is a disjoint caller-owned destination. Each workgroup exclusively owns its private LDS and its output tile. A-half logical epochs are separately tracked even though they share42-dword rows.

| Transition | Before / private pending state | Action and after invariant |
|---|---|---|
| Prologue | No LDS contents valid; sum128=+0. | Fill both A halves/headers of group0 and Xq(0,0), using bounded loads; one all-wave publish barrier. A epochs=(0,0), Y epoch=(0,0). No next-group access yet. |
| Compute h0 of g | A half0=g and Y=(g,0); A half1=g. | If g+1<G, issue A(g+1,0) payload into private pending16. Always issue Xq(g,1) into pending18. Execute current h0 dot/fold while loads are outstanding. LDS remains unchanged throughout all readers' work. |
| Retire h0 / publish h1 | Every wave completes h0 fold; pending loads may still be outstanding. | All-wave read-complete barrier R0; consume/wait pending values before stores. Replace only retired A half0/header with g+1 if valid; replace Y with(g,1). All-wave publish barrier P1. A half1 still=g; h1 readers may now start. |
| Compute h1 of g | A half1=g, Y=(g,1); half0 may already=g+1. | If g+1<G, issue A(g+1,1) and Xq(g+1,0) payloads into the reused34 registers. Execute current h1 dot/fold without changing LDS. On final g, issue neither next-group load. |
| Retire h1 / advance | Every wave completes h1 fold. | All-wave R1. If next exists, publish pending A half1/header and Y(g+1,0), then all-wave P0. Both A halves now=g+1 and Y=(g+1,0). Increment g. |
| Drain / epilogue | At final R1, no next-group pending state exists. | Skip the uniformly unnecessary P0; write each owned Y once with unchanged final set/add expression. No speculative load past G; no remaining consumer of scratch escapes launch completion. |

All four waves participate in every executed barrier; next-group validity is workgroup-uniform. No lane-local early return. R0/R1 establish **all-reader completion**, including header reads during the fold, before any in-place store; P1/P0 establish all-writer publication. Do not merge these distinct barriers. Count: prologue1 +2G h0 barriers +G h1-retire barriers +(G−1) next-h0 publish barriers =**4G**, equal to the current source count. The benefit sought is hidden global payload latency, not invented barrier removal.

G=1 is a required case: warmup, h0 with only pending Xq1, R0/P1, h1 with no prefetch, R1, epilogue. G=2 and odd G test alternating half epochs. Compiler-generated waits may move earlier than desired: disassembly must show pending global loads before current WMMA and their completion dependency deferred until turnover, not a global-load wait at the start of the dot. There is no AMD asynchronous-copy promise in this C++ schedule.

Host transition stays: valid prepared generation or freshly quantized sidecar → same-stream GEMM read → unique Y write → existing stream/replay lifecycle. No generation bump, free, reservation, producer rerun, frame-counter change, KV/GDN/conv state transition, or rollback mechanism is introduced. A device fault retains the existing error semantics, not an atomic transaction guarantee.

## 5. Whole-kernel VGPR arithmetic and occupancy gate

Use the predecessor's hardware model:2 SIMD/CU,1,536 lane-register units/SIMD,16 waves/SIMD,65,536 LDS bytes/CU; confirm exact object/launch constraints rather than relying on the module profiler (`/home/kaden/ClaudeCode/warpfront/wt-lloyd/docs/plans/2026-09-15-halo-gap-plan.md:80-100`).

**First-order delta estimate, not an emitted count:**

| Component | Per-lane registers |
|---|---:|
| Historical shipping allocation context |190|
| Additional f32 outputs:128−64 |+64|
| One A128/Xq128 pending payload bundle:16+18 |+34|
| Second independent int32x8 output accumulator |+8|
| **Planning estimate** |**296**|

This estimate conservatively keeps the incumbent's other126 registers rather than assuming unmeasured compiler-fat removal. It is not additive compiler accounting proof: new addressing, changed unrolling and accumulator lifetime can exceed it. Streaming current K16 operands avoids keeping more fragment words than the old eight-word source bundle; headers execute after pending stores and cannot add a persistent register. Do not keep current+next register slabs simultaneously or unroll all n-pairs into simultaneously live accumulators.

For explicit live-state auditing, the selected computation needs128 sum +16 accumulator +6 current fragment +34 pending =**184 named lane scalars**, leaving **112** within the296 planning allocation for addresses, loop/control temporaries, fold conversions, compiler live ranges and allocation effects. A source scalar inventory is necessary but not sufficient for zero scratch. Record actual per-entry high-water allocation, not separate fictional loader/compute allocations.

At296 allocated registers, `floor(1536/296)=5` waves/SIMD; two four-wave WGs need only4 waves/SIMD. Their LDS is2×30,720=61,440 B≤65,536. A third WG fails LDS. Thus the target is **two WGs/CU, eight waves/CU, two independent chains/wave**, versus the incumbent's two WGs,16 waves, one source chain/wave. Both have16 source-level simultaneously available output chains/CU; the new schedule additionally places pending global loads across current compute. This is a reason to measure it, **not proof that two chains replace wave scheduling or that occupancy implies throughput**.

Hard gate for **each exact set/add entry**, before any correctness launch:

1. Allocated VGPR≤300 for the whole kernel, target approximately296 or lower; **round using the actual kernel descriptor granularity first**. If296 used rounds to304 allocated, it fails. The count must also permit two WGs at block128; no ceiling relaxation to384.
2. Zero VGPR spills, zero scratch/private segment; missing spill metadata means unknown, not zero. Inspect SGPR counts/spills and other block constraints too.
3. Fixed+dynamic LDS exactly30,720 B and launch128, with two-WG resource capacity proven from the object. Reject unexpected static LDS or local-memory arrays.
4. Archive compiler options, source/prelude/object identity, entry descriptors, kernarg offsets/size, `.vgpr_count`, `.sgpr_count`, spill report, `.private_segment_fixed_size`, `.group_segment_fixed_size`, allocated granularity and disassembly. Keep the40-byte user payload separate from compiler padding/implicit arguments.
5. Inspect actual schedule: two independent C chains interleaved, exact IU4 flags/count, outstanding payload global loads across current WMMA, same fold DAG, no dead prefetch or premature wait. A static source ordering is insufficient.

The largest feasibility risk is **halving runnable waves without achieving useful per-wave load/WMMA overlap**, compounded by the narrow≤300 budget. If metadata/ISA does not support that mechanism within the first bounded iteration, do not time an overflowing or spilled kernel and call it a candidate.

## 6. Bit-exactness and cross-architecture rules

For every output and half, candidate u=0..7 is exactly shipping `(t=u/2, low_or_high=u%2)`. Keep the identical A/Xq nibble addresses, unsigned-A/signed-B intrinsic flags, and zero initial int32x8. Interleave only **different outputs**; never split one output into partial K accumulators or combine partial dots. A128 dot is bounded by128×15×8=15,360 in absolute value for nibble-valid input, so no i32 overflow is introduced. This bound does not authorize a different integer reduction order: keep the old order anyway.

Copy the existing per-half fold expression and its conversions into the new output mapping. Each sum slot sees g=0..G−1, h0 then h1, exactly once. Preserve f32 operation grouping, contraction/fast-math flags, signed-zero/exceptional-value behavior wherever the baseline defines it, and final `Y += v` versus `Y = v`. Sharing a B fragment between independent chains changes neither operand bits nor the individual accumulation order. Moving which lane owns an output is not itself a numeric change; the direct bit oracle decides compiler-level parity.

**No numeric-gate variant is needed or authorized.** If the implementation needs split-K, reordered folds, FP32 reassociation, reduced precision, or altered header conversion to obtain speed, stop this bit-exact W2. Such an arm must have distinct `_w2_reordered_gfx1151` naming, an independently approved numeric/KLD/state gate and default-off route in a separate spec; it cannot inherit this plan's result or weaken a failed bit oracle.

Final selector in `Gpu::gemm_mq4g256v2_mmq_prequant_iu4`: select the W2 full entry only for exact `self.arch=="gfx1151"`, positive M/K/N, full128-aligned M/N, valid existing K%256 contract, `!self.replay.is_recording()` and `!self.graphs.capture_mode`. Existing public callers and flag behavior remain unchanged. Capture/recording, gfx1100, non-full tiles and empty/off-envelope cases retain their old symbol/block. Other architectures retain the existing rejection/route. Existing eager predicate conventions are visible at `/home/kaden/ClaudeCode/warpfront/wt-lloyd/crates/rdna-compute/src/gemm.rs:28506-28514,29095-29102,31769-31776`.

The selector must choose **symbol and block together**: W2 `[32,4,1]`; baseline `[32,8,1]`; same row/column grid and30,720-byte LDS. Keep force-blob execution supported and numerically checked, without enabling capture. No new public unchecked helper or environment flag: direct oracle launches the unique symbols; temporary A/B binaries select old/new at the private launcher. Rejected candidates do not remain behind a latent switch.

## 7. Ordered, bounded composer units

Frozen ownership: **K** owns only the IU4 HIP file; **O** owns the new throwaway `/home/kaden/ClaudeCode/warpfront/wt-lloyd/crates/hipfire-runtime/examples/tmp_halo_iu4_oracle.rs`; **H** exclusively owns `/home/kaden/ClaudeCode/warpfront/wt-lloyd/crates/rdna-compute/src/gemm.rs` and `/home/kaden/ClaudeCode/warpfront/wt-lloyd/crates/hipfire-runtime/Cargo.toml` for integration. H serializes the manifest with other live example owners. No edits are needed to kernels.rs, dispatch.rs, scratch.rs, the shared quantizer, model callers, attention, or GDN. Coordinate before any shared-file mutation. Main, not composers, runs all compile/object/GPU gates; no sibling runs shared validation.

1. **W2.0 — complete candidate, metadata-only gate (K;≤4 h; first artifact gate).** Implement the exact guarded symbols/helpers and complete §4 arithmetic/state machine, leaving baseline bytes/routes intact. No placeholder/no-op “skeleton” whose smaller resources are substituted for the final kernel. Supply production-concatenated source to Main. Main compiles only these objects with production flags/prelude and inspects §5, **without launching any kernel**. Compile baseline gfx1151 plus non-Halo preprocessing/object controls as appropriate; confirm no W2 entry on gfx1100/gfx12 and no baseline route mutation. Acceptance: each final-body entry meets resource/ABI/schedule gate. Metadata failure blocks W2 execution.
2. **W2.1 — standalone bit oracle (O;≤4 h; runnable after0).** Prepare this independently of K using the frozen symbol/ABI/grid/block contract, so its development can overlap0. Add a lab-gated example via H using the existing example convention (`/home/kaden/ClaudeCode/warpfront/wt-lloyd/crates/hipfire-runtime/Cargo.toml:189-192`). Its `--compile-only` mode must return before allocation/launch/timing; its explicit correctness mode runs only after Main approves the object manifest. Use production `GEMM_MQ4G256V2_RESIDUAL_MMQ_IU4_SRC`, not A0's diagnostic twin, as reference source. Acceptance: §8.1 output/input/guard bits and phase-edge cases; no production host route changes.
3. **W2.2 — controlled in-model route and first measurement (H;≤2 h composition, Main≤3 h gates; after0/1).** Modify only the private selector/block choice under §6 in a candidate binary, keeping a frozen accepted-predecessor binary. Public wrapper/caller APIs do not change. All ensure/launch/timer names must be the same selected W2 symbol. Acceptance: params/blob parity, prepared/unprepared consumers, tail/non-Halo/capture route isolation, pp512 per-call profiles and required pp512/2048/8192 bench rows. This is **iteration1**; isolated timings alone cannot admit it.
4. **W2.3 — one bounded scheduling correction, or stop (K then Main;≤4 h composition plus≤3 h gates).** Only if iteration1 identifies a specific schedule defect, permit one adjustment of prefetch lead distance or scalar lifetime/current-fragment scheduling with the exact same34-word maximum, geometry, LDS ownership, output arithmetic and tile walk. No third candidate, larger register bundle, LDS-double-buffer fork, A5 swizzle, new precision or occupancy relaxation. Repeat metadata→bit oracle→same model gates. This is **iteration2**. Acceptance: §8 thresholds; otherwise remove W2 routing/candidate, report reject/inconclusive and stop.
5. **W2.4 — admission/veto (Main/reviewer/H;≤4 h; after measured pass).** Reviewer receives object manifests, source/ABI identity, bit/guard evidence, all raw timing/run-order data, exact per-symbol call counts, model eval/serve/isolation results and weighted delta versus the frozen predecessor. Reviewer may veto nominal speed for lost isolation, unmatched work or invalid measurement. This plan itself authorizes no deployment or commit.

**Total timebox: three working days, at most24 active hours across composition and parent gates; at most two scheduling iterations.** Missing metadata, unresolved noise, numerical mismatch or failure to reach the hard8% model gate within that box ends W2 rather than starting an unmeasured performance branch.

## 8. Claim-scoped validation and promotion

### 8.1 Numeric/oracle gate, before timing

The old `tmp_halo_iu4_oracle.rs` is absent in this snapshot; recreate the narrowly scoped harness rather than claiming it was run. A0 provides useful packing/event plumbing, but its no-load/no-fold arms are **not** references and its attribution launcher hardcodes set (`/home/kaden/ClaudeCode/warpfront/wt-lloyd/crates/hipfire-runtime/examples/tmp_halo_iu4_calibrate.rs:512-567,686-727`).

Minimum direct fixtures:

- **Gate set `(17408,5120,512)` and down add `(5120,17408,512)`**, shipping `_full_set_occ3`/`_full_add_occ3` versus matching W2 entry; identical A/Xq and separately cloned Y. Down begins with nonzero, varied finite Y. Reset Y before every add sample outside timing; do not accumulate across repetitions.
- Warmup/drain with G=1,2,3 (`K=256,512,768`), including nonzero distinct per-half headers, nonzero s, signed Xq extremes−8/+7, unsigned weight codes0/15, alternating nibble patterns, all-zero activations, mixed-sign/cancellation outputs and guarded buffers. A0's weight filler uses only1..7 and Xq filler−1..3, so extend it for these boundaries rather than inheriting incomplete coverage (`/home/kaden/ClaudeCode/warpfront/wt-lloyd/crates/hipfire-runtime/examples/tmp_halo_iu4_calibrate.rs:523-536,552-559`).
- Full route inventory: set `(M,K)=(1024,5120),(6144,5120),(10240,5120),(12288,5120),(17408,5120)`; add `(5120,5120),(5120,6144),(5120,17408)`; N128/256/512 and1024 only if its model envelope is admitted. Compare all f32 output **bits**, immutable A/Xq bytes and guards, not an allclose tolerance or a checksum-only match.
- Exercise both freshly `ensure_int4_mmq_x`-quantized inputs and C2-prepared sidecars through the real caller path; no additional quantizer or generation bump in candidate profiles. Include unchanged off-envelope M/N tail, gfx1100, capture/recording selection, and both params/force-blob launch forms. Do not launch an unsupported W2 symbol as an architecture test.

Any mismatch blocks all timing/promotion. Save the first differing output's row/column/K-half fixture and candidate/baseline bits; do not mask it with a tolerance. Recheck metadata before exercising any revised source.

### 8.2 Timing gate: isolated attribution is necessary, model transfer is decisive

All timing belongs to Main on the Halo, exact gfx1151, with the accepted predecessor fixed for unrelated F/G changes. **No CPU builds of any tree on the APU during timing.** Build/JIT first, then reach the same warm thermal state; archive device/CU identity, clocks/power context, compiler, source/object/model/prompt hashes, flags including unset values, call counts and run order. A0's40-CU divisor belongs to its recorded rates; do not mix it with a different campaign's CU assumption. Per-call/whole-request comparisons do not require inventing a peak-rate denominator.

1. After metadata and oracle, run100 interleaved event samples of gate-set and down-add after matched warmup. Match tensors/output resets and output work. Report median/min/max, paired spread and raw microseconds for **both** shapes. These shape results cannot be substituted for blended in-model symbol means.
2. Run the existing `profile_prefill_qwen35` at pp512 with warmup3 and Q8 KV, using prebuilt binaries, baseline/candidate ABBA. The executable/arguments are documented in `/home/kaden/ClaudeCode/warpfront/wt-lloyd/docs/plans/2026-09-15-halo-gap-plan.md:319-328`. Archive IU4 set/add counts, total_us and avg_us, and quantizer/producer counts. The supplied baseline is **set1688 µs×272, add1643 µs×128**; it is an in-model family average, **not** the isolated17408-row gate time. Calls sum to400; IU4 subtotal is669.44 ms of approximately818 ms profile total. Rebaseline simultaneously; do not compare a throttled candidate with these historical unthrottled numbers.
3. Use three fresh-process ABBA cycles (six observations/arm), as in the common recipe. Run the prebuilt production `hipfire bench` matrix with `--pp 512,2048,8192 --ctx 128,2048,32768 --runs 1 --warmups 10 --kv-mode q8`. Archive all rows, not the best rows from separate processes. pp512/2048/8192 are mandatory; longer rows may be supplemental, not replacements.
4. **Hard per-call gate:** at least8% reduction in **each** pp512 in-model set and add average, with matched call counts/routes; weighted IU4 total must also improve≥8%. At the historical baselines those boundaries are set≤1552.96 µs and add≤1511.56 µs. No family may hide an add regression inside a weighted set win. A passing isolated gate and a failing in-model gate is a rejection.
5. Require positive whole-request wall transfer beyond paired spread, and no>2% regression in any required prompt/decode row. Gain≤paired spread is not admission; allow one extra paired cycle within the same box, otherwise inconclusive/reject. If either family remains below8% improvement after two iterations, **kill W2**.

**Projection, not measurement:** hiding approximately40–70% of the observed≈25.7% exposure would yield≈10–18% per-call reduction before overheads. This gives set≈1384–1519 µs, add≈1347–1479 µs, and≈67–120 ms off the669.44-ms IU4 subtotal at unchanged400 calls. Do not add a separate2.34× dependency credit or1.19× LDS credit; these effects interact and probes did not plateau. Do not extrapolate that subtotal saving directly to bench wall or longer prompts. The projection includes a real possibility of zero/negative gain.

### 8.3 Product parity/isolation and final evidence

After kernel/model timing admission, Main runs the common one-chunk `eval_hipfire` baseline/candidate digest check and actual serve battery with deterministic decoded-text comparison, plus retained-path isolation required by the campaign (`/home/kaden/ClaudeCode/warpfront/wt-lloyd/docs/plans/2026-09-15-halo-gap-plan.md:340-358`). Direct tensor bits remain the numerical proof; the eval file's aggregate digest is supplemental, not an all-hidden-state proof. No new retained replay behavior is claimed. Shared build/test/lint validation, if required, runs once under Main after accepted sibling integrations, never during Halo measurement.

## 9. Risks, abandon rules, and planning evidence

- **Primary performance risk: fewer runnable waves plus failed compiler overlap.** Two chains may not replace lost wave-level latency hiding, and early `vmcnt` waits can turn the pending bundle into overhead. Reject on the ISA gate or the two-iteration8% model gate; do not declare speed from the source state machine.
- **Register risk:**296 is a planning estimate close to300, not measured compiler output. Scalarization, address materialization, allocation rounding or unintended unrolling may exceed it. A spilled/misrounded artifact is rejected before correctness execution. Never borrow a per-role budget or weaken occupancy.
- **A-stream locality across400 calls:** A5's column-adjacent order was about−7% isolated yet+4% in-model; the recorded paired direction was also contaminated by a concurrent CPU build, so its magnitude/mechanism is not a clean causal measurement. It is sufficient warning to **retain the original tile walk** and make model transfer decisive (`/home/kaden/ClaudeCode/warpfront/wt-lloyd/docs/plans/2026-09-15-halo-gap-plan.md:139-152`). W2's half-row64-byte A loads and earlier within-WG next-group accesses can still hurt cache/transaction behavior without changing block order. Do not claim cache neutrality.
- **LDS bank conflicts:** preserve stride42, metadata slots and Y stride18. A2's42-row stride replaced44 for bank behavior; that measured result does not prove this four-wave reader/writer schedule conflict-free (`/home/kaden/ClaudeCode/warpfront/wt-lloyd/kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip:48-51`). Verify b64 alignment/broadcast patterns and actual LDS instructions; no hidden padding/stride tweak in iteration2.
- **Half-epoch races:** writing next A half0 before all current h0 folds finish corrupts scale/payload reads; replacing both headers together corrupts current h1. R0/R1 plus narrow stores are mandatory. No barriers inside lane-divergent conditions and no out-of-bounds final prefetch.
- **APU measurement rule:** no cargo/CK/other CPU build on the Halo during any timing, even in another checkout. Discard contaminated samples; never normalize them using the historical1688-µs value. Source and object preparation must finish before the timed window (`/home/kaden/ClaudeCode/warpfront/wt-lloyd/docs/plans/2026-09-15-halo-gap-plan.md:143-152`).

**CPU-only planning checks actually executed:** baseline eight-wave and proposed four-wave mapping both covered exactly16,384 output elements once; proposed sum indices cover0..127; proposed A-half mapping has2,048 unique dword writers and Xq mapping2,304; epoch simulation for G=1,2,3,20,68 preserved current-half inputs, never prefetched beyond G, and produced4G barriers; u→(t,low/high) exactly reproduced each output's eight-WMMA sequence. Resource and projection arithmetic above was recomputed. These checks establish indexing/epoch consistency only, not compiled bit parity, race freedom of future code, actual occupancy, or performance.

After the smoke/oracle/model gates prove an admitted implementation, H records actual metadata, timings and verdict in this plan and the existing campaign ledger/changelog, and removes throwaway scaffolding/unused candidate arms. If rejected, restore the old private selection and remove the W2 candidate/oracle registration; retain the rejection evidence, not a dormant performance flag. No cleanup/source changes are part of this planning assignment. Reviewer veto remains final even when numeric timing thresholds pass.
