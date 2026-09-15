# W2: exact-gfx1151 IU4 prefill GEMM — rejected after metadata gate

Date: 2026-09-15, revision after W2.0. Worktree: `/home/kaden/ClaudeCode/warpfront/wt-lloyd`, branch `mq4-lloyd`. **Verdict: reject the implemented W2; do not launch it or begin a second scheduling iteration.** This revision changes only this plan. No source edits, builds, GPU runs, formatters, linters, or project suites were performed by the planning agent. Main's reported object failure is accepted evidence, not a request to reproduce it. Main owns execution/integration; an independent reviewer owns the final veto.

## 1. Decision and corrected architectural limit

**The original296-VGPR proposal was impossible on this target.** Main reports the gfx11 wave32 architectural per-wave maximum is256 VGPR. The old plan's≤300 ceiling, inherited from `/home/kaden/ClaudeCode/warpfront/wt-lloyd/docs/plans/2026-09-15-halo-gap-plan.md:84-100,388`, was invalid even if a CU-wide register-residency quotient appeared to allow it. This revision supersedes that W2 budget: **hard cap256 allocated after descriptor-granularity rounding, zero spills, zero scratch; a redesigned candidate must target≤232 including planning headroom**. Hardware capacity and workgroup residency are separate constraints. A register allocator clamped at256 with spills is a failed object, not an artifact that passes because its reported VGPR count is256.

Main's metadata result, exact gfx1151 and production flags, for the committed implementation:

| Entries | VGPR | SGPR | Reported spill field | Scratch/private bytes | Verdict |
|---|---:|---:|---:|---:|---|
| `gemm_mq4g256v2_residual_mmq_iu4_full_{set,add}_w2_gfx1151` |256|18|391|1252|**Reject before any launch**|
| Shipping `gemm_mq4g256v2_residual_mmq_iu4_full_{set,add}_occ3` |190|21|0|0|Retain|

Do not convert the391 spill-report value into391 additional simultaneously live registers or infer that deleting16 pending scalars eliminates the reported scratch. It is compiler evidence of failure, not a linear live-range measurement. The prior10–18% per-call/67–120 ms projection and two-WG/296 resource argument are **withdrawn**. Correct indexing simulations from the first plan did not establish that the implementation was hardware-feasible.

**Alternatives(a)–(c) are explicitly costed in §5.** Only(a), eight waves with `sum[64]` and Xq-only prefetch, has a plausible headroom budget. It intentionally accepts **one** WG/CU. However, available A0 evidence does not establish that exposed A fill is the smaller term or that the one-WG compute schedule loses less than the very small residual speed budget. No candidate supports an evidence-grounded≥8% net per-call projection. Therefore choose **(d): reject W2 outright for this campaign**, retain shipping, and close the implemented branch. The paper contract for(a) below is an evaluated alternative, **not authorization to implement it or to launch new attribution probes**. Reopening needs new evidence and a separate parent decision; it is not iteration2 of the failed design.

A0's load-exposure result remains valid parent context: E=25.8% at `(M,K,N)=(17408,5120,512)`,25.4% at `(5120,17408,512)`; no-fold bounds5.2%/3.0%. The independent rate probe did not plateau: grid160/320 differed9.9%, exceeding the old5% rule. E passed the load-exposure screen, **not** a split A-versus-Xq attribution or a calibrated peak-rate gate (`/home/kaden/ClaudeCode/warpfront/wt-lloyd/docs/plans/2026-09-15-halo-gap-plan.md:118-135`; `/home/kaden/ClaudeCode/warpfront/wt-lloyd/crates/hipfire-runtime/examples/tmp_halo_iu4_calibrate.rs:850-875,937-950`).

Non-goals: source changes in this assignment; altered tile walk/order or grid swizzle; weight/activation formats, quantizers, split-K or numerical reassociation; T0/S1/F2/F3 work; new architecture/retained/capture routes; generic compiler tuning, source rewriting to chase hypothetical register savings, or a new unmeasured performance branch. FA2 remains independent H24/KV4/D256 work.

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
| Registers/occupancy | Source records190 VGPR/zero spills from A2; Main now independently reports190 VGPR,21 SGPR,zero spills/scratch for the shipping full entries. Full set/add choose `_occ3`, tails choose the base entry. Physical two-WG residency, not three, follows the §4 resource arithmetic; object allocation granularity and actual launch LDS must still be archived (`/home/kaden/ClaudeCode/warpfront/wt-lloyd/kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip:48-51,291-297`; `/home/kaden/ClaudeCode/warpfront/wt-lloyd/crates/rdna-compute/src/gemm.rs:19065-19070`; `/home/kaden/ClaudeCode/warpfront/wt-lloyd/docs/plans/2026-09-15-halo-gap-plan.md:78-100`). |

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

These callers and the private selector remain unchanged after rejection; there is no migration or new per-family API. LSP returned empty definition/reference results despite visible calls, reported as a worktree/server mismatch. This inventory was recovered with source search/read. H must repeat references with a worktree-attached server before implementation and account for new sibling changes.

## 3. What A0 establishes — and what it does not

Parent-reported event rates at grid160, loop4096, in wave-level WMMA/µs/CU using A0's40-CU divisor:

| Probe | Rate | Interpretation |
|---|---:|---|
| dependent |69.5| One live accumulator, four dependent WMMAs/iteration. |
| independent |162.5| Four live accumulators, one WMMA each/iteration. |
| independent+LDS |194.0| Same four chains with intended two b64 LDS operand reads/WMMA. |

`69.5/162.5=0.428≈0.43`: dependence materially limits the one-chain issue pattern as well as global-load exposure. `194/162.5=1.194≈1.19`: this probe does not show an LDS co-issue penalty at that operating point. It does **not** prove LDS instructions are free, that additional LDS causes a19% speedup, or that two chains obtain the four-chain rate. The intended probe loops are at `/home/kaden/ClaudeCode/warpfront/wt-lloyd/crates/hipfire-runtime/examples/tmp_halo_iu4_calibrate.rs:62-214`; ratio/count reporting at812-839. Confirm actual LDS instructions before drawing an emitted-ISA conclusion: invariant source reads can be optimized, and the no-plateau limitation in §1 remains. No fixed WMMA cycle or peak-utilization claim is authorized.

The 25% term comes from **exposed serialized fill and associated work**, not a measurement that25% of time is DRAM bandwidth. Shipping fill completes before WMMA begins, with no next-slab register lifetime across compute (source cited in §2). A0's noload twin fills group0 normally, then skips A/Xq fills, retaining barriers and WMMA/fold on initialized stale data. Removing fills also removes their address/conversion/LDS-write work. Its intentionally invalid output cannot be a numeric oracle. The nofold twin substitutes an integer checksum folded into output, so5.2%/3.0% are confounded bounds, not a new arithmetic optimization budget (`/home/kaden/ClaudeCode/warpfront/wt-lloyd/crates/hipfire-runtime/examples/tmp_halo_iu4_calibrate.rs:320-375,377-475,879-950`).

## 4. Hardware cap and metadata-first resource rules

Use the campaign's stated CU model only for **residency**:2 SIMD/CU,1,536 lane-register units/SIMD,16 waves/SIMD,65,536 LDS bytes/CU (`/home/kaden/ClaudeCode/warpfront/wt-lloyd/docs/plans/2026-09-15-halo-gap-plan.md:80-100`). Add the missing independent constraint from Main: **a wave32 kernel cannot address more than256 VGPR per lane**. `1536/(waves/SIMD)` is not permission to exceed that architectural limit. No fictitious role-specific VGPR allocations, scratch-backed “extra registers,” or occupancy-only admission.

For any separately authorized future candidate:

1. **Allocated VGPR≤256**, with actual descriptor granularity applied before comparison; **design/object target≤232 including named state, compiler-overhead allowance and explicit headroom**. Do not keep a “296 if no spills” arm. If the complete candidate cannot reach the target with zero spills/scratch, stop rather than filling the remaining architecture ceiling with more state.
2. Zero VGPR/SGPR spills and zero private/scratch bytes; a missing spill field is unknown. The reported W2.0 values391/1252 violate this gate regardless of its256 count.
3. Sum static+dynamic LDS from the exact launch. For eight-wave block256 with30,720 B, two WGs need eight waves/SIMD and allocated VGPR≤192. At224 or232, `floor(1536/R)=6` waves/SIMD: **one** eight-wave WG/CU fits, not two. `_occ3` remains a name/launch-bounds value, not three physical blocks.
4. At four-wave block128, two WGs need four waves/SIMD, so registers≤256 can support two on the CU register model **if** other constraints fit. That does not make the four-wave `sum[128]` variants fit per-wave state/headroom. At39,936 LDS, `floor(65536/39936)=1` WG independently of register count; two need79,872 B.
5. Archive each exact entry's source/prelude/object identity, compiler flags, VGPR/SGPR counts, spill output, private/group segments, descriptor allocation granularity, kernel-argument offsets and disassembly. User payload remains40 bytes (three pointers/four i32); compiler padding/implicit arguments are separate. A metadata-only mode must exit before allocation/launch/timing.
6. Compilation of an empty body or a body with pending state removed cannot establish resources of a complete pipeline. Main owns compilation/inspection before any oracle launch; module-level occupancy summaries are insufficient.

The existing shipping190 count must likewise be rounded from its actual descriptor before residency calculations; the familiar192 boundary is the two-WG ceiling, not a claim that VGPR rounding is universally8. Example rounding below uses8 only to expose the arithmetic, never as a substituted hardware measurement.

## 5. Costed alternatives and the projection decision

### 5.1 Register and LDS costs

Start from Main's **measured** shipping190 VGPR, sum64 and source one-chain microkernel, not from the failed object's capped256 plus its spill count. Additive rows are **planning estimates only**; live-range/compiler changes still require a complete metadata gate. Include16 registers of explicit compiler/scheduling headroom, not just payload scalars.

| Requested alternative | Estimated live-allocation change before headroom | With16 headroom; illustrative rounding to8 | LDS / physical WG target | Verdict |
|---|---:|---:|---|---|
| **(a) Eight waves, sum64, Xq-half-only, conservative18-register pending allowance, two output chains** |190+18+8=216|**232**,24 below architectural256|30,720 B; **one WG/CU** at232|Only headroom-feasible family; performance condition unestablished.|
| (a), coalesced ownership actually needed |190+9+8=207|223→**224**,32 below256|30,720 B; **one WG/CU** at224|Preferred paper mapping; same performance obstacle.|
| **(b-X) Four waves, sum128, only Xq18, two chains** |190+64+18+8=280|296|30,720 B; hypothetical two WGs if it fit|Reject: above256 before headroom.|
| (b-X), remove second chain |190+64+18=272|288|Same LDS|Reject; still above256 and loses issue independence.|
| **(b-A) Four waves, sum128, only A16, two chains** |190+64+16+8=278|294→296|30,720 B; hypothetical two WGs if it fit|Reject: above256 before headroom.|
| (b-A), remove second chain |190+64+16=270|286→288|Same LDS|Reject; still above256.|
| **(c) Four waves, sum128, no register prefetch, Xq-only double LDS, one chain** |190+64=254|270→272|**39,936 B; one WG/CU**|Reject: no real register headroom; 254→256 consumes the entire architecture allocation even before allowance.|
| (c), two chains |190+64+8=262|278→280|39,936 B; one WG/CU|Reject: above256 before headroom.|

The requested(a)18-word allowance is explicitly costed, but18 dwords is a **whole `block_i4_128`**, not the minimal per-lane share of an eight-wave CTA's activation half. A128-column Xq half is2,304 dwords;256 lanes need**9** each. Four waves need18 each. Keep original coalesced ownership `l=256*q+32*wave+lane`, q=0..8; do not allocate nine unused registers, duplicate the half, or introduce a strided one-block-per-lane mapping to force the18 figure. The baseline copy loop/representation establishes these counts (`/home/kaden/ClaudeCode/warpfront/wt-lloyd/kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip:241-259`; `/home/kaden/ClaudeCode/warpfront/wt-lloyd/kernels/src/block_i4_128_quant.hip:21-29`).

A named-state sanity check for(a) is64 sum +16 integer accumulators +6 streamed current fragment words +9 pending Xq words =**95 named lane scalars**. Preserve the incumbent's remaining address/conversion/live-range cost in the190-based estimate, and add16 headroom; do not conclude95 is the compiled VGPR count. The conservative18-word ledger similarly has104 named scalars and the232 total budget above. There is no A-prefetch array or next-A pointer live across compute.

For(b), merely deleting one pending array from the failed implementation does not establish zero scratch. Even the optimistic190-based estimates require **48 (Xq) or46 (A) registers removed** to reach232 before adding any new headroom; including16 headroom makes the reductions64/62. No measured compiler live-range evidence supports those cuts. Do not “solve” the deficit by predicting that the existing unrolled consumer will use much less overhead than shipping.

For(c), Y0 `[0,9216)`, Y1 `[9216,18432)`, A `[18432,39936)` gives30,720+9,216=**39,936 B/WG**, not a two-WG layout. The second Y plane stores no extra output sums. With all four waves still both loading and computing, and no register-prefetched global data or dedicated producers, another LDS plane alone does **not** create concurrent global fill and WMMA: each wave's blocking fill must still precede its compute. Introducing a producer/consumer split would be another mapping/resource design, not a free property of(c). Its register and wave-residency losses are not justified by a demonstrated overlap mechanism.

### 5.2 Why byte counts do not admit the one-WG alternative

(a) hides only Xq fill and leaves A fill/headers exposed. Shipping logical per-group bytes are A17,408 and Xq18,432 (§2): A48.57%, Xq51.43%; A is smaller by just1,024 bytes. **That is not an arithmetic proof that exposed A time is the smaller term.** A0 removed both streams and their fill work together. A's136-byte row-group addressing, Xq reuse, cache residency, global transaction alignment and headers can make their exposed costs very different; the A5 lesson forbids substituting an isolated traffic model for in-model transfer.

Make the unsupported assumptions visible rather than choosing a favorable estimate:

- Let x be the fraction of baseline call time removable by the Xq-only fill change, η the fraction actually hidden, and p the added baseline-normalized cost of lower residency plus new scheduling. Net reduction is **r=η*x−p**. E≈25.7% bounds the combined removed-fill experiment; it does not measure x, η or p.
- Under an **optimistic byte-proportional sensitivity model**, x≈E*(18432/35840)*(1−1/(2G)), removing the unhideable initial Xq half. This is12.9369% for gate G20 and12.9668% for down G68. These are hypothetical component shares, not measurements or reliable upper bounds for each component.
- Even assuming η=80%, the entire savings budget is only10.35%/10.37%. Reaching8% then requires **p≤2.35/2.37 percentage points**. At p=5 percentage points the estimate becomes only5.35%/5.37%; at η=60%, the estimate is below8% **before** any residency cost.
- Moving the eight-wave CTA from two WGs to one halves runnable waves from16 to8. Two independent chains might partly compensate, but the A0 dependent/independent ratio0.43 used **four** chains and did not plateau. It does not bound the two-chain, one-WG penalty p to≤2.35%, nor prove η≥80% for cached activation loads. Do not multiply the candidate by2.34 or credit the1.19 LDS ratio as a speedup.

Thus(a) meets a paper register budget but **does not meet the requested condition to accept one WG/CU**: exposed A being the smaller term and enough net saving are not established. (b)/(c) fail the resource/headroom or mechanism screen. **Select(d): W2 rejected; projected admission credit0%/0 µs.** The hypothetical10.35% pre-penalty number is explicitly withdrawn as a product forecast. No new “Xq-only iteration2,” attribution task or performance timebox starts under this plan. Only genuinely new parent-supplied split-exposure/compute evidence and explicit reopening can alter this verdict.

## 6. Paper-only feasible alternative(a): ownership and numerical contract

This section completes the design evaluation so rejection is reproducible; it is **not** a composer implementation assignment. It costs no persistent state and would use the existing128×128 tile, grid `[M/128,N/128,1]`, block `[32,8,1]`, `sum[64]`,30,720-byte LDS, unchanged136-byte A loads/headers, and the same seven-argument ABI. A separately approved implementation would use an exact `#if defined(__gfx1151__)` body and distinct `*_full_{set,add}_w2_xq_gfx1151` symbols, never silently reuse the failed four-wave entry names with a different block.

One current Xq half remains in `tile_y`; nine next-half dwords/lane live privately in registers. No A half epochs or A replacement during h0 exist. Keep A current for both halves. All eight waves participate in every barrier.

| Transition | Before / action | After invariant |
|---|---|---|
| Prologue | Initialize sum64=+0; synchronously fill full A(0) and Xq(0,0); publish barrier. | A group0 and Y(0,0) are visible; no pending data is read. |
| h0(g) | Prefetch Xq(g,1) into private9 words; compute current h0 dot/fold with A(g),Y(g,0). No LDS writes during readers. | Pending is private; all outputs have exactly their h0 contribution. |
| h0 turnover | All-wave read-complete barrier; consume/wait pending Xq, store to Y, then all-wave publish barrier. | A remains g; Y becomes(g,1); no stale h0 reader survives. |
| h1(g) | If g+1<G, prefetch only Xq(g+1,0); compute h1 using unchanged A(g),Y(g,1). Final g performs no next load. | Both per-output folds of g complete in old order. |
| h1 turnover | All-wave read-complete barrier. If next exists, publish Xq and synchronously call the old full A loader for g+1, then publish barrier. | Both A(g+1) and Y(g+1,0) ready; **A fill remains exposed** and may also force an earlier global wait. |
| Drain | Final h1 read-complete barrier, no next publish. | Same unique Y set/add writer; no extra load beyond G, no global/persistent state. |

For G>0, prologue1 plus2G h0 barriers plusG h1-retire plusG−1 next publish =4G barriers, just as shipping. No barrier reduction is credited. Prefetch scheduling must survive emitted wait placement; loading Xq before current compute does not guarantee an outstanding memory operation survives through that compute.

For two-chain evaluation, retain shipping eight-wave row/column ownership. At each j0, pair n=0,1; for u=0..7 issue each output's WMMA in the same K16 sequence `(t=u/2,low/high=u%2)`, sharing B across the independent outputs. Two int32x8 accumulators plus three b64 current operands use16+6 registers. Fold n0 then n1 with the exact old f32 conversion/expression; each sum slot receives kb increasing, h0 then h1. No split-K, partial-dot recombination, changed signedness, reassociated fold, header bitcast substitution or FP16 accumulator. The same final set/add expression and force-blob ABI would remain mandatory (`/home/kaden/ClaudeCode/warpfront/wt-lloyd/kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip:185-210,264-287`).

A/Xq remain caller/scratch-owned read-only; one CTA owns its LDS and each destination element once. Prepared generation validation, quantization frequency, active-stream ownership, errors, capture/recording and retained state stay unchanged. This rejection creates no new state transition and claims no atomic rollback. Any reordered-arithmetic arm would require a separate numeric/state gate and explicit authorization; it is not a fallback for this rejected bit-exact W2.

## 7. Audited committed implementation: exact deletion boundary

Inspected with `git show HEAD:kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip` in the authorized worktree; inspected HEAD is **`baa6494cb09cf83783ab961ddefc10026e9a4097`**. Source is present, but that is not evidence it was launched or admitted. Current line anchors below are in `/home/kaden/ClaudeCode/warpfront/wt-lloyd/kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip`:

| Failed construct | Current source range | What to remove, not retune |
|---|---|---|
| Exact-gfx1151 W2 guard and description |299-303,581|Remove the entire self-contained guarded W2 section299-581; retain the enclosing RDNA branch's final `#endif` at584.|
| `prefetch_iu4_half_w2_gfx1151` |305-336|Both A_pending16 and Xq_pending18 loaders, `do_A/do_Xq` switches and four-wave ownership.|
| `publish_iu4_half_w2_gfx1151` |338-383|Half-A replacement/header conversion and Xq publication helpers. It publishes A headers before releasing Xq_pending; the original claimed lifetime should not be assumed from comments.|
| `vec_dot_i4_x128_w2_gfx1151` |387-454|The four-wave i0=64 mapping, fully unrolled j0/pair loops, two accumulators and sum128 indexing. Do not retain this as an alias/helper for(a).|
| `gemm_iu4_body_w2_gfx1151` |456-553|sum128/A_pending16/Xq_pending18 declarations at474-476, prologue at478-490, half-epoch pipeline at492-524, four-wave epilogue at526-551.|
| Two `*_full_{set,add}_w2_gfx1151` exports |555-579|Both `__launch_bounds__(128,2)` wrappers and their references.|

Do **not** delete or rewrite shipping `load_iu4_tile`, `vec_dot_i4_x128`, `gemm_iu4_body`, `IU4_ENTRY`, or the five incumbent entries at123-297. The shared activation quantizer and non-gfx1151 preprocessing remain untouched. The failed pipeline is not merely a dormant optimization that should be retained after rejection.

Host selection currently remains shipping: `/home/kaden/ClaudeCode/warpfront/wt-lloyd/crates/rdna-compute/src/gemm.rs:19065-19070,19106-19110` still selects `_occ3`/base with block `[32,8,1]`. **There is no observed production W2 route to revert.** Do not invent a host repair or change prepared consumers. If a concurrently prepared, not-yet-integrated route exists, its owner must discard it rather than landing it.

The formerly absent oracle now exists and is W2-specific: `/home/kaden/ClaudeCode/warpfront/wt-lloyd/crates/hipfire-runtime/examples/tmp_halo_iu4_oracle.rs:1-7,41-48,246-250` names the two failed entries and four-wave block; calls use that block at482-483,558-559,599-600,613-614. Its lab registration is `/home/kaden/ClaudeCode/warpfront/wt-lloyd/crates/hipfire-runtime/Cargo.toml:674-676`. On closure, remove that throwaway example/registration after Main archives its source and the failed object report. Do not leave an oracle that unconditionally ensures deleted symbols, or claim its output-parity gate passed because it compiled.

## 8. Composer-ready closure units and frozen ownership

The old W2.0 metadata-first unit is **completed with failure**. W2.1 correctness, W2.2 timing/routing and the proposed W2.3 second iteration are cancelled. The remaining bounded units are cleanup of the rejected experiment, not continued kernel development:

1. **W2.R0 — preserve evidence/stop execution (Main,≤15 min).** Archive the reported exact entry manifests, source/prelude/compiler/object identity and inspected commit. Record vgpr256/sgpr18/spill391/scratch1252, shipping190/21/0/0 and the corrected256 cap. Do not rerun the failed launch or reconstruct GPU corruption. Acceptance: failed artifact is explicitly non-executable for this campaign; no W2 timings/parity claimed.
2. **W2.R1-K — remove rejected HIP section (kernel composer,≤30 min; afterR0).** Own only `/home/kaden/ClaudeCode/warpfront/wt-lloyd/kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip`. Delete exactly §7's guard/helpers/entries, not incumbent code. Acceptance: shipping symbols/body remain; no W2 exports/helper references survive; no replacement performance candidate.
3. **W2.R1-O — remove rejected throwaway caller (oracle composer/H,≤30 min; independent ofR1-K afterR0).** Oracle composer owns the one example; H serializes `/home/kaden/ClaudeCode/warpfront/wt-lloyd/crates/hipfire-runtime/Cargo.toml` with other example owners. Delete only the W2 oracle and its `[[example]]` entry; preserve A0/F3 and unrelated diagnostics. Acceptance: no surviving caller/registration requests removed W2 symbols. Frozen interface: R1-K exports no W2 symbols; R1-O leaves no such consumers. These two ownership slices may execute independently.
4. **W2.R2 — integrate the rejection and final veto (H/Main/reviewer,≤1 h; after bothR1 slices).** Leave the observed shipping host selector/callers alone. H records the corrected256 ceiling and explicit rejection in the existing gap-plan W2 budget/ledger, so the invalid≤300 line cannot authorize a later composer. No new tests/probes or performance flags. Main owns any one shared no-GPU compile/registration check after sibling integration; composers run no builds/formatters/linters/suites. Reviewer confirms source boundary, unmatched references, baseline isolation and honest zero performance credit.

This planning assignment performs **none** of these source/manifest/deletion steps. It supplies their exact ownership and acceptance criteria. Closure has no performance experiment timebox to spend: do not consume the prior “two iterations/three days” allowance after a hardware-impossible first design. A future reopened experiment would need fresh explicit authorization, metadata first, and a maximum of two bounded iterations with the same≥8% in-model kill, not permission borrowed from this rejected arm.

## 9. Retained correctness/timing gates — not satisfied or scheduled

No direct numeric result, GPU smoke, profile or bench is claimed for W2. Preserve these requirements for any independently reopened bit-exact IU4 change; they do not turn rejection into a pending implementation:

- Exact shipping-versus-candidate bits on gate set `(17408,5120,512)` and down add `(5120,17408,512)`, with cloned nonzero varied add destinations reset outside timing. Immutable A/Xq and output guards unchanged. G=1/2/3 and signed Xq−8/+7, unsigned weight0/15, distinct half headers, nonzero sums, zeros and cancellation fixtures; prepared and freshly quantized real caller paths; params/force-blob ABI parity. Never run an A0 invalid no-load/no-fold twin as the reference (`/home/kaden/ClaudeCode/warpfront/wt-lloyd/crates/hipfire-runtime/examples/tmp_halo_iu4_calibrate.rs:320-475,512-567,686-727`).
- All eligible source shapes retain original output ownership and accumulation order; non-full tiles, gfx1100/other architectures, capture/recording, retained replay and prepared-generation semantics keep incumbent behavior. No numerical-tolerance escape hatch, split-K, or changed quantization recipe.
- Metadata must pass before any correctness launch, which must pass before100 interleaved direct gate/down event samples. Per-shape samples are not blended in-model symbol times.
- In-model `profile_prefill_qwen35` pp512 baseline context is set1688 µs×272, add1643 µs×128:400 calls and669.44 ms IU4 of approximately818 ms profile total. Require≥8% reduction in **each** family and weighted IU4 with unchanged counts/routes; historical thresholds set≤1552.96 µs, add≤1511.56 µs. Contemporary matched baselines, not these historical values, decide promotion (`/home/kaden/ClaudeCode/warpfront/wt-lloyd/docs/plans/2026-09-15-halo-gap-plan.md:319-328`).
- Production bench pp512/2048/8192, decode contexts128/2048/32768, three fresh-process ABBA cycles, positive wall transfer beyond paired spread and no>2% prompt/decode regression. Any future candidate below8% per-family in-model improvement after two bounded iterations is killed. Isolated gains cannot override a failed in-model gate.
- Direct bit evidence remains primary; common model-eval digest, actual serve decoded-text battery and retained-path isolation are supplemental product gates, owned by Main (`/home/kaden/ClaudeCode/warpfront/wt-lloyd/docs/plans/2026-09-15-halo-gap-plan.md:340-358`). None was earned by the failed metadata object.

## 10. Risks and actual planning evidence

**Biggest remaining risk in the sole register-feasible alternative:** A fill may dominate the exposed term, while losing a resident WG costs more than the≈2.35-percentage-point residual budget of the optimistic Xq model. This is exactly why(a) is not admitted from its232-register arithmetic. Source bytes and the non-plateau four-chain probe cannot resolve it.

Keep A5's rejection lesson: about−7% isolated but+4% in-model on400 back-to-back calls, with the contemporaneous CPU-build contamination noted. Retain row/column tile walk and original A addressing; do not claim A-stream locality from adjacency arithmetic (`/home/kaden/ClaudeCode/warpfront/wt-lloyd/docs/plans/2026-09-15-halo-gap-plan.md:139-152`).

LDS stride42 and Y stride18 must not be silently tuned. A2's bank-conflict result is evidence for its old reader/writer schedule, not proof that a new schedule is conflict-free (`/home/kaden/ClaudeCode/warpfront/wt-lloyd/kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip:48-51`). New planes require capacity and publication accounting; merely allocating39,936 bytes cannot hide global latency.

**APU rule remains absolute:** no cargo/CK/other CPU build on the Halo during GPU timing, including another checkout. Build/JIT must finish first; discard contaminated samples rather than normalizing against1688 µs (`/home/kaden/ClaudeCode/warpfront/wt-lloyd/docs/plans/2026-09-15-halo-gap-plan.md:143-152`). No timing is authorized in this revision.

CPU-only revision checks actually executed: all(a)/(b)/(c) register/headroom and LDS arithmetic above; eight-wave Xq mapping covers all2,304 dwords once with9 values/lane; sensitivity calculation gives10.3495%/10.3734% gross at the explicitly hypothetical80% hide rate and only2.3495/2.3734 percentage points of overhead allowance for8% net. Committed W2 source was read and the deletion boundary/symbols inspected. The original four-wave mapping/epoch checks remain only historical indexing evidence; they do not counter the measured spill failure. No source code, build, GPU gate or project validation was executed by this planning revision.

**Final verdict: W2 rejected, shipping retained, projected admission credit0%; reviewer owns final closure veto.**
