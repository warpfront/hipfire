# W2: peak-driven exact-gfx1151 IU4 prefill GEMM

Date: 2026-09-15, revision after the plateaued peak probe. Worktree: `/home/kaden/ClaudeCode/warpfront/wt-lloyd`, branch `mq4-lloyd`. **Plan of record for a new candidate, not admission or a measured kernel result.** This revision changes only this document. No source edits, builds, GPU runs, formatters, linters, or project suites were performed by the planning agent. Main owns compilation and every Halo gate; an independent reviewer owns the final veto.

> **STATUS 2026-09-15 (parent) — WS4 REJECTED after both bounded iterations.**
> Metadata: first object 256 VGPR / 391 spills (the ≤300 ceiling was above the
> 256 architectural cap); WS4 revision 256 / 124; after producer role-split +
> packetization 256 / 27; after pinning the fold before the barrier (ISA showed
> SelectionDAG sinking all four jbase folds past B0, 16 chains live) 216 / 0.
> Paired oracle on gfx1151 (100 interleaved): gate set −6.2 %, down add −7.8 %
> vs shipping — far from the ≥30 % (70 % of the 107.8 TOPS peak) gate — and
> 1-ulp f32 fold drift everywhere (the pin changed FMA contraction), so not
> bit-exact either. Producer waves at one WG/CU do not reproduce the probe's
> issue rate. Removed. Successor: W3 — independent chains inside the shipping
> 8-wave / 2-WG structure, ≤192 VGPR, fold macro untouched (local hipcc
> resource-usage workflow).

> **W3 (multi-chain in the shipping structure) REJECTED 2026-09-15:** mc4
> (4 chains, 188 VGPR, 2 WG/CU) −2.7 % gate / −5.1 % down; mc2 (noinline
> blocks, 151 VGPR) +14.5 % / +8.9 %. Issue latency is already hidden by the
> 16 resident waves. Mismatch counts were identical across WS4/mc2/mc4 ⇒ the
> drift is FMA-contraction re-decision in the shipping fold, not the
> candidates: any bit-exact restructure first needs the fold pinned with
> explicit `__fmaf_rn`/`__fmul_rn` matching shipping's current ISA. Candidates
> and oracle removed. Next lever (W4): 64×256 tile (same LDS, 2 WG/CU, weight
> re-reads 4×→2× per launch, fill per output halved).

> **W5 (in-register Xq prefetch inside the shipping structure) REJECTED
> 2026-09-15:** only 1 of 9 Xq dwords fits under the 192-VGPR 2-WG/CU cap
> (shipping is at 190); paired oracle ±1–3 % (noise). rocprof on the shipping
> kernel: SQ_WAIT_CNT_ANY 36 % (gate) / 47 % (down) of wave-cycles, barriers
> 2 %, LDS bank conflicts ~2 % ⇒ memory-latency-bound with no register room
> for more loads in flight; 1 WG/CU designs (WS4) lose more residency than they
> gain. GEMM track closed at this tier; the admitted GEMM change is A5 (grid
> order, gate L2 miss 37 %→ fixed), worth −9 % gate / −4.6 % down on the Halo.

## 1. Decision and immutable constraints

**Select WS4: eight compute waves with four independent output chains/wave, plus two loader waves; one320-thread WG/CU; two A planes and two Xq-half planes,61,440 bytes LDS.** Keep `sum[64]`, the128×128 output tile, original row/column tile walk, K256 group/fold boundaries and seven-argument ABI. Loader waves stream into planes not read by the compute waves; there is no pending global-load array in compute waves. Balance each steady-state loader phase as64 **whole weight rows** of the next group plus one128-column Xq half:17,920 logical bytes. Each group uses two producer/consumer rendezvous barriers, rather than four serialized fill/compute barriers.

**Target:** at least70% of the parent-measured107.8-TOPS IU4 ceiling on both production shapes, with the stricter concrete gates **gate set≤1200 µs, down add≤1180 µs**, bitwise shipping parity, zero spills/scratch. Conditional planning projection: **78–84 TOPS**, approximately**1087–1170 µs/call** on either full shape. These are unmeasured targets, not promised results; all performance credit remains zero until gates pass.

The previously implemented four-wave `sum[128]`/34-register-prefetch W2 remains **rejected**: parent metadata is256 VGPR,18 SGPR,391 reported spills,1252 scratch bytes, versus shipping190/21/0/0. The architectural hard cap is **256 allocated VGPR per wave32 lane after descriptor rounding**, not300; the new **whole-kernel design/object target is≤232 including headroom**, zero spills/scratch. A CU residency quotient never permits exceeding the per-wave256 limit. The failed296-register premise cannot be repaired by calling the capped256 object a pass. Delete its distinct guarded helpers/entries as specified in §8; do not reuse its four-wave body.

The prior rejection of all W2 performance work was based on missing peak evidence and an Xq-only framing. **Main's new plateau measurement explicitly reopens W2 under this different contract.** It does not validate the failed body or restore the old≤300 budget. The earlier≤300 ceiling in `/home/kaden/ClaudeCode/warpfront/wt-lloyd/docs/plans/2026-09-15-halo-gap-plan.md:84-100,388` is superseded by256/target232 here. Accepting one WG/CU is intentional: independent compute chains and producer overlap, not increased residency, are the mechanism.

Non-goals: tile/grid swizzles; weight/activation format or quantizer changes; split-K, reassociated folds, reduced-precision accumulators; T0 fusion, S1 multistream scheduling, F2 chunk/cadence changes, F3 attention work; new architecture/model/capture/replay/retained routes; compiler register-limit workarounds or scratch-backed state. FA2 remains independent H24/KV4/D256 work. No performance credit is borrowed from another campaign unit.

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
| Registers/occupancy | Source records190 VGPR/zero spills from A2; Main now independently reports190 VGPR,21 SGPR,zero spills/scratch for the shipping full entries. Full set/add choose `_occ3`, tails choose the base entry. Physical two-WG residency, not three, follows the §6 resource arithmetic; object allocation granularity and actual launch LDS must still be archived (`/home/kaden/ClaudeCode/warpfront/wt-lloyd/kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip:48-51,291-297`; `/home/kaden/ClaudeCode/warpfront/wt-lloyd/crates/rdna-compute/src/gemm.rs:19065-19070`; `/home/kaden/ClaudeCode/warpfront/wt-lloyd/docs/plans/2026-09-15-halo-gap-plan.md:78-100`). |

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

These callers keep their argument/sidecar contracts and migrate through the single private selector only after admission; there is no new per-family API. LSP returned empty definition/reference results despite visible calls, reported as a worktree/server mismatch. This inventory was recovered with source search/read. H must repeat references with a worktree-attached server before implementation and account for new sibling changes.

## 3. New measured peak and its actual implications

Parent-measured on exact gfx1151, block `[32,8,1]`, loop4096, three-sample medians, plateau across grids320/640/1280/2560/5120 WGs:

| Probe structure | TOPS | VGPR | Spill | Implication |
|---|---:|---:|---:|---|
| IU4, four independent chains plus LDS operand reads |**107.8** (329 wave-WMMA/µs/CU)|41|0|Four chains suffice for the measured issue ceiling.|
| IU4, eight independent chains plus LDS |107.0|73|0|Doubling accumulator count adds no measured peak throughput.|
| IU8, four independent chains plus LDS |54.6|45|0|Not an attractive replacement for native IU4.|
| F16, four independent chains plus LDS |55.3|55|0|No reason to abandon the bit-exact IU4 representation.|

Probe source has four int32x8 accumulators with one LDS-fed IU4 issue each per iteration, initialized LDS and no timed writes (`/home/kaden/ClaudeCode/warpfront/wt-lloyd/crates/hipfire-runtime/examples/tmp_halo_iu4_calibrate.rs:157-231`). Peak sweep/counting uses8192 operations per **wave-level**16×16×16 WMMA,40 CUs, the five grids above and event medians (`:1193-1215,1236-1296`). This revision accepts Main's symbol-qualified measurements; it does not rerun them. Source at`:1320-1327` currently swaps the local variables for the IU4-eight-chain and IU8 summary labels; use the correctly named per-symbol rows at`:1314-1318` and Main's supplied values, not that final mixed-label summary. The issue was reported to Main; no source edit here.

A0's old194 wave-WMMA/µs/CU≈63.6 TOPS at grid160 was **occupancy-limited parent context, not the ceiling**. Likewise dependent/independent=69.5/162.5≈0.43 and LDS co-issue=194/162.5≈1.19 describe that underfilled operating point. They diagnosed the one-chain weakness but cannot cap the redesigned kernel at63.6 TOPS or multiply its speed by either ratio.

The two full production shapes `(17408,5120,512)` and `(5120,17408,512)` each contain **91,268,055,040 dense integer operations**, using2*M*K*N. At107.8 TOPS the issue-only time is846.64 µs;70% is75.46 TOPS and1209.49 µs. The requested concrete gates are slightly stronger:1200 µs gate-set=76.06 TOPS/70.55% peak;1180 µs down-add=77.35 TOPS/71.75% peak. Main's shipping gate1697 µs is53.78 TOPS, about49.9% peak. Do not call the blended in-model set1688 µs a measurement of that one gate shape.

A0's combined exposed-fill bounds still matter:25.8% gate/25.4% down, nofold5.2%/3.0%. They come from stale-data no-load and checksum-fold twins, not independent A/Xq time decompositions (`/home/kaden/ClaudeCode/warpfront/wt-lloyd/crates/hipfire-runtime/examples/tmp_halo_iu4_calibrate.rs:581-733,1332-1404`). Now the opportunity is **jointly** four-chain WMMA/LDS issue, fewer synchronization boundaries and overlapped fill. It is not confined to hiding the Xq share of25%. Independent issue still does not eliminate the f32 fold, header conversions, bank effects, finite grids or loader contention. The low-register41-VGPR probe is not proof that a232-VGPR production kernel sustains the same rate with only one WG.

## 4. Explicit alternatives against the four-chain peak structure

Use190 shipping VGPR as an empirical allocation starting point, not a mathematical compiler model. Keep sum64 wherever possible. Four live int32x8 chains replace one: **+24 registers**. Reusing two A and two B b64 fragments for a2×2 output-subtile group needs8 fragment words, no more than the current source bundle. Thus190+24=214, leaving18 planning registers before a232 target. Actual complete-object metadata is decisive.

| Design | Resource arithmetic and comparison to peak probe | Verdict |
|---|---|---|
| Eight compute waves, **two** independent chains, existing single LDS |190+8≈198 plus18 headroom→216;30,720 B; one WG once allocation exceeds192. Half the four-chain probe's independent C state. Four global fill/compute barriers remain. |Useful structure to cost, but not selected: two chains are not established sufficient at the reduced residency.|
| Eight compute waves, **four** independent chains, existing single LDS |190+24≈214 plus18→232; one WG; same four-chain count as the peak probe but exposed fills/four barriers remain. |Necessary arithmetic component, insufficient overlap design.|
| Eight chains/wave |190+56≈246 **before**18 headroom;264 exceeds256. Peak107.0 is no better than four-chain107.8. |Reject extra state; no performance justification.|
| Eight waves, four chains, **Xq-only LDS double buffer**, fill both halves before compute, no barrier between h0/h1 |A21,504+2*Xq9,216=**39,936 B**, one WG/CU;≈232 target. Prefill A/Xq0/Xq1→barrier→dot/fold0→dot/fold1→barrier. The absence of any writer between halves makes the middle barrier genuinely unnecessary. |Safe bit-exact alternative, two barriers/group, but **all global fill remains exposed**. An extra LDS plane is not asynchronous copy. At best the new issue roof plus an unchanged≈0.43-ms fill term is already≈1.28 ms before other overhead, above the target; this is a screening model, not an exact decomposition. Not selected.|
| Eight total waves, four-chain compute, **full61,440-B LDS** but all eight still both fill/compute |One WG,≈232 if no long-lived pending global payload. Matches chain count but no dedicated issuer during compute. Loading the next plane in a serial phase does not overlap the current phase. Retaining next payload in those compute registers raises pressure. |Full LDS alone does not solve the phase problem. Not selected.|
| Eight total waves with dedicated loaders, e.g.4 compute+4 loaders |Compute sum128;190+64+24=278 before headroom. A6-compute/2-loader asymmetric ownership would also need a fresh proven mapping and worst-wave budget; none is supplied. |Reject four-compute shape; do not advertise eight total waves by hiding extra output ownership.|
| **Selected:8 compute+2 loaders, four chains, two A/two Xq-half planes** |320 threads,61,440 B, **one WG/CU**. Compute sum64;190+24+18=**232 whole-kernel target**. Loader phase has70 raw dwords/lane, budgeted below the same allocation; role-dependent limits are not assumed. Two rendezvous/group. |Four-chain issue as in peak probe, genuine concurrent global fills/LDS writes in other waves, stable current read planes. First complete-object gate must pass.|
| Larger K512 slab per barrier |Two A256 groups require43,008 B; four Xq128 halves require36,864 B: **79,872 B**, already above64 KiB for complete preload. Double-buffering A512 plus only two Xq halves needs104,448 B. A512 with only two Xq halves uses61,440 B but still needs half turnover/publication; it is not four ready Xq halves or a free barrier reduction. |Keep K256 group,128-half fold order. No unsupported K512/1024 preload or reassociation.|

Prior requested register-prefetch variants remain unattractive against the corrected peak: eight-wave Xq-only with18 conservative pending registers plus four chains is190+18+24+18 headroom=250 (rounding may consume256); coalesced9-word Xq gives241 before rounding, above232. Four-wave sum128 plus only Xq18 is296 before headroom with four chains; only A16 is294; no-prefetch four-wave four-chain is278. These are not replacements for the selected producer-role split. The failed sum128/34-prefetch/two-chain296 estimate stays permanently rejected.

**Why one WG can now plausibly beat the incumbent two:** selected residency supplies eight compute waves total, approximately four compute waves/SIMD with balanced placement, each with four chains:16 independent output chains/SIMD, plus a loader wave/SIMD. That is at the low end of the parent's approximately4–8 resident-wave/SIMD sufficiency window for independent issue. The incumbent has more runnable compute waves but serial per-wave C dependencies and no load/compute role overlap. The41-VGPR peak sweep is not a direct occupancy-controlled proof for232 VGPR/61 KiB: the target-shape timing gate must settle this trade. Do not infer role placement or issue rate from an occupancy formula alone.

## 5. Frozen selected WS4 interfaces and ownership

### 5.1 Source, symbols, launch and private state

Kernel owner adds the new implementation only under **`#if defined(__gfx1151__)`** in `/home/kaden/ClaudeCode/warpfront/wt-lloyd/kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip`, replacing/removing the failed section rather than keeping two W2 variants. Existing macros and baseline helpers/entries stay unchanged. Freeze unique symbols:

- `gemm_mq4g256v2_residual_mmq_iu4_full_set_ws4_w2_gfx1151`
- `gemm_mq4g256v2_residual_mmq_iu4_full_add_ws4_w2_gfx1151`
- Internal `gemm_iu4_body_ws4_w2_gfx1151`, `vec_dot_i4_x128_ws4_w2_gfx1151`, `load_iu4_stage_ws4_w2_gfx1151`.

Same flat arguments `(A,Xq,Y,M,K,N,add)`, same pointer-array/blob packing. Entries use `__launch_bounds__(320,1)`, block `[32,10,1]`, unchanged full-tile grid `[M/128,N/128,1]`, dynamic LDS61,440 B. No new public API or module-registration copy; the existing source concatenation carries the symbols. No non-Halo no-op exports. Only full, positive M/K/N with existing K%256 contract are eligible.

Waves0..7 compute, wave id w unchanged from shipping: `i0=32*(w/2)`, column offset=`16*(w%2)`. Local output coordinates and64 sum indices remain exactly §2's mapping. Waves8..9 are producers; loader id e=`threadIdx.y−8`∈{0,1}, lane0..31. Producers never perform WMMA, fold, or Y output stores; consumers never fetch A/Xq globally in steady state. Each physical wave still receives the **same whole-kernel VGPR allocation**. Put role-specific work in wave-uniform branches; all ten waves rejoin every workgroup barrier. No early producer return, lane-local barrier, per-role launch bound or dynamic register repartitioning.

### 5.2 Four-chain consumer, exact per-output order

For each jbase∈{0,64}, maintain four `int32x8` C values indexed by `(c,n)` with c∈{0,1}, n∈{0,1}; column subtile j=`jbase+32*c`. These are four existing independent output subtiles, **not four K partial sums**.

For u=0..7 in increasing order, load A b64 for n0/n1 at K-half offset `16*h+2*u` and B b64 for c0/c1 at `2*u`. Four b64 fragments =8 lane dwords. Issue `(c0,n0),(c0,n1),(c1,n0),(c1,n1)` consecutively, sharing A/B fragments. Each chain receives exactly the old `(t=u/2,low_or_high=u%2)` WMMA sequence, unsigned A/signed B and unchanged flags. Four WMMAs now have four independent C destinations. Source needs four b64 loads/four WMMAs, versus the probe's intended two b64 loads/WMMA and shipping's four per two; actual instructions/bank transactions must be inspected.

After u7, fold `(c0,n0),(c0,n1),(c1,n0),(c1,n1)` using the **same** header conversion, d/s reads, f32 multiply/add expression and sum index `((j/16)+n)*8+l`, l0..7. Each sum sees g0h0,g0h1,g1h0,g1h1…; next jbase is independent. Keep at most four live C vectors and current fragments; no arrays of accumulators for both jbase groups, K unrolling into persistent operands, or dynamic-indexed scratch. There are still64 WMMAs/compute wave/half,512/CTA/half,1024/CTA/group. Do not bring the diagnostic probe's periodic accumulator mask/checksum into GEMM.

### 5.3 Exact LDS planes and producer addresses

| Plane | Bytes | Contents/ownership |
|---|---|---|
| Y0 |`[0,9216)`|Xq half0 of current group, then next group;128×18 dwords.|
| Y1 |`[9216,18432)`|Xq half1 of current group;128×18 dwords.|
| A0 |`[18432,39936)`|128 weight rows×42 dwords; parity0 group.|
| A1 |`[39936,61440)`|128 weight rows×42 dwords; parity1 group.|

A stride168 B, header offsets32..39, pads40..41 and Y stride72 B are unchanged. The A double buffer is real; Y0/Y1 are separate128-K halves reused at explicit ownership boundaries, **not two complete K256 activation groups**.

For a producer stage b∈{0,1} filling rows64*b..64*b+63 of target A group g into the selected free A plane:

- Loader e/lane loads q=0..31 with `r=64*b+2*q+e` and payload word=`lane`.
- Global dword is `A+136*((row0+r)*(K/256)+g)+8+4*lane`; LDS destination=`Aplane[r*42+lane]`.
- Metadata row is `r=64*b+32*e+lane`; load both32-bit headers at gp+0/+4. For ksc0/1 preserve the old f16→f32→half2 conversions and four copies at `r*42+32+4*ksc+v`, v0..3. Pads remain untouched.

Each payload wave therefore reads a full contiguous128-byte packed row, **not** the failed four-wave design's64-byte half-row. One stage's A traffic is64×136=8,704 bytes:8,192 payload+512 headers. It covers64 whole rows of both K halves; b is a **row-half**, not a K-half.

Xq stage target `(gx,hx)` starts at `Xq+((2*gx+hx)*N+col0)`. Each loader copies q=0..35 with `l=64*q+32*e+lane` to the chosen Y plane. All2,304 dwords have one writer, full coalesced64-thread sweeps. Xq traffic is9,216 B/stage. Steady-state total=**17,920 B/phase**, equally balanced between h0 and h1 when next A exists.

Loader staging maximum: `a_code[32]`, `a_header[2]`, `xq_words[36]` =**70 raw dwords/lane**. Stage the applicable global bytes before conversions/stores for loader MLP, then publish into the already-free destinations. No second pending bundle or global scratch; no call to the shipping eight-wave `load_iu4_tile` from loader wave ids8/9. A is optional in the final group's stages; an absent target is not dereferenced. Prologue may use the same bounded loader helper with Xq disabled for its second row-half. All payload arrays must scalarize, and the complete whole-kernel object must pass §6.

### 5.4 State machine: producer/consumer overlap and every barrier

Let G=K/256>0, current A parity p=g&1 and next parity p^1. A/Xq global inputs remain immutable; each CTA owns all four LDS planes and its output tile. Only the producer role writes LDS. All compute waves finish every current-plane read **and fold** before relinquishing ownership.

| Transition | Compute waves0..7 | Loader waves8..9 / after invariant |
|---|---|---|
| Prologue | Initialize sum64=+0; no LDS read. | Fill A0 group0 rows0..63 and64..127, plus Y0=(0,0). Y1/A1 invalid and unread. All ten waves execute publish barrier Binit. |
| h0(g) | Read complete A[p]=g and Y0=(g,0); execute four-chain dot/fold0. | Write **other A plane** rows0..63 for g+1 if next exists, and Y1=(g,1). Y1 was retired at the prior group's end, or never read on first group. No writer touches A[p]/Y0. |
| Barrier B0(g) | All h0 reads/folds finish. | All staged stores finish. All-wave barrier simultaneously releases Y0 and publishes Y1. Next A plane top64 rows may now be ready, but bottom64 are not promised. A[p] remains current for h1. |
| h1(g) | Read unchanged A[p]=g and Y1=(g,1); execute four-chain dot/fold1. | If next exists, fill other A plane rows64..127 and Y0=(g+1,0). Y0 is free only because B0 completed. On final group, no global next-group loads/stores. |
| Barrier B1(g) | All current A and Y1 reads/folds finish. | All next-group stores finish. For next group, other A plane is now fully g+1 and Y0=(g+1,0); old A[p]/Y1 become free. All ten waves participate even on final g. |
| Advance/drain | Increment g and swap parity, or epilogue after final B1. | No producer pending state survives drain. Producers do not write Y output. Compute lanes perform the same unique final set/add store. |

Count=**1+2G workgroup barriers**:G1→3, G20→41, G68→137, versus shipping4G (4/80/272). There is **one necessary rendezvous between halves**, not zero: Y1 is being produced during h0. The no-middle-barrier, preload-both-halves alternative in §4 is safe but leaves global fill exposed. Do not claim that variant's zero barrier while using this producer pipeline. B0/B1 each serve both publication and reader retirement because producers write **disjoint** planes during compute; unlike the failed in-place scheme, a separate pre-store barrier is unnecessary.

Warmup: never read uninitialized A1/Y1. Last group: h0 still produces current Y1, but neither phase accesses A(g+1), and h1 produces no next Y0. No out-of-bounds speculative prefetch, partial-WG escape or half-initialized A swap. G1/G2/G3 are mandatory oracle cases. The loader may finish early and wait at the barrier; that is bounded ownership, not proof that fill is hidden. Every wave must follow the same g loop/barrier count regardless of role.

## 6. Whole-kernel cap, headroom and claimed residency

Architectural cap **256 allocated VGPR**, zero spill/scratch. Design/object target **≤232**, with actual allocation granularity applied; a used count232 rounding to240 misses the target and must be reduced before launch. Do not weaken the hard256 ceiling or resurrect≤300. The old failure256/18/391/1252 is rejected regardless of reported occupancy.

For the shipping block256/eight-wave kernel, two WGs need eight waves/SIMD: allocated VGPR≤192 and2×30,720=61,440 LDS. Three WGs need92,160 LDS and cannot reside. Thus `_occ3` is not physical three-WG occupancy; retain the observed190-VGPR shipping control and round its allocation from the actual descriptor. WS4 deliberately trades those two eight-wave WGs for one ten-wave producer/consumer WG.

Compute-path planning ledger: measured baseline190 +24 for four versus one C vectors +18 explicit compiler/address/control headroom =**232**. Current fragments remain8 dwords. Named compute state64 sum+32 C+8 fragments=**104 lane scalars**, not a compiled104 count; the ledger retains incumbent overhead rather than assuming it disappears. The41-VGPR peak core plus64 sum values illustrates why four chains are much cheaper than the failed doubled-output layout, but **41+64 is not an allocation prediction**: GEMM fold/addressing and compiler live ranges remain.

Producer ledger:70 staged raw scalars; allow16 for address/current-source temporaries,8 for conversion/store scratch,10 for counters/control =104, plus18 headroom→122. For conservative whole-CFG accounting, even if64 sum registers remain live through a producer branch, this paper path totals186, below232. These are sizing inventories, not separate role allocations. Compiler predication, unrolling or cross-branch liveness can still exceed them: every lane in the320-thread kernel gets one measured allocation, and only the complete object counts.

With the campaign CU model2 SIMD/CU/1,536 lane-register units/SIMD/64 KiB LDS, a232 allocation allows `floor(1536/232)=6` wave slots/SIMD; a ten-wave WG needs5/SIMD under balanced distribution.61,440 LDS permits **one WG/CU**. Expected eight compute waves plus two producers approximately split4C+1P per SIMD; this is a capacity/placement model, not observed runtime role residency. No co-resident second IU4 WG or FA2 overlap is claimed. Launch bounds must be320,1, not256,3 or128,2.

**First complete-artifact gate, before correctness execution:** archive exact source/prelude/options/object identity, per-entry VGPR/SGPR counts/spills, scratch/private segment, fixed+dynamic group segment, descriptor allocation granularity, kernel ABI offsets/size, block geometry and ISA. Require≤232 allocated (hard architectural256),zero all spill/scratch,exact61,440 B,one-WG capacity. Missing spill fields are unknown. All eight compute waves must use four independent C chains, correct IU4 signedness/count, no f32 fold DAG change and no local-memory arrays. Loader role must actually issue global loads and LDS writes on the non-reading planes while compute executes; a source-level branch does not prove throughput or hide an accidental early global wait.

A compile-only scaffold with fewer chains, absent loader arrays or dead math is not a passing metadata result. No correctness/timing launch of a spilled candidate. Main owns object compilation/inspection; composers do not run shared validation or any Halo gate.

## 7. Numerical, host and cross-architecture invariants

Every individual output receives the identical eight K16 integer WMMA steps per K128 half with old flags, zero integer initialization, nibble bits, and half metadata. Four independent chains interleave **different outputs**, never fragments of the same output. The nibble-domain128-term dot has absolute bound15,360, but even an overflow-free alternative order is not authorized. Preserve h0 then h1 and increasing group order for each FP32 sum; same casts, f16 conversion path, multiply/add expression, compiler contraction/fast-math settings and final `Y += v` versus `Y = v`. No numeric-gate variant is required. A bit mismatch is a stop, not a reason to change tolerance or import a probe mask.

A/Xq stay caller-owned or scratch-owned read-only; same valid prepared generation enters the same-stream GEMM and stays live until existing stream completion. Producers only write CTA-local planes, consumers only their uniquely owned Y elements. There is no global scratch, sidecar generation bump, extra quantization, frame reservation, KV/GDN/conv state mutation, cross-stream dependency or atomic rollback promise. Baseline behavior for tails/unsupported inputs and device faults remains unchanged.

After explicit admission, H modifies only `Gpu::gemm_mq4g256v2_mmq_prequant_iu4` to select WS4 for exact gfx1151, positive M/K/N, full128-aligned M/N, existing K%256 validity, and `!self.replay.is_recording() && !self.graphs.capture_mode`. Symbol, **block** and **dynamic LDS** are selected together:WS4 `[32,10,1]`/61,440 B; incumbent `[32,8,1]`/30,720 B. Grid/tile walk stays `[M/128,N/128,1]`. Keep profiling the selected symbol and both params/force-blob launch forms. gfx1100, tails, recording/capture, other rejected architectures and unchanged callers retain incumbent routes. No new environment flag, public unchecked launcher, quantizer copy or model-family API. Source registration remains the existing concat in §2.1.

## 8. Replace the failed body and freeze composer slices

The failed committed implementation was inspected with `git show HEAD:kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip` at **`baa6494cb09cf83783ab961ddefc10026e9a4097`**. Current source still contains it. Kernel owner removes its exact standalone gfx1151 section299-581 in `/home/kaden/ClaudeCode/warpfront/wt-lloyd/kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip` before inserting WS4:

- `prefetch_iu4_half_w2_gfx1151`:305-336;
- `publish_iu4_half_w2_gfx1151`:338-383;
- four-wave `vec_dot_i4_x128_w2_gfx1151`:387-454;
- `gemm_iu4_body_w2_gfx1151`, including sum128/34 pending words and old epilogue:456-553;
- both failed `*_full_{set,add}_w2_gfx1151` exports:555-579, closing guard581.

Retain shipping helpers/entries123-297 and enclosing final RDNA `#endif`584. No alias to the old names. The host selector is currently still shipping at `/home/kaden/ClaudeCode/warpfront/wt-lloyd/crates/rdna-compute/src/gemm.rs:19065-19070,19106-19110`; do not invent a production W2 rollback.

The current oracle exists and names the failed symbols/block (`/home/kaden/ClaudeCode/warpfront/wt-lloyd/crates/hipfire-runtime/examples/tmp_halo_iu4_oracle.rs:1-7,41-48,246-250,482-483,558-559,599-600,613-614`). Its lab registration is `/home/kaden/ClaudeCode/warpfront/wt-lloyd/crates/hipfire-runtime/Cargo.toml:674-676`. **Update/reuse it for the new unique WS4 symbols,320-thread block and61,440 LDS**, retaining separate baseline256-thread/30,720 values; do not delete the shared baseline oracle machinery or retain calls to removed entries. Compile-only must still stop before allocation/launch/timing. No passed numeric result is claimed for the old body.

Frozen ownership: K owns only the IU4 HIP file; O owns only the oracle; H exclusively owns `/home/kaden/ClaudeCode/warpfront/wt-lloyd/crates/rdna-compute/src/gemm.rs` and any necessary manifest/plan/ledger integration. H serializes shared-file changes with siblings; no kernels.rs, scratch, quantizer, attention or model-caller edits are needed. Main owns **all** compilation/object/GPU gates; composers skip builds/formatters/linters/project suites. K and O prepare independently against §5's frozen symbol/ABI/launch contract; no overlapping file edits.

1. **W2.P0 — complete WS4 and metadata-only artifact (K,≤4 h).** Archive failed-body evidence, replace it with the complete guarded WS4 kernel, including all four chains and live producer stages. Shipping code untouched, no product route. Main compiles only the exact production-prelude objects, inspects §6 before any launch and checks non-Halo exclusion. Acceptance: resource/ABI/ISA gate on both final-body set/add entries, not an empty skeleton.
2. **W2.P1 — oracle/epoch integration (O,≤3 h; development may overlapP0).** Replace old symbols/block/LDS with the frozen WS4 contract and preserve baseline raw launcher. Cover §9's direct bits/guards, G1/2/3 and same-stream prepared/unprepared cases; compare exact per-output values, not a checksum. Main runs only afterP0. Acceptance: all bits/guards and input immutability pass; no retained stale W2 exports/callers.
3. **W2.P2 — target-shape and model gate, first bounded iteration (H≤2 h; Main≤4 h; afterP0/P1).** Use direct prebuilt oracle for100 paired gate/down samples, then an isolated candidate binary's private exact-gfx1151 selector for pp512 profiles and required bench matrix. Names/block/LDS/timer routes match. Acceptance: **both direct shapes≥70% peak and≤1200/1180 µs**, plus in-model and parity gates below. An isolated speed win is not admission.
4. **W2.P3 — one specific correction or kill (K/O/H as owned,≤4 h; Main≤3 h).** Only one measured/ISA-identified loader packet, producer issue ordering or current-fragment live-range correction; same8C+2P,4 chains,planes,barriers,ABI and bitwise arithmetic. No sum128, extra resident buffers, eight-chain arm, larger-K preload, no-middle-barrier claim, swizzle, numeric relaxation or register-cap increase. Repeat complete metadata→oracle→same shape/model gates. If goals remain unmet after this second bounded iteration, reject WS4 and stop.
5. **W2.P4 — admission, cleanup and independent veto (H/Main/reviewer,≤3 h).** Only after actual gates pass, keep the one admitted exact route, record per-entry metadata/TOPS/µs and measured wall transfer, remove obsolete failed symbols/scaffolding, and correct the existing gap-plan≤300 budget/ledger through H. If rejected, remove WS4 routing/candidate calls rather than retaining a dormant flag. Reviewer owns final veto even if numerical timing thresholds pass.

**New authorization timebox: three working days,≤26 active hours total, at most two WS4 scheduling iterations.** The failed four-wave artifact is historical, not an unmeasured iteration of this new design. Missing metadata, unresolved spill or unchanged issue schedule stops before execution. No extra performance idea is launched because the timebox has unused hours.

## 9. Claim-scoped validation and numerical/timing gates

### 9.1 Metadata then exact bits, before timing

Main first applies §6 to every final-body entry. Then direct shipping-versus-WS4 oracle on gate set `(17408,5120,512)` and down add `(5120,17408,512)`, identical A/Xq, disjoint cloned Y with varied nonzero add starts. Reset add Y before **every** timing repetition outside the interval. Compare all f32 bits, immutable A/Xq and guards. Do not run no-load/no-fold diagnostic twins as numerical references.

Required boundaries: G1/G2/G3 with K256/512/768; distinct top/bottom row-half payload and both K-half headers to detect partial-A publication; distinguish every group/column, unsigned weight0/15, signed Xq−8/+7, nonzero exact s, zeros and cancellation. Poison unused/opposite LDS via a diagnostic fixture if helpful, but correctness must not depend on zero initialization. A0's narrow filler ranges are not sufficient coverage (`/home/kaden/ClaudeCode/warpfront/wt-lloyd/crates/hipfire-runtime/examples/tmp_halo_iu4_calibrate.rs:772-826`). No barrier test can be only a source-text assertion.

Cover full existing set shapes `(M,K)=(1024,5120),(6144,5120),(10240,5120),(12288,5120),(17408,5120)` and add `(5120,5120),(5120,6144),(5120,17408)`, N128/256/512 and1024 only if its host envelope is admitted. Include real freshly quantized and prepared sidecars without new producer/quantizer calls, force-blob/params parity, unchanged non-full tails, gfx1100 and capture/recording route isolation. Unsupported symbols are never launched as an architecture test. A mismatch blocks timing; save its exact coordinate/bytes and repair without changing the mathematical contract.

### 9.2 Direct peak-utilization gate and conditional projection

After exact bits,100 interleaved event samples per target shape after matched warmup; report raw min/median/max and paired spread, object identity and exact add/set behavior. Compute TOPS as`2*M*K*N/(event_us*1e6)` and utilization asTOPS/107.8, not by dividing the lane count twice or including quantizer operations. **Pass both≥70% and concrete1200/1180 µs gates.** Failed allocation never reaches this gate.

Selected conditional target78–84 TOPS =72.36–77.92% peak =**1170.10–1086.52 µs per shape**, approximately31–36% below the measured1697-µs direct gate baseline. The add comparison must use a contemporary direct down-add baseline;1643 µs is a blended in-model family number, not automatically that shape's direct baseline. No peak-rate result is being claimed for WS4.

Reasonableness model, not attribution proof: the selected pipeline must hide most of the≈25.7% exposed-fill bound and improve dependent compute issue, rather than assigning all improvement to registers. Hiding80% of that bound saves≈20.6% of old time; saving a further≈10% of total time from four-chain issue/barrier reductions would produce≈31% total, near78 TOPS. These fractions interact and are not independent measured credits. Do not add the nofold bound, peak ratios, or another campaign's savings. Finite-grid/prologue/loader/VALU time must fit the remaining~0.24–0.32 ms above the846.6-µs issue-only floor. If it does not, abandon the projection.

One balanced phase carries512 compute WMMAs and17,920 producer logical bytes. A roughly2-µs steady-state phase requires≈256 wave-WMMA/µs/CU (below329 peak) and≈8.96 logical GB/s/CU producer throughput. Across40 CUs that is358.4 logical GB/s; **not DRAM bytes** because A/Xq cache reuse is substantial but unmeasured here. This is a feasibility requirement and a warning about loader pressure, not a claimed bandwidth measurement. Gate has544 WGs/G20, down160 WGs/G68; single-WG residency and finite grid tails/prologues must appear in actual target-shape measurements, not just an infinite-loop rate.

### 9.3 In-model transfer, product parity and kill

Run prebuilt `profile_prefill_qwen35` pp512, warmup3, Q8 KV, matched ABBA baseline/candidate and unchanged accepted predecessor for unrelated F/G work (`/home/kaden/ClaudeCode/warpfront/wt-lloyd/docs/plans/2026-09-15-halo-gap-plan.md:319-328`). Baseline context: set1688 µs×272, add1643 µs×128;400 calls,669.44 ms IU4 of≈818 ms profile total. Archive new per-symbol counts/total/average, and producer/quantizer counts. A renamed WS4 symbol must not disappear into “other.” The same call/shape mix and prepared path are mandatory.

**In-model floor remains≥8% improvement in each set/add family and weighted IU4**, no family regression hidden in the average. Historical thresholds are1552.96/1511.56 µs; contemporaneous matched baselines decide. This floor is **additional to**, not a replacement for, the new70%-peak/1200/1180 direct target. A candidate that merely clears8% while missing the new target is not the requested W2 deliverable.

Run production bench pp512/2048/8192 and decode contexts128/2048/32768, three fresh-process ABBA cycles (six samples/arm), positive wall transfer beyond paired spread and no>2% regression in any required row. No gain≤spread is admitted; one extra cycle only within the same timebox. **After at most two iterations, failure of either direct target, either family floor, bits, resources or wall transfer kills WS4.** Do not claim a70% kernel result from an in-model blended symbol average.

Main then runs common one-chunk model-eval digest, actual serve deterministic decoded-text battery, and retained-path isolation gates (`/home/kaden/ClaudeCode/warpfront/wt-lloyd/docs/plans/2026-09-15-halo-gap-plan.md:340-358`). Direct output bits remain primary; aggregate eval digest is supplemental. Shared no-GPU validation runs once after accepted sibling integration, not by each composer and never while taking Halo numbers. No new retained/capture behavior is claimed.

## 10. Risks and planning evidence

**Largest risk:** eight compute waves at232 VGPR and61 KiB may not sustain four-chain issue at the needed rate once real strided LDS, FP32 folds and concurrent producer writes contend for issue/banks/bandwidth. The41-VGPR read-only peak probe does not measure that contention. A producer that misses the~2-µs phase budget turns the rendezvous into exposed fill; more LDS then buys no speed. Direct target timing, not a source schedule, is the veto gate.

The complete-object cap is equally strict: no separate low-VGPR producer allocation, no dead-code metadata proxy and no spilled “it might still win” arm. Unexpected compiler liveness must be fixed under the232 target/256 architectural ceiling before any launch, or the unit stops. Four-chain source ordering may still serialize issued WMMAs or hoist fragment lifetimes; inspect the actual object.

Keep A2 stride42/Y18 and ordinary full128-byte wave-wide A row loads. Four simultaneous output chains can change LDS bank/port pressure; A2's bank result does not certify the new pattern (`/home/kaden/ClaudeCode/warpfront/wt-lloyd/kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip:48-51`). Full next-A planes preserve current data, but speculative publication of only top64 rows must never cause a group swap before bottom64 arrive. All ten waves must reach Binit/B0/B1, including final drain.

**A-stream locality across400 calls:** A5 was about−7% isolated and+4% in-model; retain the contemporaneous CPU-build contamination caveat and do not treat its mechanism as proven. Keep the original block/tile walk; within-CTA next-group lookahead can still alter cache behavior and must transfer in-model (`/home/kaden/ClaudeCode/warpfront/wt-lloyd/docs/plans/2026-09-15-halo-gap-plan.md:139-152`). Do not revive column-adjacent, macrotile, larger-K or weight-layout experiments under WS4.

**APU measurement rule:** no cargo/CK/other CPU build of any checkout on the Halo during timing. Build/JIT first, reach matched warm thermal state, archive device/CU identity/clocks/power context/model/source/object/flags/run order, and discard contaminated samples. Do not normalize a throttled arm using1697 or1688 µs (`/home/kaden/ClaudeCode/warpfront/wt-lloyd/docs/plans/2026-09-15-halo-gap-plan.md:143-152`). No GPU run was performed by this planning agent.

CPU-only checks actually executed for this revision: both64-row producer stages each cover2,048 payload dwords and64 two-header rows once; Xq producer mapping covers2,304 dwords once; A/Y epoch simulation for G1/2/3/20/68 preserves all current reads and reports1+2G barriers (3/5/7/41/137); four-chain grouping gives each incumbent output subtile exactly its old ordered eight K16 WMMAs. Whole-kernel214+18=232,61,440 LDS,17,920 bytes/phase,70 raw loader scalars, operation counts/TOPS/utilization/time thresholds and projection were recomputed. These establish indexing/epoch/arithmetic consistency only, not compiled bit parity, zero spills, runtime role placement, actual producer overlap or achieved TOPS.

After actual smoke/oracle/model admission, H updates this plan and the existing gap ledger/changelog with measured evidence and removes failed/unused scaffolding. Until then WS4 is an unmeasured, metadata-first candidate with explicit abandonment criteria; reviewer veto remains final.
