# gfx1201 iu4: slab issue/wait model and one bounded scheduling slice

2026-09-18. Planning/evidence only. No production source changed, no GPU launched, no other process touched. Baseline: `wt-iu4`, branch `gfx12-iu4-k32`, assignment pin `fd0969955`, fragment-order iu4 K32 kernel. The candidate below is not implemented or performance-validated.

## 1. Decision and claim boundary

**Largest counted phase:** the per-128-K fold, including its metadata reads, supporting scalar instructions and B2b, is 407 issued instructions/wave, approximately 188.9 us of the whole gate/set in the explicit 2.93-GHz serial-issue model. Its arithmetic alone is 307 VALU packets, not the old kernel's 315 and not the probe's normalized 256. WMMA arithmetic contributes another 256 service cycles/wave/trip, approximately 118.8 us.

**Top removable scheduling bottleneck [INFERENCE]:** three next-block global-fetch epochs are serialized with essentially no useful compute between load and first wait. The alleged next-A/compute1 and next-W/fold overlap in the source comments does not occur in the observed ISA. Choose one bounded change: **batch the next block's A/W/DS/SZ raw fetches immediately after B1, execute current slab1 compute and the unchanged current fold, then publish next state between B2a/B2b.** Keep all four workgroup barriers, tile geometry, byte layout, integer chains, RN fold DAG and stores. This targets exposed memory latency, not imaginary WMMA/VALU coexecution or a new format.

The issue model accounts for **410.30 us** at nominal 2.93 GHz (393.61 us steady-loop issue plus 16.69 us epilogue issue), leaving **94.70 us / 18.75% explicitly unexplained** against 505 us. That residual includes memory/dependency/barrier waiting, prologue, first/last-trip differences, clock uncertainty and dispatch/tail effects; it is **not a measured staging duration**. Acceptance permits an explicit residual: this plan does not fit an invented barrier cost to reach 505 us. At 2.7 GHz the same calculation is 445.25 us, leaving 59.75 us / 11.83%. Scalar/memory issue overlap also makes this an accounting model, not a cycle-accurate simulator.

## 2. Evidence, authorities and exact artifacts

Abbreviations:

- `KERNEL`: `/home/kaden/ClaudeCode/warpfront/wt-iu4/kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx12.hip`.
- `F`: `/home/kaden/ClaudeCode/warpfront/wt-iu4/scratch-2026-09-17/iu4frag/`.
- `E`: `/home/kaden/ClaudeCode/warpfront/wt-lloyd/.codeinsight+research/scratch-2026-09-17/Iu4Sol/`.
- `P`: `/home/kaden/ClaudeCode/warpfront/wt-lloyd/.codeinsight+research/scratch-2026-09-17/wmma_probe/`.
- `ORACLE`: `wt-iu4/crates/hipfire-runtime/examples/tmp_iu4_gfx12_oracle.rs`.

Observed kernel SHA256: `90ccba37923fe6e73e303bd77165b55cdae7ad0c39dc478de8e1688ce2d9f10b`.
Observed `F/isa-new.txt` SHA256: `d30f3ebc8af05c497ee942f3973f9cd34857c026c063930dc0c8875e242c0e09`.

`F/RESULTS.md`, `oracle-new2.log`, `time-new-r2.log` and `isa-new.txt` are the worker's baseline receipts: 9/9 bit-identical, 183 VGPR, 40 SGPR, zero spills/scratch, occupancy API result 3, and gate/set 504.8 us in the second new-layout run. The first new-layout run was 497.4 us; old-layout baseline was 484.2 us. Fragment-order placement was neutral/slower by 1–5%, not a demonstrated optimization. The source and this ISA are the same fragment-order generation; the hashes prevent accidental comparison with a concurrently changed checkout.

CPU-only proof executed: `python3 E/analyze.py` (expand E). It parses the existing JIT disassembly, follows the full-tile epilogue CFG, verifies phase sums, reproduces the arithmetic below, checks all 512 staging b64 slots against consumer byte coordinates, and models 40 next-buffer generations. Outputs: `E/analysis.json`, `E/baseline-steady.isa`, `E/baseline-epilogue-fulltile.isa`. This is analytical proof, not GPU synchronization or oracle proof.

An additional CPU-only compile on a COPY of the probe was run in E:

```text
hipcc --genco --offload-arch=gfx1201 -O3 -save-temps fold_probe_copy.hip -o fold_probe_copy.hsaco
```

No executable was run. Probe-copy SHA256: `8513406412e60e212bc342f509d3f40d98e09b2f6afcb49d3e76cc52f361e5ed`. Relevant artifacts: `fold_probe_copy-hip-amdgcn-amd-amdhsa-gfx1201.s`, `probe-L-loop.s`, `probe-L-census.json`.

Primary authorities:

1. [AMD RDNA4 ISA Reference Guide, 7-April-2025](https://docs.amd.com/api/khub/documents/uQpkEvk3pv~kfAb2x~j4uw/content): §§2.1/2.3 wave32 and four SIMD32s/WGP; §5.6 split workgroup barriers; §§5.7/5.8 dependency counters and ALU delays; §7.8 VOPD; §7.12 integer WMMA; §12.1 LDS banks.
2. [AMD Matrix Instruction Calculator](https://github.com/ROCm/amd_matrix_instruction_calculator/blob/main/matrix_calculator.py#L2568-L2588), retrieved with GitHub `file_read`, version 1.3.2: RDNA4 `v_wmma_i32_16x16x32_iu4`, `cycles=8`, `coexec=False`, `coexec_delay=-1`.
3. [LLVM GFX12 execution synchronization](https://llvm.org/docs/AMDGPUExecutionSynchronization.html#gfx12) and [execution-barrier memory model](https://llvm.org/docs/AMDGPUUsage.html#amdgpu-amdhsa-execution-barriers-memory-model): execution barriers alone do not publish memory; named/subset barriers are GFX12.5+, not gfx1201.
4. [LLVM AMDGPU scheduling model](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/SISchedule.td): one ready instruction packet/cycle is the throughput-model convention, not permission to charge five dependency-latency cycles to every independent FP32 operation.
5. `P/coexec_RESULTS.md`: measured WMMA/VALU mixtures are additive or worse on gfx1201; no independent-pipe overlap to exploit. Its LDS-fed measurements are comparisons, not a matched production-phase decomposition.

LDS throughput here follows the ISA's 32 DWORD banks per classic CU, one read/one write per bank per clock: **128 B/clock/CU per direction**, or 256 B/clock/WGP if balanced across both sides. Cross-side accesses/bank conflicts can lower it. **600–800 cycles global latency is the assignment's sensitivity assumption**, not a latency measured for this kernel; cache hits need not pay DRAM latency.

## 3. Normalize geometry, occupancy and the probe before multiplying

Gate/set is M=17408, K=5120, N=512:

- Grid = `[136,4,1]`, **544 workgroups**; block = 256 threads = **8 wave32s**.
- One wave owns 32 rows ×64 tokens = eight 16×16 output fragments = **64 output values/lane**.
- `IU4_BK=64` is a physical staging slab. Each **128-K fold trip comprises TWO such slabs**. There are 40 fold trips /80 physical slabs per workgroup, not 40 physical 64-K slabs.
- Four K32 steps/trip ×eight output fragments = **32 WMMAs/wave/trip**, 256/workgroup/trip, 5,570,560 WMMAs/kernel.
- Total math = **91,268,055,040 operations =91.26805504 GOP**, counting multiply and add separately. 505 us corresponds to 180.73 TOPS. At 277 TOPS the equivalent time is **329.49 us**, not 300 us: the raw difference is 175.51 us.
- R9700 topology confirmed by Iu4Frag from `rocminfo`: **64 classic CUs, two SIMD32/CU =128 SIMD32**, grouped into 32 WGPs. `profiler.rs::hip_mp_count_to_cu_count` explicitly converts HIP's RDNA WGP count to classic-CU count.
- The worker clarified that the occupancy receipt's “3 WG/CU” is **3 WG per HIP reporting unit/WGP** here. Use **24 resident waves/WGP, nominally six waves/SIMD**, not twelve. The occupancy API receipt is authoritative for this launch; do not infer residency from `__launch_bounds__`, raw VGPR count, or a generic LDS-size table alone. The ISA describes 128 KiB physical LDS/WGP, while the reported launch occupancy is still 3; a 64-KiB allocation-budget shorthand is not proof of the physical capacity.
- Grid work is 4352 waves /128 SIMD =**34 waves/SIMD across the launch**, or **1360 wave-trips/SIMD**. At 96 concurrently resident WGs this is 5.667 occupancy-sized batches; the last 64 WGs do not fill 96 slots. This is throughput accounting, not a claim of lock-step waves or a fixed tail duration.

### Probe scope correction

`P/fold_probe.hip::k_L` uses the requested one-A-plus-four-W fragment shape, and the same source-level `fold_i32` RN DAG. Nevertheless, its **emitted work is not identical** to production:

| Per normalized 32 WMMAs | Production full_set | Copied k_L, normalized from 16 WMMAs |
|---|---:|---:|
| Fold scalar operations | 388 | 256 |
| Fold issue packets | 307 | 256 |
| FP32 multiplications | 192 | 64 |
| i32→f32 conversions | 68 | 64 |
| FP32 FMAs / adds | 64 /64 | 64 /64 |
| Fold DS metadata packets | 20 | 0 |
| Raw VGPR | 183 | 79 |

In k_L, `sc*d` is loop-invariant and hoisted (`v20`, assembly4051), `zp*d=0.125` and `s=3` are constants. The production scales vary by row and half-block. k_L's actual loop `.LBB10_8`, assembly4081–4370, has 290 packets: 16 WMMAs, 128 fold packets, 20 b64 DS loads, 48 other VALU and 78 scalar packets. Production shares each activation fragment across TWO row groups as well: 24 logical b64 fragment reads per32 WMMAs, versus normalized k_L's40. Thus even the read count and register-residency regime differ. The 277-TOPS measurement remains valid; treating its time as the literal production K-loop cost is not justified. No replacement GPU performance result is claimed by this offline census.

## 4. One-wave, one-128-K-trip issue and wait model

### 4.1 Complete dynamic-path census

Use `full_set` at `isa-new.txt` function base0xE000. A representative **non-last** trip spans **0xE5B8..0xF2D8 inclusive**, with an outer-g rollover every second trip. It contains **616 instructions**. A linear cutoff at0xF2C4 omits the two A-publish stores at0xF2C8/0xF2D0 and the backedge0xF2D8. Fold appears at the beginning in address order because the first entry jumps over it; semantically it closes the previous compute trip.

| Category | Emitted packets/trip | Meaning |
|---|---:|---|
| WMMA |32|32×8=256 shared arithmetic service cycles|
| Exact fold VALU |307|388 scalar operations; 81 VOPD packets save81 issues|
| Other VALU |44|Addressing, predicates, masks, header conversion; not free|
| DS fragment reads |12|5 `ds_load_2addr_b64` +7 `ds_load_2addr_stride64_b64` =24 logical b64 reads|
| DS metadata reads |20|`ds_load_2addr_b32`:16 SZ pairs +4 token DS pairs|
| DS publishes |8|2 dual-b64 stores +4 b64 stores +2 b32 stores =8 logical b64 +2 b32 writes|
| Global loads |10|8 b64 nibble loads +2 b32 header/DS loads|
| Global invalidations |4|`global_inv scope:SCOPE_SE` after the four barrier waits|
| Scalar/control/wait/delay |179|Includes4 signal+4 wait,21 DS waits,9 load waits,3 combined load/DS waits,34 ALU waits,53 ALU-delay packets; remainder51 scalar/control packets|
| **Total** |**616**|**840 service-cycle accounting units** after replacing each WMMA's1 issue with8 service cycles|

Fold arithmetic is exactly68 conversions (64 C +4 reused token sums),192 multiply,64 FMA,64 add. Its 307 packets consist of226 singletons +81 dual packets. The 44 non-fold VALU include10 low+10 carry address additions,7 NC additions,11 cndmasks (8 activation dwords,2 metadata zero masks,1 control predicate),2 integer comparisons,2 64-bit multiply-add address operations,1 shift and1 fp16→fp32 conversion. Address hoisting does **not** mean zero loop address instructions: the emitted global pointers still use per-load 64-bit add/carry pairs.

There is no separate array-copy instruction stream for `A_pf`/`W_pf`: the ordinary registers are direct VMEM destinations and later DS sources. Four b64 slab1 loads hold8 dwords/lane; next-A holds4. The traffic cost is loads, late/early masks, publication and register lifetime, not eight extra copies by definition.

### 4.2 Cycle-level phase table

Convention: each non-WMMA encoded instruction costs one issue accounting unit; WMMA costs8. It deliberately includes scalar waits/delays as issued packets **but not the cycles spent waiting**. Scalar/VMEM/DS resources can overlap; these columns must not be described as disjoint measured wall times. The per-wave/trip → whole-kernel factor at nominal2.93GHz is `1360/2930=0.464164 us`. The 128-K trip consists of the slab0 and slab1 rows below; neither individual64-K slab has its own fold.

| Phase (ISA inclusive range) | Instructions | Service-cycle units/wave/trip | Whole-kernel us @2.93GHz | Wait/ownership note |
|---|---:|---:|---:|---|
| B2b +current fold +DS/SZ reads, E5B8–EDB0 |407|407|188.91|307 arithmetic +2 address VALU;20 DS; metadata deps; final publish acquire|
| Inner-trip control, EDB8–EDBC |2|2|0.93|Outer rollover separately below|
| Slab1 prefetch and pointer setup, EDC0–EE78 |32|32|14.85|4 b64 global loads,9 VALU,19 scalar|
| Slab0 compute, EE84–EF64 |32|144|66.84|16 WMMA,6 dual DS,4 early cndmasks,2 early global waits|
| B0, EF6C–EF84 |7|7|3.25|Global drain; all slab0 A reads finish before overwrite|
| Publish slab1, EF90–EF98 |2|2|0.93|2 dual DS stores place A and W1|
| Next-A “prefetch,” EFA0–F034 |26|26|12.07|2 b64 loads **immediately waited**,4 masks|
| B1, F03C–F050 |6|6|2.78|Combined global/DS drain; publishes slab1|
| Slab1 compute, F05C–F11C |27|139|64.52|16 WMMA,6 dual DS,1 address VALU|
| Next-W direct stage, F124–F1C4 |30|30|13.92|2 b64 loads **immediately waited/stored**, after compute1|
| Next DS/SZ stage, F1CC–F2A0 |36|36|16.71|2 b32 loads **immediately waited/converted/stored**, after W waits|
| B2a, F2A8–F2C4 |6|6|2.78|All slab1 A reads finish before overwrite|
| Next-A publish/backedge, F2C8–F2D8 |3|3|1.39|2 b64 writes to A; next executes B2b|
| **Steady subtotal** |**616**|**840**|**389.90**|No stall duration assigned|
| Outer-g rollover, E574–E5B4, every other trip |16/2|8|3.71|All scalar; boundary approximation|
| **Loop accounting total** | |**848**|**393.61**|40-trip steady-state approximation|
| Full-tile SET epilogue |1438/wave|35.95 amortized/trip|16.69|16 barriers/wave;64 b32 global stores|
| **Accounted total** | |**883.95**|**410.30**| |
| **Unexplained residual to505us** | |**204.03 equivalent units**|**94.70**|Not fitted to a named phase|

The epilogue CFG was followed under true M/N bounds, not counted by summing mutually exclusive text blocks. It contains492 VALU,770 scalar/control,96 DS (32 dual b32 writes +64 b32 reads),64 global stores and16 global invalidations. It writes64KiB/WG =35.65MB/kernel. Full_add additionally reads Y/adds; it cannot inherit SET's epilogue census unchanged. Prologue and exact first/last-trip corrections remain in the residual; the last trip skips next-block loads and stores. The source has **1 prologue +160 loop +16 store =177 barrier signal/wait pairs/wave** over this K, not80 total.

Clock sensitivity, without inventing a measured under-load clock:

| Assumed GHz | Loop issue us | Epilogue issue us | Residual to505us |
|---:|---:|---:|---:|
|2.40|480.53|20.37|4.10|
|2.60|443.57|18.80|42.63|
|2.70|427.14|18.11|59.75|
|2.93 nominal boost|393.61|16.69|94.70|

A close sum at an assumed frequency is not validation of that frequency or of serial issue. The definitive missing information is a same-dispatch effective clock plus normalized hardware stall/resource counters. SQ_BUSY alone is not a stall-reason decomposition.

### 4.3 What the actual waits protect, and what they accidentally serialize

- **B0:** `s_wait_loadcnt 0` atEF6C, signalEF70, waitEF80. A is single-buffered; all consumers must finish A/slab0 before A/slab1 overwrite. Outstanding slab1 VMEM also happens to drain. W1 itself does not require a read-before-overwrite fence here.
- **B1:** combined `s_wait_loadcnt_dscnt 0` atF03C, signalF040, waitF04C. Publishes A/slab1 and W1. Crucially next-A globals atEFFC/F008 have already been forced complete byF014/F028 for cndmasks, then this combined wait closes any remaining window. They cannot overlap compute1 atF078 onward.
- **B2a:** combined drainF2A8, signalF2AC, waitF2B4. Protects A/slab1 read completion before next A/slab0 publish. Next-W globalsF198/F1A4 wait/storeF1B4..F1C4, then metadata globalsF244/F260 wait/convert/storeF26C..F2A0; this is a serial W→metadata chain after compute1, **not W overlapping fold**.
- **B2b:** combined drainE5B8, signalE5BC, waitE608. Publishes next A/W0/DS/SZ. Current fold conversion begins between signal/wait; DS reads follow the wait. There are only a few overlappable conversions here, not the whole307-packet fold. Current metadata parity remains separate from next metadata writes.

`__builtin_amdgcn_sched_barrier(0)` is a compiler scheduling fence, not another workgroup barrier. The current four calls in `iu4_bundle` do not create four extra `s_barrier` pairs. Conversely, deleting a full workgroup barrier removes its waits/invalidation and changes legal load motion as well as execution rendezvous: a no-barrier marginal is not the intrinsic cost of `s_barrier_wait`.

A useful latency sensitivity is `extra cycles/wave-trip ≈ Σ max(0,L_j-H_j)/R_eff`, with L the actual cache/memory completion latency, H useful issue service while that epoch is in flight, and R_eff the ready/resident-wave hiding factor (at most nominal6 here, often smaller at a rendezvous). Three completely uncovered600–800-cycle epochs divided by6 would add300–400 normalized cycles, or139–186us. The nominal residual is only94.7us, so **do not charge three DRAM misses to every trip**. Some loads hit caches, other work hides waits, and issue classes overlap. This sensitivity identifies a schedule defect; it does not measure its exact cost.

### 4.4 LDS/global bandwidth sanity checks

Per WG/per128-K trip:

- Global requested bytes:16KiB nibbles +1KiB DS +1KiB SZ/header requests =**18KiB**, including duplicate packed-header reads by the two sc/zp lanes. Whole kernel401.08MB, about794GB/s requested at505us; this is not DRAM traffic. Unique input footprints are47.35MB weights and1.475MB Xq, repeatedly reused across tiles; Y is35.65MB.
- DS fragment reads:24 logical b64/wave ×32 lanes ×8B ×8waves =**48KiB/WG**. At256B/clock/WGP this is192 WGP cycles, or96 normalized SIMD-service cycles/wave-trip. It is below563 cycles of WMMA+fold arithmetic.
- Metadata read payload before broadcast is40KiB/WG, but SZ values broadcast across16 lanes and token DS repeats across k halves. Unique bank-serviced requests are approximately6KiB/WG:24 WGP cycles /12 normalized wave-trip cycles; instruction/dependency overhead remains real.
- Slab b64 producer writes touch16 DWORD banks with4 requests/bank, versus32 banks with2 requests/bank for contiguous fragment reads. Thus the bank-service minimum is4 cycles/b64 producer instruction, **2× the b64 width floor**, not4×. The source's “8 banks ×4” describes starting banks only. Eight logical b64 writes +two b32 writes/wave give approximately68 normalized wave-trip write-service cycles including the bank distribution.
- These LDS throughput floors are **resource alternatives to the issue model**, not extra numbers to add blindly to it. Fragment reads+metadata are about108 normalized cycles; writes about68; even a conservative shared-service sum176 is well below arithmetic563. Raw LDS byte bandwidth is not the leading standalone capacity bound in this layout.

## 5. Reconciliation with Iu4Frag's completed ablations

Receipt: `F/attrib2/ATTRIB2-RESULTS.md`, received/read2026-09-18; raw logs `F/attrib2/brackets/<tag>-{a,b,c}.log`. Exclusive ordinal1, one runner JITs each TU, three fresh-process brackets each containing10 warmups and five20-launch batches. The table uses median-of-three bracket medians. The same-session full is **500.6us**, not the earlier504.8/505us; this4.4us difference is normal bracket context, not a candidate win. A marginal is `(full − variant)`, never an additive phase duration.

| Variant | Three bracket medians, us | Median | Full−variant | JIT VGPR | Reconciliation |
|---|---|---:|---:|---:|---|
|full|500.6 /498.9 /503.8|500.6|—|183|Control consistent with505us model target|
|no-stage|456.5 /461.1 /457.8|457.8|+42.8|163|Nibble stage removal helps8.55%; agrees staging is material, **not** evidence of200us staging time|
|no-fold|447.6 /448.1 /446.6|447.6|+53.0|105|Much less than modeled142.5us fold arithmetic; added serialized sink and resource changes prevent direct subtraction|
|no-compute|381.1 /378.7 /376.0|378.7|+121.9|117|Close in scale to118.8us WMMA arithmetic, but also removes24 fragment reads and restructures fold; not independent confirmation|
|no-store|523.7 /524.6 /526.6|524.6|−24.0|166|Contradicts a naive positive epilogue subtraction; replacement sink/scheduling confounds mean “tail cover” is not causally proved|
|no-barrier|449.9 /446.6 /447.0|447.0|+53.6|179|Far above bare signal/wait issue cost; agrees drains/rescheduling matter; also removes epilogue/prologue synchronization and creates races|
|sync-stage|499.0 /499.8 /499.3|499.3|+1.3|188|**Strong agreement:** old prefetch machinery has no resolved timing benefit; matches observed immediate waits|
|single-W-buffer|504.3 /511.1 /509.6|509.6|−9.0|183|Does not justify deleting W1; baseline buffer helps or variant worsens scheduling/races|

Reported spills are zero throughout. Worker symbol census retains32WMMAs except no-compute0; full-symbol b64 global-load count12→4 in no-stage (prologue included); signal/wait-instruction count42→10 no-store and42→0 no-barrier. These are **static full-symbol counts**, not dynamic177-pair launch counts. Fold conversions are68 normally,0 no-fold, and12 no-compute. Thus the corrected no-compute input is defined, but56 C conversions disappeared or moved/reused because its artificial C values are constant across trips. It is not a WMMA-only matched intervention. No variant-specific occupancy measurement was supplied, so equal occupancy is not assumed from common LDS or zero spills.

### 5.1 Checks and disagreements that change interpretation

- Initial no-fold/no-store allowed dead-code elimination. Worker repaired them with full-value checksum sinks before these timings. Likewise the initial no-compute undef input was replaced with a volatile-derived defined vector, and the sync-stage generator's missing next-A replacement/early-A-clobber was repaired. These repaired sources, not the first objects, underlie the table.
- **No-fold sink cost is not sub-microsecond.** Independent inspection of `F/attrib2/tu-3-nofold.dis.txt`, full_set0xCBC4..0xCD38, finds **one `v_xor_b32` plus31 `v_xor3_b32` per trip**, not a single static XOR representing a rolled loop. Those32 VALU packets alone normalize to14.85us at2.93GHz, before dependency delays. The sink ends with **`flat_store_b32 scope:SCOPE_SYS` at0xCD2C and immediate `s_wait_storecnt 0` at0xCD38 every trip**. The report's original “1 XOR, rolled64 iterations, sub-us” characterization misses `v_xor3` and is rejected. This was sent back with addresses.
- Iu4Frag subsequently acknowledged the32-packet XOR chain, system-store/drain and aliasing corrections, and reported amending its attribution report:53us is a contaminated fold-removal net, negative no-store is an observed combined effect, and208us is not an additive reconciliation. The disagreements here document the initial interpretation, not a dispute over the timing observations.
- Both checksum variants write `((volatile ...*)Y)[tid]`: **all544 workgroups target the same1KiB**, unlike disjoint production tiles. The added globally scoped store/drain, write contention/races and altered VGPR allocation are confounds. The no-store slowdown proves only that this intervention is slower; it does not prove the original epilogue costs negative time or uniquely identify tail-drain cover. No correctness or shipping inference is made from these race-bearing controls.
- No-stage retains DS/SZ staging; it removes nibble staging, not every global load. Its second W plane lacks a defined initial value unless seeded. No-barrier is race-bearing; single-W-buffer may race next W0 stores against slower current readers. They are attribution interventions, not candidate implementations.
- **Do not reconcile by adding42.8+53.6+an arbitrary residual**, or by calling121.9+53.0 a measured175us compute+fold phase. Those interventions overlap and alter schedules/resources; the no-fold sink itself serializes. This plan therefore disagrees with those additive/causal readings in the initial worker report while preserving every measured number.

The absolute issue accounting remains563 arithmetic service cycles/wave/trip (261.32us at2.93GHz), with fold+metadata the largest individual modeled phase. The marginals say **no isolated measured non-compute removal yielded100us**, so the early “three full DRAM epochs dominate everything” hypothesis is too strong. The selected schedule is now a conservative latency-hiding experiment: removing actual nibble staging yielded42.8us, removing all synchronization yielded53.6us, and making staging synchronous was neutral. Expect a **modest, unmeasured15–50us target range**, not a promised200us recovery; even15us must survive same-session gating. The model's0–95us residual opportunity is not a forecast or additive upper bound on the candidate.

### 5.2 Counter units and clock reconciliation

Worker databases: `F/attrib2/counters/full_both_results.db`, `nostage_both_results.db`;110 dispatches, `SQ_BUSY_CYCLES` and `GRBM_GUI_ACTIVE`. Reported medians:

| Control | Dispatch wall | SQ_BUSY sum/dispatch,32 instances | SQ per-instance | GUI_ACTIVE,1 instance |
|---|---:|---:|---:|---:|
|full|490.2us|39.92M cycles|1.247M cycles|1.265M cycles|
|no-stage|445.8us|36.57M cycles|1.143M cycles|1.163M cycles|

Ratios no-stage/full: SQ0.916, GUI0.919, wall0.910. This agrees with a roughly9% shorter workload while the shader engines remain occupied. **SQ_BUSY counts cycles with a wave present, not issued VALU work**: proportional busy time cannot distinguish stalled resident waves from useful issue, prove identical occupancy, or assign the53.6us synchronization marginal to hardware rendezvous.

The report observes a2350MHz DPM table maximum but derives2.6–2.9GHz from GUI/wall. GUI/full wall gives approximately**2.58GHz if active for the whole interval**; assuming2.93GHz instead gives approximately88% active. Those are two unknowns related by one equation, **not an independent measured2.93GHz clock**. Accordingly §4 retains its explicit frequency sensitivity and94.70us residual, rather than choosing a clock to close the model. At the ablation control500.6us the same nominal410.30us accounting leaves90.30us unexplained. The counter experiment does not license a more precise stall split.

## 6. Why this one schedule change, not the other levers

| Option | Quantitative opportunity / cost | Decision |
|---|---|---|
| **One next-block fetch epoch after B1, consumed after compute1+fold** | Removes up to two exposed serial fetch epochs and provides128 WMMA +307 fold arithmetic cycles of useful work before use; no extra global/LDS bytes, no new barrier primitive | **Selected bounded hypothesis.** Nominal residual leaves0–95us unassigned, not a promised200us. After ablation reconciliation, use an unmeasured15–50us target range; stop below the measured3% gate. |
| Double-buffer A as well as W | Adds4096B used LDS:16384→20480; host19456 must change. Could reduce4→2 barriers/trip if ownership is redesigned correctly; full drains alone may still kill prefetch | Next independent experiment only if the chosen schedule fails for a measured barrier reason; not bundled into this slice. A+W full128-K double-buffering instead costs36864B including metadata, a different residency regime. |
| Increase staged K to256 | Two independent128-K folds **must remain** because scales/zero-points change by half. Larger planes increase LDS and often reduce residency; barrier count does not fall merely by changing a macro | Reject as first change. `IU4_BK` alone cannot change the hardcoded two K32 steps per64-K slab or half-metadata semantics. |
| Stream W directly into fragment registers | CurrentW:4 global b64/wave/trip,4 logical b64 DS writes,8 logical b64 DS reads. DirectW:8 global b64/wave/trip,0 W DS reads/writes; W requested bytes double8→16KiB/WG/trip (+178.26MB/kernel) because two token-half waves reuse each weight. Source W is read once per WG only in the cooperative staged route | Saves LDS work that is not the leading capacity bound, adds VMEM requests and scattered row-stride transactions. Not selected without measured cache/global-load proof. |
| Producer/consumer wave specialization | 8 current waves already own64 outputs/lane; fewer compute waves need more accumulators or changed geometry. Named subset barriers and dynamic per-role VGPR allocation unavailable on gfx1201 | Not a bounded byte-schedule fix here. Shared WMMA/VALU issue is still shared; role separation is not coexecution. |
| Different wave count | 4 same-area waves require128 float +128 integer accumulator registers/lane before operands;16 waves can lower pressure but changes all maps, barriers and store slots (20480B) | Wider independent geometry experiment; not needed to prove the observed serialization defect. |

## 7. Composer contract: exact next-block fetch/publish reschedule

### 7.1 Exact symbols and call sites

Sole production writer owns `KERNEL`:

- `IU4_FOLD_RN`194–202: **unchanged**.
- `iu4_bundle<SB,FIRST>`214–246 and calls437/438/468/469: same four calls, first-zero then ordered three accumulations; no operand/map changes.
- `gemm_mq4g256v2_residual_mmq_iu4_gfx12_body<ADD>`251–613: only prefetch/publication/fold placement within415–565 changes. Preserve setup, prologue367–399 and epilogue569–613.
- Entries `gemm_mq4g256v2_residual_mmq_iu4`617 (runtime add, body calls624/626), `_full_add`631 (call637), `_full_set`641 (call647): same7-argument ABI and256-thread launch.
- `crates/rdna-compute/src/gemm.rs::Gpu::gemm_mq4g256v2_mmq_prequant_iu4`19167: gfx1201 arm19193, symbol selection19199–19202, grid19226/19228, block/LDS19233–19238. Set/add wrappers call it at19357/19369. **No host change.**
- `crates/rdna-compute/src/kernels.rs::GEMM_MQ4G256V2_RESIDUAL_MMQ_IU4_GFX12_SRC`3471–3474 concatenates quantizer prelude+this TU. No copied production TU or second dispatch entry.
- `ORACLE::launch_iu4`169–195: same grid/block/LDS; `cpu_ref`198 onward unchanged; `time_case` calls490–499; occupancy502–511; nine oracle cases515–529.

LSP references were attempted against the absolute sibling-worktree Rust path and returned no references despite the visible two same-file callers; the tool limitation was reported. Exact searches/read recovered the above. No exported symbol is being changed. Any scope expansion to host/API/geometry requires a correctly rooted LSP reference pass first.

### 7.2 Frozen data and lifetime interface

Use existing `u32x2_t A_pf[2]`, `W_pf[2]` registers for raw next-block nibble words after their slab1 publication dies. Add only two scalar raw32-bit metadata holders: one token DS word and one packed weight header word per lane. Total next-state payload **10 dwords/lane**:4 A +4 W +1 DS +1 header. Names may remain local; there is no exported helper/struct/API requirement.

Keep immutable SGPR/lane address contracts: `st_avoff`, `st_wvoff`, `st_ldsoff`, `st_aok`, current `g/h/kb`, next `g+h`, next `h^1`, next `kb+1`. Clamp all addresses exactly as before; zero selects happen on the loaded values at publication, not by changing source addresses. Preserve raw header bit extraction and fp16→fp32 conversion exactly.

No new accumulator bank: `cacc[4][2]` and `acc[4][2]` retain64+64 registers/lane. Expected extra long-lived payload is at most the nextW4 +metadata2 relative to current nextA-only window, while many C values die through the fold. **[INFERENCE] target183–192 VGPR; hard stop above200 raw VGPR, any scratch/spill, or occupancy below the observed3 WG/WGP.** Allocation-rounded VGPR and live peak, not these estimates, decide. LDS remains16384B used /19456B passed; SGPR growth and actual occupancy must be recorded. No increased LDS budget is authorized for this slice.

### 7.3 Transition-by-transition ownership

| Transition | Before | Action / allowed owners | After / invariant |
|---|---|---|---|
| Prologue | No published slab | Existing cooperative DS0/SZ0/A0/W0 writes; existing barrier | All waves see kb0/slab0 and metadata0; sums zero |
| Compute0 /slab1 prefetch | A=kb/slab0, W0=kb/slab0, current DS[h]/SZ[h] immutable | Unchanged slab1 raw loads and first two bundle calls | Current cacc contains K[0..63]; slab1 raw words private |
| B0 →publish1 | Every wave must stop reading old A | Existing B0; publish A/slab1,W1 using existing byte map | Exactly one writer per b64 destination; no consumer until B1 |
| B1 | A/W1 published; old W0 and consumed pf payload dead | Existing B1; **do not issue next-A before it** | Ready to consume kb/slab1; shared state stable |
| **New prefetch epoch** | Current planes remain read-only | If `!last`, issue all6 next-block loads (4 b64 +2 b32) into raw A_pf/W_pf/DS/header registers; no masks, conversion, DS stores or first use yet | Next payload in flight/private; current consumers unaffected |
| Compute1 | Reads A/slab1,W1 and current metadata | Existing last two bundle calls | cacc contains exactly ordered128-K integer sum |
| **Current fold, moved before B2a** | Current DS[h]/SZ[h] still published; next payload private | Exact unchanged rg/nb/j fold traversal; same RN DAG into persistent acc | Sum includes kb exactly once; cacc/current metadata consumption complete |
| B2a | All current slab1/metadata reads must finish | Existing full barrier; it may drain next loads now, **after useful compute+fold** | Old A and W0 safe to republish; all next payload needed locally is complete before use |
| Publish next | No remaining current LDS consumers | If `!last`, apply original masks/header conversion and publish A/slab0,W0,DS[h^1],SZ[h^1] with original producer map; only now re-point W toW0 | Shared next generation written, not yet readable |
| B2b | Next state written | Existing full publish barrier | Next iteration enters with consistent next A/W0/DS/SZ generation |
| Last trip /store | `last` forbids all kb+1 accesses | Fold current; execute B2a/B2b with no next writes; existing epilogue | Every sum final; LDS reuse safe; original guarded SET/ADD stores |

This uses no asynchronous producer role, no subset barrier, no flag ring, and no weakened fence. Every wave executes the same barrier sequence; `last` is workgroup-uniform. Metadata parity is still double-buffered; moving the fold earlier only shortens its lifetime. In particular next W writes must not be hoisted back above compute1 in a way that changes the selected W base, and next A must not overwrite current A before B2a.

Compiler scheduling is part of acceptance, not wishful comments. Preserve the existing bundle scheduling fences. If necessary, use at most narrowly placed existing `__builtin_amdgcn_sched_barrier(0)` fences to separate raw-load issuance, compute/fold and first-use/publication; no runtime wait or global barrier may be inserted between the new load epoch and its useful window. Defer cndmask/header conversion until the final segment. Inspect emitted ISA: if it still places `s_wait_loadcnt`/next-value masks before compute1/fold, the schedule has not been implemented, regardless of source order. Permit at most two source-scheduling forms, then abandon rather than adding an inline-assembly pipeline framework.

### 7.4 Exactness and architectural boundary

Weights/Xq are immutable for the launch; a raw load moved earlier returns the same byte at the same clamped address. Every nibble placement is the already-verified bijection, with unsigned weight/signed activation WMMA modifiers unchanged. Four K32 contributions remain in the same order, cacc restarts at each128-K boundary, and each output's folds remain ordered kb0,kb1,…,kb39. With15×8×128 the absolute integer sum bound is15360, well inside exact int32/FP32 conversion range. No arithmetic regrouping is authorized even where integer math would permit it.

Moving the entire fold relative to independent next-block loads does not change `RN(sc*d)`, `RN(t1*float(C))`, `RN(zp*d)`, `RN_FMA(t2,float(s),p)`, `RN_ADD(sum,term)`. It also does not commute two folds. SET output and the one residual ADD remain unchanged, including tails and nonzero initial Y. The analytical generation/byte-map proof in E passed; only the oracle/ISA/runtime gates can establish the compiled implementation's bit identity and synchronization.

The production route is exactgfx1201. This TU is also compiled under `__gfx1200__`; the same ordinary gfx12 schedule must compile there, but no performance or runtime validation on gfx1200 is claimed and no dispatch eligibility expands. gfx1100/gfx1151 use their separate iu4 implementation; gfx906/CDNA/non-gfx12 guards and stubs remain unchanged. No cache/replay/graph transaction schema changes: caller buffer lifetimes, launch arguments and stream ordering are unchanged. Rebuilt JIT/daemon artifacts are nevertheless required so retained source/HSACO caches cannot hide the candidate.

Explicit non-goals: quantizer or MQ4V2 format changes, one fold per256K, approximate arithmetic, fold VOPD redesign, wave-count/tile/store changes, global W streaming, A-plane double-buffering, novel barrier primitives, other architectures' tuning, decode implementation changes, new runtime switches and profiler infrastructure.

### 7.5 Independently executable composer slices

These are assignments for the implementing caller, not work performed by this planning agent. All skip shared builds/formatters/linters/project-wide tests; one GPU owner runs ordered gates.

**Slice A — sole kernel writer.** Own only KERNEL's body prefetch/publication/fold-placement regions. Implement the frozen10-dword next payload and transition table. Preserve ABI/geometry/planes/fold/store. Produce baseline/candidate source hashes, compile command and candidate TU/HSACO artifacts in scratch. Acceptance: exact state machine, no new switches or losing code left in production; emitted load epoch reaches the intended useful window.

**Slice B — independent CPU ISA/model review.** Own only scratch census/review artifacts; no kernel edits and no GPU launches. Immediately reproduce baseline616/307/32 and the full-tile1438 epilogue path, then consume A's frozen source/ISA hashes. Count both VOPD halves; inspect all used ADD instantiations, first/last paths, load-to-use distance, masks/waits, barriers, raw/allocated registers, metadata lifetime and unchanged store coordinates. Interface to A is artifact paths+hashes, not negotiated algorithm changes. Output an accept/reject report with instruction-address evidence; reviewer veto is binding.

**Slice C — oracle/acceptance preparation.** Own only scratch harness/configuration, no production source. Reuse ORACLE's unmodified CPU reference and seven-argument launcher; prepare current nine cases plus targeted K256/512/long-K parity wraps, independent half scales, nonzero Y/add and M/N tails for A's unchanged geometry. Record baseline recipes/cache identity and prepare randomized repeated runs to expose cross-wave skew. A/C can proceed independently; B can finish its baseline review concurrently. GPU execution consumes A/B's artifacts only after CPU review; no simultaneous ordinal1 timing jobs.

## 8. Ordered gates, abandon rule and cleanup

1. **Oracle:** existing9/9 cases, zero bit mismatches, repeated identical launches. Add/set and runtime-add wrapper must match their specialized entry; nonzero residual Y, last trip, at least two metadata-parity wraps, and M/N tails are mandatory. Use exact bits, not tolerance. Do not repin an oracle after a mismatch.
2. **Metadata:** same compiler/JIT flags, wave32, raw/allocated VGPR, SGPR, spill counts, private scratch, static/dynamic LDS and occupancy for all entries. Target≤192VGPR, hard stop>200, zero spill/scratch, unchanged16384/19456 LDS and256-thread geometry, occupancy≥baseline3 in the same HIP reporting unit. No guessed residency from a register formula.
3. **ISA census/legality:** hot non-last32WMMAs and exact388 fold scalar operations (or a separately proved identical-value CSE), same four barrier pairs and unmodified epilogue behavior. New next-A/W/metadata loads must issue afterB1 and before compute1; their first wait/consumer must be after useful compute/fold, not before. Count extra moves, masks, pointer arithmetic and scalar delays. Reject numerical DAG drift, early A overwrite, parity races or lower occupancy disguised by faster microbench timing.
4. **TIME same session:** exclusive ordinal1 coordinated with Iu4Frag; no launches on0/2/3. Interleave baseline/candidate/baseline/candidate, all five existingN512 rows, exact TU/HSACO cache identities, raw samples/medians, drift and clock/thermal context. Require reproducible gate/set≥3% improvement and no>2% reproducible regression on another row. A modeled forecast is not a measured win; no requirement to hit300us is inferred from the mismatched probe. Report achieved time and unchanged raw controls.
5. **Pins then integrated bench:** only after1–4 pass, caller rebuilds both CLI and daemon from the accepted source, records binary+JIT identities, and verifies unchanged chunk1 `032ebad84f2c1dd5e2980fa805215ac0`, chunk2 `dc7e53181662271780374f0a85fd7732`, chunk24 `0821993b56021caf4505c1dc4a6b9990`, and KLD **0.063410** under the established recipes. Then interleaved OFF/ON bench at the established four prefill lengths, retaining raw JSON; no best-row selection. One gate/set matrix is not a fused gate+up benchmark.
6. **Separate decode, then serve:** fresh decode A/B bracket separate from long-prefill thermal history; do not infer unaffected decode from source scope. Run the established serve battery with rebuilt daemon, all turns, prefill/decode measurements, termination and repetition/attractor/empty/retrieval/stream-error checks. Existing baseline failures are reported separately, never hidden or used to waive a new regression.
7. **Final reviewer veto and clean cutover:** caller owns shared validation. Reviewers can reject even a timing win for state/numerical defects. Remove only losing production scaffolding/new experiment switches and throwaway integration scripts after proof; retain scratch evidence and update this plan with the actual outcome. Host/API/oracle reference code and other architectures are intentionally unchanged.

**Abandon criterion:** stop immediately on any bit drift, race, scratch/spill, residency loss, register cap breach or wrong emitted ordering. After at most two narrowly different source schedules, if load-use distance is still not materially enlarged, stop without GPU performance claims. If the intended ISA is achieved but same-session TIME gains<3% or any other row reproducibly loses>2%, reject this candidate. A no-stage/no-barrier upper bound, SQ_BUSY change, prettier source comments or the277-TOPS probe alone cannot justify continued production work.
